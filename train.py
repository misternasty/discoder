import argparse

import torch
import os
import wandb
from torch import optim
import dac
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from discoder.discriminator import Discriminator
from discoder.models import DACEncoder, DisCoder
from discoder import utils
from discoder.meldataset import MelDataset
from discoder.trainer import Trainer


os.environ["WANDB__SERVICE_WAIT"] = "300"


def get_dataloaders(config: dict, use_multi_gpu: bool):
    with open(config["train_datafile"]) as file:
        train_files = [os.path.join(os.path.dirname(config["train_datafile"]), line.rstrip()) for line in file]

    train_dataset = MelDataset(
        training_files=train_files,
        hparams=None,
        segment_size=config["segment_size"],
        n_fft=config["mel"]["n_fft"],
        num_mels=config["mel"]["n_mels"],
        hop_size=config["mel"]["hop_length"],
        win_size=config["mel"]["win_length"],
        sampling_rate=config["sample_rate"],
        fmin=config["mel"]["f_min"],
        fmax=config["mel"]["f_max"],
        n_cache_reuse=config["n_cache_reuse"],
        shuffle=False,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True
    )
    train_sampler = DistributedSampler(train_dataset) if use_multi_gpu else None
    train_dataloader = DataLoader(train_dataset, shuffle=False, num_workers=config["num_workers"],
                                  sampler=train_sampler, persistent_workers=True,
                                  prefetch_factor=config["prefetch_factor"], drop_last=True,
                                  batch_size=config["batch_size"], pin_memory=True)

    with open(config["validation_datafile"]) as file:
        val_files = [os.path.join(os.path.dirname(config["validation_datafile"]), line.rstrip()) for line in file]

    val_dataset = MelDataset(
        training_files=val_files,
        hparams=None,
        segment_size=config["segment_size_val"],
        n_fft=config["mel"]["n_fft"],
        num_mels=config["mel"]["n_mels"],
        hop_size=config["mel"]["hop_length"],
        win_size=config["mel"]["win_length"],
        sampling_rate=config["sample_rate"],
        fmin=config["mel"]["f_min"],
        fmax=config["mel"]["f_max"],
        n_cache_reuse=0,
        shuffle=False,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
        is_seen=True,
        validation=True
    )
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, pin_memory=True, drop_last=True)

    return train_dataloader, val_dataloader


def init_wandb(config: dict):
    """Initializes wandb."""
    resume_config = {}
    if (config["wandb"]["mode"] != "disabled") and (config["wandb"]["checkpoint"] is not None and not config["wandb"]["fork_checkpoint"]):
        api = wandb.Api()
        resume_run_id = api.artifact(f"{config['wandb']['wandb_prefix']}/{config['wandb']['checkpoint']}").logged_by().id
        resume_config = {"resume": "must", "id": resume_run_id}

    print(f"Using wandb dir {config['wandb']['dir']}")
    wandb.init(project=config["wandb"]["project"], mode=config["wandb"]["mode"], job_type="train", dir=config["wandb"]["dir"], config=config, **resume_config)
    wandb.run.log_code(".", include_fn=lambda path, root: path.endswith(".ipynb") or path.endswith(".py"))


def init_models(config: dict, rank):
    """Initializes DAC decoder, custom encoder and discriminator."""
    print("Init models...")
    # DAC
    model_path = dac.utils.download(model_type=utils.sample_rate_str(config["sample_rate"]))
    dac_model = dac.DAC.load(str(model_path))
    for param in dac_model.parameters():
        param.requires_grad = False
    dac_model.eval()
    dac_data = torch.load(model_path, map_location=f"cuda:{rank}")

    # Init quantizer
    quantizer = dac.nn.quantize.ResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=dac_data["metadata"]["kwargs"]["n_codebooks"],
        codebook_size=dac_data["metadata"]["kwargs"]["codebook_size"],
        codebook_dim=dac_data["metadata"]["kwargs"]["codebook_dim"],
        quantizer_dropout=dac_data["metadata"]["kwargs"]["quantizer_dropout"]
    )
    dac_encoder_quantizer = dac_data["state_dict"]
    quantizer_data = {k.removeprefix("quantizer."):v for k,v in dac_encoder_quantizer.items() if k.startswith("quantizer.")}
    quantizer.load_state_dict(quantizer_data)

    for param in quantizer.parameters():
        param.requires_grad = False

    dac_encoder = DACEncoder(
        dac_model=dac_model,
        n_codebooks=config["model"]["n_codebooks"],
        codebook_dim=config["model"]["codebook_dim"]
    ).to(rank)

    # DisCoder
    vocoder = DisCoder(
        config=config,
        dac_decoder=dac_model.decoder,
        dac_encoder_quantizer=None if config["model"]["predict_type"] == "z" else quantizer
    ).to(rank)

    # Optimizer
    optim_params = {"lr": config["learning_rate"], "betas": (config["adam_b1"], config["adam_b2"])}
    optimizer = optim.AdamW(vocoder.parameters(), **optim_params)

    # Discriminator and Optimizer
    disc = disc_optimizer = disc_scheduler = None
    if config["use_discriminator"]:
        disc = Discriminator(
            periods=config["disc"]["periods"],
            discriminator_channel_mult=config["disc"]["discriminator_channel_mult"],
            use_spectral_norm=config["disc"]["use_spectral_norm"]
        ).to(rank)
        disc_optimizer = optim.AdamW(disc.parameters(), **optim_params)

    # check if wandb or local checkpoints should be used
    if (config["wandb"]["mode"] != "disabled" and config["wandb"]["checkpoint"]) or (config["local"]["checkpoint_model"]):
        # wandb
        if config["wandb"]["mode"] != "disabled" and config["wandb"]["checkpoint"]:
            if rank == 0:
                artifact = wandb.run.use_artifact(config["wandb"]["checkpoint"])
            else:
                api = wandb.Api()
                artifact = api.artifact(f"{config['wandb']['wandb_prefix']}/{config['wandb']['checkpoint']}")
            datadir = artifact.download(config["wandb"]["dir"] + "/artifacts")
            model_path = os.path.join(datadir, next(f for f in os.listdir(datadir) if f.startswith("encoder_")))
            disc_path = os.path.join(datadir, next(f for f in os.listdir(datadir) if f.startswith("disc_")))

        elif config["local"]["checkpoint_model"]:
            print("Using local checkpoint...")
            model_path = config["local"]["checkpoint_model"]
            disc_path = config["local"]["checkpoint_discriminator"]
        else:
            print("Invalid checkpoint config...")
            exit(1)

        # load encoder and optimizer
        vocoder_data = torch.load(model_path, map_location=f"cuda:{rank}")
        vocoder_state_dict = {k.replace("module.", ""): v for k,v in vocoder_data["model_state_dict"].items()}
        try:
            vocoder.load_state_dict(vocoder_state_dict, strict=False)
            optimizer.load_state_dict(vocoder_data["optimizer_state_dict"])
        except Exception as e:
            print(e)
            print("Failed to load model, exiting...")
            exit(1)

        # optionally load discriminator and discriminator optimizer
        if config["use_discriminator"]:
            disc_data = torch.load(disc_path, map_location=f"cuda:{rank}")
            disc.load_state_dict( {k.replace("module.", ""): v for k, v in disc_data["model_state_dict"].items()})
            disc_optimizer.load_state_dict(disc_data["optimizer_state_dict"])

        # get last epoch and last training_step
        last_epoch = vocoder_data["epoch"]
        training_step = vocoder_data["training_step"]
        print(f"Continuing training from epoch {last_epoch} and training step {training_step}")
    else:
        last_epoch = -1
        training_step = 0
        print("Training from scratch...")

    # schedulers
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_gamma"], last_epoch=last_epoch)
    if config["use_discriminator"]:
        disc_scheduler = optim.lr_scheduler.ExponentialLR(disc_optimizer, gamma=config["exp_gamma"], last_epoch=last_epoch)

    return vocoder, dac_encoder, optimizer, scheduler, dac_model, disc, disc_optimizer, disc_scheduler, training_step


def ddp_setup(rank, world_size, port: int):
    """Sets up distributed training.

    rank: unique identifier for each process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def process(rank: int, world_size: int, config: dict):
    use_multi_gpu = (world_size > 1)
    if use_multi_gpu:
        ddp_setup(rank=rank, world_size=world_size, port=config["backend"]["master_port"])

    if rank == 0:
        init_wandb(config)

    train_dataloader, val_dataloader = get_dataloaders(config=config, use_multi_gpu=use_multi_gpu)
    vocoder, dac_encoder, optimizer, scheduler, dac_model, disc, disc_optimizer, disc_scheduler, training_step = init_models(config=config, rank=rank)
    trainer = Trainer(
        vocoder=vocoder,
        dac_encoder=dac_encoder,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        training_step=training_step,
        rank=rank,
        use_multi_gpu=use_multi_gpu,
        discriminator=disc,
        disc_optimizer=disc_optimizer,
        disc_scheduler=disc_scheduler
    )
    trainer.train()
    if use_multi_gpu:
        destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    # init config
    device, world_size = utils.get_devices(print_info=True)
    config = utils.read_config(config_path=args.config)
    torch.manual_seed(config["seed"])
    print("Using config")
    print(config)

    if world_size > 1:
        mp.spawn(process, args=(world_size, config), nprocs=world_size)
    else:
        process(rank=0, world_size=world_size, config=config)


if __name__ == "__main__":
    main()
