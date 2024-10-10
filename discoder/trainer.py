import os

import torch
from torch import nn, optim
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from discoder import losses, utils, evaluate, meldataset
from discoder.models import DDP, DACEncoder


class Trainer:

    def __init__(
            self,
            vocoder: nn.Module,
            dac_encoder: DACEncoder,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler.LRScheduler,
            config: dict,
            training_step: int,
            rank: int,
            use_multi_gpu: bool = False,
            discriminator: nn.Module = None,
            disc_optimizer: optim.Optimizer = None,
            disc_scheduler: optim.lr_scheduler.LRScheduler = None) -> None:

        if use_multi_gpu:
            self.vocoder = DDP(vocoder.to(rank), device_ids=[rank])
        else:
            self.vocoder = vocoder.to(rank)

        self.dac_encoder = dac_encoder.to(rank)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.training_step = training_step
        self.rank = rank
        self.use_multi_gpu = use_multi_gpu
        self.highest_visqol = 0
        self.unfrozen_decoder = False
        # losses
        self.waveform_loss = losses.WaveformLoss().to(rank)
        self.ms_stft_loss = losses.MultiScaleSTFTLoss().to(rank)
        self.ms_mel_loss = losses.MultiScaleMelSpectrogramLoss(config["sample_rate"]).to(rank)
        # distributed
        self.is_main = self.rank == 0

        # discriminator
        if config["use_discriminator"]:
            if use_multi_gpu:
                self.disc = DDP(discriminator.to(rank), device_ids=[rank])
            else:
                self.disc = discriminator.to(rank)
            self.disc_optimizer = disc_optimizer
            self.adv_loss = losses.GANLoss(disc=self.disc)
            self.disc_scheduler = disc_scheduler
            if self.is_main and self.config["wandb"]["mode"] != "disabled":
                wandb.watch(self.disc, log_freq=config["batch_grad_log"])

        self.last_vocoder_path = self.last_disc_path = None
        self.send_ground_truth = True

        if self.is_main and self.config["wandb"]["mode"] != "disabled":
            wandb.watch(self.vocoder, log_freq=config["batch_grad_log"])

    def _run_batch(self, mel: torch.Tensor, audio: torch.Tensor) -> None:
        custom_out, skip = self.vocoder.encode(mel)

        # get targets
        target_z, target_codes, target_latents, _, _ = self.dac_encoder(audio)
        target_quant_latents = self.dac_encoder.quantize_latents(target_latents)

        loss = {}
        pred_type = self.config["model"]["predict_type"]
        if pred_type == "quant_latents":
            loss["loss_quant_latents"] = F.l1_loss(input=custom_out, target=target_quant_latents)
            custom_z = self.vocoder.project_quant_latents(custom_out)
        elif pred_type == "z":
            loss["loss_z"] = F.l1_loss(input=custom_out, target=target_z)
            custom_z = custom_out
        else:
            print(f"Undefined predict_type {pred_type}!")
            exit(0)

        custom_recon = self.vocoder.decode(custom_z, skip).squeeze(dim=1)  # [B, segment_size]

        # discriminator loss
        sep_loss = {}  # separate losses for logging
        if self.config["use_discriminator"]:
            sep_loss["loss_adv_disc_mpd"], sep_loss["loss_adv_disc_mrd"] = self.adv_loss.discriminator_loss(fake=custom_recon.unsqueeze(dim=1), real=audio)
            sep_loss["loss_adv_disc"] = sep_loss["loss_adv_disc_mpd"] + sep_loss["loss_adv_disc_mrd"]
            self.disc_optimizer.zero_grad()
            sep_loss["loss_adv_disc"].backward()
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1000.)
            self.disc_optimizer.step()

            sep_loss["loss_adv_gen_mpd"], sep_loss["loss_adv_feat_mpd"], sep_loss["loss_adv_gen_mrd"], sep_loss["loss_adv_feat_mrd"] = self.adv_loss.generator_loss(fake=custom_recon.unsqueeze(dim=1), real=audio)
            loss["loss_adv_gen"] = sep_loss["loss_adv_gen_mpd"] + sep_loss["loss_adv_gen_mrd"]
            loss["loss_adv_feat"] = sep_loss["loss_adv_feat_mpd"] + sep_loss["loss_adv_feat_mrd"]

        # encoder losses
        loss["loss_ms_mel"] = self.ms_mel_loss(x=custom_recon, y=audio.squeeze())
        loss["loss_ms_stft"] = self.ms_stft_loss(x=custom_recon, y=audio.squeeze())
        loss["loss_waveform"] = self.waveform_loss(x=custom_recon, y=audio.squeeze())

        total_loss = torch.tensor(0., device=self.rank)
        for loss_key, loss_value in loss.items():
            if not loss_key.startswith("loss_adv_disc"):
                total_loss += loss[loss_key] * self.config["loss_multiplier"][loss_key]

        # optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.vocoder.parameters(), 1000.)
        self.optimizer.step()

        # logging
        step_log_dict = {**loss, **sep_loss, "loss": total_loss.item(), "training_step": self.training_step, "grad_norm": grad_norm}
        if self.config["use_discriminator"]:
            step_log_dict["disc_grad_norm"] = disc_grad_norm
        self._log(step_log_dict)
        self.training_step += 1

    def _run_epoch(self, epoch: int) -> None:
        bs = len(next(iter(self.train_dataloader))[0])
        print(f"[GPU {self.rank}, {self.training_step}]: epoch {epoch}, batch size {bs}")
        if self.use_multi_gpu:
            self.train_dataloader.sampler.set_epoch(epoch)

        for mel, audio, audio_path in tqdm(self.train_dataloader, disable=(not self.is_main), desc="Iteration"):
            if self.training_step >= self.config["unfreeze"]["steps"] and not self.unfrozen_decoder:
                self._unfreeze_decoder()

            mel = mel.to(self.rank)
            audio = audio.to(self.rank)
            self._run_batch(mel, audio)

            if self.is_main and self.training_step % self.config["step_media_log"] == 0:
                self._validate(epoch)

            if self.is_main and (self.training_step > 0) and (self.training_step % self.config["step_checkpoint"] == 0):
                self._save_checkpoint(epoch)

        # lr schedulers
        self.scheduler.step()
        if self.config["use_discriminator"]:
            self.disc_scheduler.step()

    def train(self):
        if self.is_main:
            print(f"Starting from wandb step {wandb.run.step} and training step {self.training_step}...")
        self.vocoder.train()

        start_epoch = 0
        if self.scheduler.last_epoch > 0:
            start_epoch = self.scheduler.last_epoch

        for epoch in tqdm(range(start_epoch, self.config["n_epochs"]), disable=(not self.is_main), desc="Epoch", initial=start_epoch):
            self._run_epoch(epoch)

    def _validate(self, epoch):
        mel_transform_config = {
            "n_fft": self.config["mel"]["n_fft"],
            "num_mels": self.config["mel"]["n_mels"],
            "sampling_rate": self.config["sample_rate"],
            "hop_size": self.config["mel"]["hop_length"],
            "win_size": self.config["mel"]["win_length"],
            "fmin": self.config["mel"]["f_min"],
            "fmax": self.config["mel"]["f_max"]
        }

        torch.cuda.empty_cache()
        self.vocoder.eval()

        def get_metric_name(dataset_type: str, metric_name: str):
            if dataset_type == "Jamendo":
                return metric_name
            else:
                return metric_name + " (Libri)"

        with torch.no_grad():
            metrics = {
                "Jamendo": {
                    "moslqos": [],
                    "mel_distances": [],
                    "mstft_distances": []
                },
                "LibriTTS": {
                    "moslqos": [],
                    "mel_distances": [],
                    "mstft_distances": []
                }
            }

            counter = 0  # number of audio/mel-spectrograms to send to wandb
            for mel, audio_clip, audio_paths in self.val_dataloader:
                sample_type = "LibriTTS" if "LibriTTS" in audio_paths[0] else "Jamendo"
                audio_clip = audio_clip.to(self.rank)
                mel = mel.to(self.rank)

                audio_recon = utils.reconstruct_audio_from_mel(
                    vocoder=self.vocoder, mel=mel, predict_type=self.config["model"]["predict_type"]
                )
                mel_recon = meldataset.mel_spectrogram(y=audio_recon, **mel_transform_config)

                # ViSQOL
                metrics[sample_type]["moslqos"].append(evaluate.visqol_score(
                    input=audio_recon, target=audio_clip.squeeze(dim=1), sample_rate=self.config["sample_rate"]
                ))
                metrics[sample_type]["mel_distances"].append(evaluate.mel_distance(input=mel_recon, target=mel))
                metrics[sample_type]["mstft_distances"].append(self.ms_stft_loss(audio_recon, audio_clip.squeeze(dim=1)))

                # logging
                caption = f"name: {os.path.basename(audio_paths[0])}"
                audio_args = {"caption": caption, "sample_rate": self.config["sample_rate"]}
                if counter < 5:
                    if self.send_ground_truth:
                        self._log({
                            get_metric_name(sample_type, "True Audio"): wandb.Audio(audio_clip[0].squeeze().detach().cpu(), **audio_args),
                            get_metric_name(sample_type, "True Mel-Spectrogram"): wandb.Image(
                                utils.get_fig_from_spectrogram(spec=mel[0].detach().cpu()),
                                caption=caption)
                        }, commit=False)

                    self._log({
                        get_metric_name(sample_type, "Custom Generated Audio"): wandb.Audio(audio_recon[0].detach().cpu(), **audio_args),
                        get_metric_name(sample_type, "Custom Generated Mel-Spectrogram"): wandb.Image(
                            utils.get_fig_from_spectrogram(spec=mel_recon[0].detach().cpu()),
                            caption=caption),
                    })
                counter += 1

            self._log({
                "Avg. ViSQOL Score": sum(metrics["Jamendo"]["moslqos"]) / len(metrics["Jamendo"]["moslqos"]),
                "Avg. Mel Distance": sum(metrics["Jamendo"]["mel_distances"]) / len(metrics["Jamendo"]["mel_distances"]),
                "Avg. Mstft Distance": sum(metrics["Jamendo"]["mstft_distances"]) / len(metrics["Jamendo"]["mstft_distances"])
            }, print_data=True)

            self._log({
                "Avg. ViSQOL Score (LibriTTS)": sum(metrics["LibriTTS"]["moslqos"]) / len(metrics["LibriTTS"]["moslqos"]),
                "Avg. Mel Distance (LibriTTS)": sum(metrics["LibriTTS"]["mel_distances"]) / len(metrics["LibriTTS"]["mel_distances"]),
                "Avg. Mstft Distance (LibriTTS)": sum(metrics["LibriTTS"]["mstft_distances"]) / len(metrics["LibriTTS"]["mstft_distances"])
            }, print_data=True)

        self.vocoder.train()
        self.send_ground_truth = False

        # save best visqol scores
        if self.is_main and (sum(metrics["Jamendo"]["moslqos"]) / len(metrics["Jamendo"]["moslqos"])) > self.highest_visqol:
            self.highest_visqol = sum(metrics["Jamendo"]["moslqos"]) / len(metrics["Jamendo"]["moslqos"])
            print("Highest ViSQOL Score:", self.highest_visqol)
            print(f"Saving checkpoint for training step {self.training_step} and epoch {epoch}")
            self._save_checkpoint(epoch)


    def _unfreeze_decoder(self):
        print("Unfreezing decoder...")
        self.vocoder.unfreeze_decoder()  # unfreeze decoder
        self.dac_encoder.freeze()  # freeze encoder and quantizer

        if "unfreeze_quantizer" in self.config["unfreeze"] and self.config["unfreeze"]["unfreeze_quantizer"]:
            print("Unfreezing quantizer...")
            self.vocoder.unfreeze_quantizer()

        # update loss multiplier
        print(f"Changing loss multipliers from {self.config['loss_multiplier']}")
        self.config["loss_multiplier"] = self.config["unfreeze"]["loss_multiplier"]
        print(f"to {self.config['loss_multiplier']}")

        self.unfrozen_decoder = True

    def _save_checkpoint(self, epoch: int):
        # save encoder
        print("Saving checkpoint...")
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"e{epoch}_step{self.training_step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        vocoder_path = os.path.join(checkpoint_path, f"encoder_e{epoch}_step{self.training_step}.pt")
        torch.save({
            "epoch": epoch,
            "training_step": self.training_step,
            "model_state_dict": self.vocoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, vocoder_path)

        # Prepare artifact
        model_artifact = wandb.Artifact(name=f"{wandb.run.name}", type="model", metadata=dict(self.config))
        model_artifact.add_file(vocoder_path)

        if self.config["use_discriminator"]:
            # save discriminator
            disc_path = os.path.join(checkpoint_path, f"disc_e{epoch}_step{self.training_step}.pt")
            torch.save({
                "epoch": epoch,
                "training_step": self.training_step,
                "model_state_dict": self.disc.state_dict(),
                "optimizer_state_dict": self.disc_optimizer.state_dict()
            }, disc_path)
            model_artifact.add_file(disc_path)

        wandb.log_artifact(model_artifact)
        # Cleanup
        if self.last_vocoder_path is not None:
            try:
                os.remove(self.last_vocoder_path)
            except Exception:
                print("Failed to remove last_encoder_path: ", self.last_vocoder_path)
        self.last_vocoder_path = vocoder_path

        if self.config["use_discriminator"]:
            if self.last_disc_path is not None:
                try:
                    os.remove(self.last_disc_path)
                except Exception:
                    print("Failed to remove last_disc_path: ", self.last_disc_path)
            self.last_disc_path = disc_path

        torch.cuda.empty_cache()

    def _log(self, data, commit=True, print_data=False):
        if self.is_main:
            wandb.log(data, commit=commit)
            if print_data:
                print(data)
