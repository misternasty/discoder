import os
import typing as t
import json

import torch
import matplotlib.pyplot as plt
import torchaudio as ta

from discoder import meldataset


def get_devices(print_info=False):
    """Gets the available devices and optionally prints CUDA info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_devices = torch.cuda.device_count()
    if print_info:
        print(f"Using {device} with {num_devices} GPUs....")
        for dev in range(torch.cuda.device_count()):
            print(f"[Rank {dev}] {torch.cuda.get_device_name(dev)}")
    return device, num_devices


def read_config(config_path: str):
    """Reads and returns config from config_path."""
    with open(config_path, "r") as json_file:
        return json.load(json_file)


def sample_rate_str(sample_rate: int) -> t.Optional[str]:
    """
    Return string representation of sample rate that matches the possible values of the DAC model (44khz, 24khz, or 16khz).
    :param sample_rate:
    :return: str representation of sample rate
    """
    if sample_rate == 44100:
        return "44khz"
    elif sample_rate == 24000:
        return "24khz"
    elif sample_rate == 16000:
        return "16khz"
    else:
        return None


def get_fig_from_spectrogram(spec: torch.tensor, transform_to_db=False, show_info=True):
    fig, ax = plt.subplots()
    if transform_to_db:
        spec = ta.transforms.AmplitudeToDB()(spec).detach().cpu().numpy()
    else:
        spec = spec.detach().cpu().numpy()
    plt.imshow(spec, cmap='magma', origin='lower', aspect="auto")
    if show_info:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.axis('off')
    plt.close()
    return fig


def reconstruct_audio_from_mel(vocoder, mel: torch.Tensor, predict_type: str):
    custom_out, skip = vocoder.encode(mel)

    if predict_type == "quant_latents":
        custom_z = vocoder.project_quant_latents(custom_out)
    elif predict_type == "z":
        custom_z = custom_out
    else:
        print(f"Undefined predict_type {predict_type}!")
        exit(0)

    return vocoder.decode(custom_z[0].unsqueeze(0), skip).squeeze(dim=1).detach()


def get_dataloader_inference(input_dir: str, config: dict) -> torch.utils.data.DataLoader:
    """Returns the dataloader for inference."""
    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    dataset = meldataset.MelDataset(
        training_files=input_files,
        hparams=None,
        n_fft=config["mel"]["n_fft"],
        num_mels=config["mel"]["n_mels"],
        hop_size=config["mel"]["hop_length"],
        win_size=config["mel"]["win_length"],
        sampling_rate=config["sample_rate"],
        fmin=config["mel"]["f_min"],
        fmax=config["mel"]["f_max"],
        n_cache_reuse=0,
        shuffle=False,
        split=False,
        pad_hop_size=True
    )
    return torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, pin_memory=True)


def get_mel_spectrogram_from_config(audio: torch.Tensor, config: dict) -> torch.Tensor:
    return meldataset.mel_spectrogram(
        y=audio, n_fft=config["mel"]["n_fft"], num_mels=config["mel"]["n_mels"], sampling_rate=config["sample_rate"],
        hop_size=config["mel"]["hop_length"], win_size=config["mel"]["win_length"], fmin=config["mel"]["f_min"],
        fmax=config["mel"]["f_max"], center=False
    )
