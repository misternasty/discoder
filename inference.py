import argparse
import json
import os

import dac
import librosa
import numpy as np
import torch
import pyloudnorm as pyln
import torchaudio as ta
from discoder import utils, meldataset
from discoder.models import DisCoder


def inference(args, encoder: torch.nn.Module, config: dict, device, normalize_volume: bool):
    filelist = os.listdir(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    meter = pyln.Meter(rate=44100)
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            print(os.path.join(args.input_dir, filename))
            wav_o, sr = meldataset.load_wav(os.path.join(args.input_dir, filename), sr_target=config["sample_rate"], resample=True)
            wav = meldataset.normalize_custom(data=wav_o)
            wav = meldataset.ensure_max_of_audio(data=wav)
            wav = torch.FloatTensor(wav).unsqueeze(dim=0)

            # pad with segment size of 16384
            to_pad = config["segment_size"] - (wav.shape[-1] % config["segment_size"])
            wav = torch.nn.functional.pad(wav, pad=(0, to_pad), mode="constant", value=0)
            assert wav.shape[-1] % config["segment_size"] == 0

            mel = utils.get_mel_spectrogram_from_config(wav, config).to(device)
            wav_recon = utils.reconstruct_audio_from_mel(vocoder=encoder, mel=mel, predict_type=config["model"]["predict_type"])
            wav_recon = wav_recon[..., 0:-to_pad]

            # normalize volume
            if normalize_volume:
                audio_org, _ = librosa.load(os.path.join(args.input_dir, filename), sr=44100)
                loudness = meter.integrated_loudness(audio_org.astype(np.float32))
                wav_recon = torch.tensor(meldataset.normalize_custom(wav_recon.squeeze().cpu().numpy(), db=loudness)).unsqueeze(dim=0).to(device)

            # save file
            audio_name, audio_ext = os.path.splitext(os.path.basename(filename))
            wav_recon_filename = f"{audio_name}_generated{audio_ext}"
            audio_recon_path = os.path.join(args.output_dir, wav_recon_filename)
            ta.save(audio_recon_path, src=wav_recon.cpu(), sample_rate=config["sample_rate"], bits_per_sample=16, encoding="PCM_S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_file")
    parser.add_argument("--config_file")
    parser.add_argument("--normalize_volume", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device, num_devices = utils.get_devices(print_info=False)

    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    dac_model = dac.DAC.load(str(dac.utils.download(model_type=utils.sample_rate_str(44100))))
    encoder = DisCoder(
        config=config,
        dac_decoder=dac_model.decoder,
        dac_encoder_quantizer=dac_model.quantizer
    )
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict["model_state_dict"].items()}
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    encoder = encoder.to(device)
    torch.manual_seed(config["seed"])

    model_path = dac.utils.download(model_type=utils.sample_rate_str(config["sample_rate"]))
    dac_model = dac.DAC.load(model_path)
    for param in dac_model.parameters():
        param.requires_grad = False
    dac_model.eval()

    inference(args, encoder, config, device, args.normalize_volume)



if __name__ == '__main__':
    main()