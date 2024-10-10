# Adapted from https://github.com/NVIDIA/BigVGAN and https://github.com/jik876/hifi-gan

import os
import math
import random
import pathlib

import torch
import torch.utils.data
import numpy as np
import torchaudio as ta
from librosa.filters import mel as librosa_mel_fn
from audiotools.data import transforms
from scipy.io.wavfile import read
import pyloudnorm as pyln
from tqdm import tqdm


def load_wav(full_path: str, sr_target: int, resample=False, normalize=False):
    sampling_rate, data = read(full_path)

    if resample:
        data = ta.functional.resample(waveform=torch.tensor(data).float(), orig_freq=sampling_rate, new_freq=sr_target).numpy()
        if np.squeeze(data).ndim > 1:
            data = np.mean(data, axis=1)
    else:
        assert len(data.shape) == 1 and sampling_rate == sr_target, f"Error with {full_path}, {data.shape}, {sampling_rate}, {sr_target}"

    if normalize:
        data = normalize_custom(data=data)
        data = ensure_max_of_audio(data=data)

    return data, sr_target


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)



def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

GAIN_FACTOR = np.log(10) / 20
meter = pyln.Meter(rate=44100)


def normalize_custom(data: np.ndarray, db = -24.0):
    """Based on DAC"""
    ref_db = meter.integrated_loudness(data.astype(np.float32))
    if np.isinf(ref_db):
        # return data
        ref_db = 0

    gain = db - ref_db
    gain = np.exp(gain * GAIN_FACTOR)

    return data * gain

def ensure_max_of_audio(data: np.ndarray, maxim: float = 1.0):
    """Based on DAC"""
    peak = np.abs(data).max()
    if peak > maxim:
        return data * (maxim / peak)

    return data


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, hparams, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, segment_size=16384, split=True, shuffle=True, n_cache_reuse=1,
                 fmax_loss=None, fine_tuning=False, base_mels_path=None, is_seen=True, validation=False, pad_hop_size=False):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams = hparams
        self.is_seen = is_seen
        if self.is_seen:
            self.name = pathlib.Path(self.audio_files[0]).parts[0]
        else:
            self.name = '-'.join(pathlib.Path(self.audio_files[0]).parts[:2]).strip("/")

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.cached_filename = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.validation = validation
        self.pad_hop_size = pad_hop_size

        self.pre_transform = transforms.Compose([transforms.VolumeNorm(), transforms.RescaleAudio()])
        self.pre_transform_args = self.pre_transform.instantiate()

        print("INFO: checking dataset integrity...")
        for i in tqdm(range(len(self.audio_files))):
            assert os.path.exists(self.audio_files[i]), "{} not found".format(self.audio_files[i])

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            if self.pad_hop_size:
                remainder = audio.shape[0] % self.hop_size
                if remainder > 0:
                    audio = np.pad(audio, (0, self.hop_size - remainder), mode='constant')

            if not self.fine_tuning:
                with torch.no_grad():
                    if audio.shape[-1] < int(0.41 * self.sampling_rate):  # for pyloud norm, otherwise exception
                        print(f"PADDING FOR {filename}")
                        audio = np.pad(audio, (0, int(0.41 * self.sampling_rate) - audio.shape[-1]), 'constant')

                    audio = normalize_custom(data=audio)
                    audio = ensure_max_of_audio(data=audio)

            self.cached_wav = audio
            self.cached_filename = filename
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    if self.validation:
                        audio_start = 0
                    else:
                        max_audio_start = audio.size(1) - self.segment_size
                        audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

                mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                      center=False)
            else: # validation step
                # match audio length to self.hop_size * n for evaluation
                if (audio.size(1) % self.hop_size) != 0:
                    audio = audio[:, :-(audio.size(1) % self.hop_size)]
                mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                      self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                      center=False)
                assert audio.shape[1] == mel.shape[2] * self.hop_size, "audio shape {} mel shape {}".format(audio.shape, mel.shape)

        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return mel.squeeze(), audio, filename

    def __len__(self):
        return len(self.audio_files)
