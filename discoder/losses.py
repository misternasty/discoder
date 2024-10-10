import typing
from typing import List

import torch
from torch import nn

from discoder.meldataset import mel_spectrogram


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
            self,
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            window_type: str = None,
    ):
        super().__init__()
        self.window_lengths = window_lengths
        self.hop_lengths = [w // 4 for w in window_lengths]
        self.window_type = window_type

        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for window_length, hop_length in zip(self.window_lengths, self.hop_lengths):
            x_stft = x.stft(window_length, hop_length, self.window_type, return_complex=True)
            y_stft = y.stft(window_length, hop_length, self.window_type, return_complex=True)

            x_stft = torch.view_as_real(x_stft)
            y_stft = torch.view_as_real(y_stft)
            x_mag = torch.norm(x_stft, p=2, dim=-1)
            y_mag = torch.norm(y_stft, p=2, dim=-1)

            loss += self.log_weight * self.loss_fn(
                x_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag, y_mag)

        return loss


class MultiScaleMelSpectrogramLoss(nn.Module):
    """
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec
    """
    def __init__(self,
                 sampling_rate: int,
                 num_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
                 window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
                 loss_fn: typing.Callable = nn.L1Loss(),
                 fmin: int = 0,
                 fmax: int = None,
                 clamp_eps: float = 1e-5,
                 mag_weight: float = 1.0,
                 log_weight: float = 1.0,
                 pow: float = 2.0 ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.window_lengths = window_lengths
        self.hop_lengths = [w // 4 for w in window_lengths]
        self.fmin = fmin
        self.fmax = fmax

        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.pow = pow

        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for num_mels, window_length, hop_length in zip(self.num_mels, self.window_lengths, self.hop_lengths):
            x_mels = mel_spectrogram(y=x, n_fft=window_length, num_mels=num_mels, sampling_rate=self.sampling_rate,
                                     hop_size=hop_length, win_size=window_length, fmin=self.fmin, fmax=self.fmax,
                                     center=False)
            y_mels = mel_spectrogram(y=y, n_fft=window_length, num_mels=num_mels, sampling_rate=self.sampling_rate,
                                     hop_size=hop_length, win_size=window_length, fmin=self.fmin, fmax=self.fmax,
                                     center=False)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10()
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)

        return loss


class WaveformLoss(nn.Module):
    def __init__(self, loss_fn: typing.Callable = nn.L1Loss()):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x, y)


class GANLoss(nn.Module):
    """
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec

    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, disc):
        super().__init__()
        self.disc = disc

    def forward(self, fake, real):
        y_d_fake, fmap_d_fake = self.disc(fake)
        y_d_real, fmap_d_real = self.disc(real)
        return y_d_fake, y_d_real, fmap_d_fake, fmap_d_real

    def discriminator_loss(self, fake, real):
        loss_mpd = self._disc_type_loss(fake, real, self.disc.compute_mpds)
        loss_mrd = self._disc_type_loss(fake, real, self.disc.compute_mrds)
        return loss_mpd, loss_mrd

    def _disc_type_loss(self, fake, real, disc_type):
        y_disc_type_fake, _ = disc_type(fake.clone().detach())
        y_disc_type_real, _ = disc_type(real)

        loss = 0
        for y_fake, y_real in zip(y_disc_type_fake, y_disc_type_real):
            loss += torch.mean(y_fake ** 2)
            loss += torch.mean((1 - y_real) ** 2)

        return loss

    def generator_loss(self, fake, real):
        feature_loss_mpd, gen_loss_mpd = self._gen_type_loss(fake, real, self.disc.compute_mpds)
        feature_loss_mrd, gen_loss_mrd = self._gen_type_loss(fake, real, self.disc.compute_mrds)

        return feature_loss_mpd, gen_loss_mpd, feature_loss_mrd, gen_loss_mrd

    def _gen_type_loss(self, fake, real, gen_type):
        y_fakes, fmaps_fake = gen_type(fake)
        y_reals, fmaps_real = gen_type(real)

        # feature loss
        feature_loss = 0
        for fmap_fake, fmap_real in zip(fmaps_fake, fmaps_real):
            for f, r in zip(fmap_fake, fmap_real):
                feature_loss += torch.mean(torch.abs(f - r))

        # generator loss
        gen_loss = 0
        for y_fake in y_fakes:
            gen_loss += torch.mean((1 - y_fake) ** 2)

        return feature_loss, gen_loss
