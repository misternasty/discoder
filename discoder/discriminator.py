import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from einops import rearrange
from audiotools import AudioSignal, STFTParams

from discoder.modules import get_padding


LRELU_SLOPE = 0.1

class MPD(torch.nn.Module):
    """
    Implementation adapted from https://github.com/NVIDIA/BigVGAN and https://github.com/descriptinc/descript-audio-codec
    """
    def __init__(self, discriminator_channel_mult: int, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(MPD, self).__init__()
        self.period = period
        self.d_mult = discriminator_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

class MRSD(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        """Complex multi-band spectrogram discriminator.

        Implementation from https://github.com/descriptinc/descript-audio-codec

        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)

        x = torch.flatten(x, 1, -1)

        return x, fmap


def WNConv2d(*args, **kwargs):
    """Implementation from https://github.com/descriptinc/descript-audio-codec"""
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class Discriminator(nn.Module):
    def __init__(self, periods: list, discriminator_channel_mult: int, use_spectral_norm: bool):
        """Discriminator that combines multiple sub-discriminators.

        Implementation adapted from https://github.com/descriptinc/descript-audio-codec
        """
        super().__init__()

        mpds = [MPD(discriminator_channel_mult=discriminator_channel_mult, period=period, use_spectral_norm=use_spectral_norm) for period in periods]
        mrds = [MRSD(f, sample_rate=44100, bands=BANDS) for f in [2048, 1024, 512]]

        self.mpds = nn.ModuleList(mpds)
        self.mrds = nn.ModuleList(mrds)

        self.discs = nn.ModuleList(mpds + mrds)

    def forward(self, x):
        return self._compute_disc(x=x, disc_type=self.discs)

    def compute_mpds(self, x):
        return self._compute_disc(x=x, disc_type=self.mpds)

    def compute_mrds(self, x):
        return self._compute_disc(x=x, disc_type=self.mrds)

    def preprocess(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return x

    def _compute_disc(self, x, disc_type):
        ys = []
        fmaps = []
        x = self.preprocess(x)
        for disc in disc_type:
            out = disc(x)
            ys.append(out[0])
            fmaps.append(out[1])
        return ys, fmaps

