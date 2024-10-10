import os

import torch
import torchaudio as ta
import torch.nn.functional as F

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2  # noqa


config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 48000
config.options.use_speech_scoring = False
svr_model_path = "libsvm_nu_svr_model.txt"
config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

api = visqol_lib_py.VisqolApi()
api.Create(config)


def visqol_score(input: torch.Tensor, target: torch.Tensor, sample_rate: int) -> int:
    """Computes the ViSQOL score for the input and target waveform. Both waveforms must have the same sample rate.

    :param input: Input waveform
    :param target: Target waveform
    :param sample_rate: Sample rate of both input and target
    :return: ViSQOL score
    """
    input = input.detach().cpu().squeeze().to(torch.float64)
    target = target.detach().cpu().squeeze().to(torch.float64)

    # need to sample it to 48kHz
    input = ta.functional.resample(waveform=input, orig_freq=sample_rate, new_freq=config.audio.sample_rate).numpy()
    target = ta.functional.resample(waveform=target, orig_freq=sample_rate, new_freq=config.audio.sample_rate).numpy()

    similarity_result = api.Measure(target, input)
    return similarity_result.moslqo


def mel_distance(input: torch.Tensor, target: torch.Tensor):
    """
    :param input: Log Mel-spectrogram of the input waveform
    :param target: Log Mel-spectrogram of the target waveform
    :return:
    """
    assert input.shape == target.shape
    return F.l1_loss(input=input, target=target).detach()