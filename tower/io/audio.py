from __future__ import annotations

import wave
from pathlib import Path

import torch
import torch.nn.functional as F


def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Read PCM wav file and return mono float waveform in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        audio_i = torch.frombuffer(raw, dtype=torch.uint8).to(torch.int16) - 128
        scale = 128.0
    elif sample_width == 2:
        audio_i = torch.frombuffer(raw, dtype=torch.int16)
        scale = 32768.0
    elif sample_width == 4:
        audio_i = torch.frombuffer(raw, dtype=torch.int32)
        scale = float(1 << 31)
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width}")

    if n_channels > 1:
        audio_i = audio_i.view(-1, n_channels).mean(dim=1)
    audio = audio_i.to(torch.float32) / scale
    return audio.contiguous(), int(sr)


def _resample_linear(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    if waveform.numel() == 0:
        return waveform
    x = waveform.view(1, 1, -1)
    target_len = max(1, int(round(waveform.shape[0] * float(dst_sr) / float(src_sr))))
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.view(-1)


def waveform_to_patch_features(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_freq_bins: int = 80,
    patch_frames: int = 8,
    patch_bins: int = 10,
    max_patches: int = 256,
) -> torch.Tensor:
    """Convert waveform to patch-level continuous features [N_patch, patch_dim]."""
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    if waveform.numel() < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.numel()))

    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    ).abs()
    spec = torch.log1p(spec)
    spec = spec.unsqueeze(0).unsqueeze(0)  # [1,1,F,T]
    spec = F.interpolate(spec, size=(n_freq_bins, spec.shape[-1]), mode="bilinear", align_corners=False)
    spec = spec[0, 0]

    mean = spec.mean()
    std = spec.std().clamp_min(1e-5)
    spec = (spec - mean) / std

    freq, frames = int(spec.shape[0]), int(spec.shape[1])
    pad_f = (patch_bins - (freq % patch_bins)) % patch_bins
    pad_t = (patch_frames - (frames % patch_frames)) % patch_frames
    if pad_f or pad_t:
        spec = F.pad(spec, (0, pad_t, 0, pad_f))

    freq, frames = int(spec.shape[0]), int(spec.shape[1])
    n_f = freq // patch_bins
    n_t = frames // patch_frames
    patches = (
        spec.view(n_f, patch_bins, n_t, patch_frames)
        .permute(0, 2, 1, 3)
        .reshape(n_f * n_t, patch_bins * patch_frames)
    )
    if patches.shape[0] > max_patches:
        patches = patches[:max_patches]
    return patches.contiguous()


def audio_file_to_patch_features(
    path: str | Path,
    *,
    target_sample_rate: int = 16000,
    **kwargs,
) -> torch.Tensor:
    """Read wav file and return patch-level features."""
    wav, sr = _read_wav(Path(path))
    wav = _resample_linear(wav, sr, target_sample_rate)
    return waveform_to_patch_features(wav, sample_rate=target_sample_rate, **kwargs)
