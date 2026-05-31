"""Super-omni decoder modules: audio vocoder + video frame decoder.

These modules convert flow-matching latent predictions back to perceptual
outputs (waveform / video frames).  During training only the latent-level
flow matching loss is used; the decoders are exercised at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioMelDecoder(nn.Module):
    """Lightweight conv-based mel-spectrogram → waveform decoder.

    Can be replaced by Vocos / HiFi-GAN / Encodec decoder at inference.
    For training we only need the latent projection; this module exists so
    that ``super_omni_gen`` can produce playable audio during demo / eval.

    Architecture: 1-D transposed conv stack that upsamples mel frames to
    waveform samples at 24 kHz (hop_size 256 by default).
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 512,
        hop_size: int = 256,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        self.proj_in = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=16, stride=8, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=16, stride=4, padding=6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1),
        )

        self.res_block = nn.Sequential(
            nn.Conv1d(hidden_dim // 16, hidden_dim // 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim // 16, 1, kernel_size=1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_mel) → waveform: (B, 1, T_wave)."""
        h = self.proj_in(mel)
        h = self.upsample(h)
        return self.res_block(h)


class AudioPatchEncoder(nn.Module):
    """Patchify a mel-spectrogram into tokens for flow matching.

    Splits (n_mels, T) into (T // patch_t, n_mels * patch_t) tokens.
    """

    def __init__(self, n_mels: int = 80, patch_t: int = 4) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.patch_t = patch_t
        self.patch_dim = n_mels * patch_t

    def patchify(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, T) → (B * T//patch_t, patch_dim)."""
        B, n_mels, T = mel.shape
        pt = self.patch_t
        T_p = T // pt
        mel = mel[:, :, : T_p * pt]
        patches = mel.reshape(B, n_mels, T_p, pt).permute(0, 2, 1, 3).reshape(B * T_p, n_mels * pt)
        return patches

    def unpatchify(self, patches: torch.Tensor, T_mel: int) -> torch.Tensor:
        """(N, patch_dim) → (1, n_mels, T_mel)."""
        pt = self.patch_t
        n_mels = self.n_mels
        T_p = T_mel // pt
        B = patches.shape[0] // max(T_p, 1)
        if B < 1:
            B = 1
        mel = patches.reshape(B, T_p, n_mels, pt).permute(0, 2, 1, 3).reshape(B, n_mels, T_p * pt)
        if mel.shape[-1] < T_mel:
            mel = torch.nn.functional.pad(mel, (0, T_mel - mel.shape[-1]))
        return mel[:, :, :T_mel]


class VideoFrameDecoder(nn.Module):
    """Decode video latent patches back to pixel frames.

    Each frame's latent patches are independently decoded using the same
    unpatchify logic as image generation, then stacked into a video tensor.
    """

    def __init__(
        self,
        patch_size: int = 16,
        merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.pixel_per_patch = 3 * (patch_size * merge_size) ** 2

    def decode_frames(
        self,
        latent_patches: torch.Tensor,
        num_frames: int,
        frame_h: int,
        frame_w: int,
    ) -> torch.Tensor:
        """latent_patches: (N_patches, pixel_per_patch) → (num_frames, 3, H, W).

        Patches are laid out as [frame0_patch0, frame0_patch1, ..., frame1_patch0, ...].
        """
        ps = self.patch_size
        ms = self.merge_size
        h_tokens = frame_h // (ps * ms)
        w_tokens = frame_w // (ps * ms)
        patches_per_frame = h_tokens * w_tokens

        n_total = min(latent_patches.shape[0], num_frames * patches_per_frame)
        actual_frames = n_total // max(patches_per_frame, 1)
        if actual_frames < 1:
            actual_frames = 1

        frames = []
        for f in range(actual_frames):
            start = f * patches_per_frame
            end = min(start + patches_per_frame, n_total)
            fp = latent_patches[start:end]

            c = 3
            merge_px = ps * ms
            n = fp.shape[0]
            h_t = int(n ** 0.5) if h_tokens * w_tokens != n else h_tokens
            w_t = n // max(h_t, 1)

            frame = fp.reshape(h_t, w_t, c, ps, ms, ps, ms)
            frame = frame.permute(2, 0, 3, 5, 1, 4, 6).reshape(c, h_t * ps * ms, w_t * ps * ms)
            frames.append(frame.unsqueeze(0))

        if not frames:
            return torch.zeros(1, 3, frame_h, frame_w, device=latent_patches.device, dtype=latent_patches.dtype)
        return torch.cat(frames, dim=0)
