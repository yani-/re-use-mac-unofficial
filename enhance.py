"""Run NVIDIA NemotronLabs RE-USE speech enhancement on macOS (MPS or CPU).

Unofficial port, not affiliated with or endorsed by NVIDIA.

Replaces the model repo's inference.py / inference_chunk.py, which require
CUDA. NVIDIA's model code and weights are downloaded from Hugging Face at a
pinned revision and imported unchanged; only ``mamba_ssm.Mamba`` is swapped
for the pure-PyTorch version in mamba_torch.py.

Usage:
    python enhance.py INPUT [INPUT ...] -o OUTPUT_DIR [--device auto|mps|cpu]
                      [--chunk-seconds 5] [--hop-portion 0.5] [--max-full-seconds 30]
                      [--bwe 32000] [--repo DIR]

INPUT may be audio files or folders (searched recursively). Files no longer
than --max-full-seconds are processed in one pass like inference.py; longer
ones use the overlap-add chunking of inference_chunk.py. Output is .flac.
"""

# Portions of this file (STFT parameter scaling, the sweep-artifact
# post-processing in enhance_segment, and the overlap-add chunking in
# enhance_chunked) are adapted from inference.py and inference_chunk.py in
# https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE
#   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: OpenMDW-1.1 (see LICENSES/OpenMDW-1.1.txt)
# Changes: CPU/MPS device selection, STFT on CPU, soundfile I/O, per-channel
# processing, automatic choice between full and chunked inference.

import argparse
import importlib
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

import mamba_torch

HF_REPO = "nvidia/NVIDIA-NemotronLabs-RE-USE"
# The model repo commit this port was verified against. Its Python files are
# executed, so it is pinned rather than following main.
HF_REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
HF_FILES = ["model.safetensors", "models/*.py", "utils/*.py", "recipes/*.yaml", "LICENSE"]
RECIPE ="recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a"}


def _import_repo(repo_dir):
    """Import the repo's model code under a private package name so its
    generic ``models``/``utils`` folders cannot collide with installed packages."""
    shim = types.ModuleType("mamba_ssm")
    shim.Mamba = mamba_torch.Mamba
    sys.modules["mamba_ssm"] = shim
    pkg = types.ModuleType("reuse_repo")
    pkg.__path__ = [str(repo_dir)]
    sys.modules["reuse_repo"] = pkg
    gen = importlib.import_module("reuse_repo.models.generator_SEMamba_time_d4")
    stfts = importlib.import_module("reuse_repo.models.stfts")
    util = importlib.import_module("reuse_repo.utils.util")
    return gen.SEMamba, stfts, util


def fetch_model_repo(revision=HF_REVISION):
    """Download (or reuse from the HF cache) the files the port needs, about 40 MB."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(HF_REPO, revision=revision, allow_patterns=HF_FILES))


def pick_device(name):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class Enhancer:
    def __init__(self, repo_dir=None, device="auto"):
        from safetensors.torch import load_file

        repo_dir = Path(repo_dir) if repo_dir else fetch_model_repo()
        SEMamba, self.stfts, self.util = _import_repo(repo_dir)
        self.cfg = self.util.load_config(repo_dir / RECIPE)
        self.device = pick_device(device)
        model = SEMamba(cfg=self.cfg)
        # strict=True: every checkpoint tensor must map onto a parameter and vice versa.
        model.load_state_dict(load_file(str(repo_dir / "model.safetensors")), strict=True)
        self.model = model.to(self.device).eval()

    @staticmethod
    def _make_even(value):
        value = int(round(value))
        return value if value % 2 == 0 else value + 1

    def _stft_params(self, sr):
        s = self.cfg["stft_cfg"]
        base = s["sampling_rate"]
        return tuple(self._make_even(s[k] * sr // base) for k in ("n_fft", "hop_size", "win_size"))

    @torch.no_grad()
    def enhance_segment(self, wav, sr):
        """wav: float tensor (1, N) on CPU. Mirrors the body of inference.py."""
        n_fft, hop, win = self._stft_params(sr)
        cf = self.cfg["model_cfg"]["compress_factor"]
        # STFT/iSTFT stay on CPU; only the network runs on the accelerator.
        mag, pha, _ = self.stfts.mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win,
                                                compress_factor=cf, center=True, addeps=False)
        amp_g, pha_g, _ = self.model(mag.to(self.device), pha.to(self.device))
        amp_g, pha_g = amp_g.cpu(), pha_g.cpu()
        # Upstream post-processing that removes a "strange sweep artifact".
        m = torch.expm1(torch.relu(amp_g))
        zero_portion = torch.sum(m == 0, 1) / m.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        out = self.stfts.mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cf)
        return self.util.pad_or_trim_to_match(wav, out, pad_value=1e-8)

    def enhance_chunked(self, wav, sr, chunk_seconds=5.0, hop_portion=0.5):
        """Hann-windowed overlap-add over chunks. Mirrors inference_chunk.py."""
        chunk = int(chunk_seconds * sr)
        hop = int(hop_portion * chunk)
        n = wav.shape[1]
        if n <= chunk:
            return self.enhance_segment(wav, sr)
        window = torch.clamp(torch.hann_window(chunk), min=1e-7)
        out = torch.zeros_like(wav)
        wsum = torch.zeros_like(wav)
        num_chunks = max(1, math.ceil((n - chunk) / hop) + 1)
        for i in range(num_chunks):
            sl = slice(i * hop, i * hop + chunk) if i < num_chunks - 1 else slice(n - chunk, n)
            seg = self.enhance_segment(wav[:, sl], sr)
            out[:, sl] += seg * window[: seg.shape[1]]
            wsum[:, sl] += window[: seg.shape[1]]
        return out / wsum

    def enhance_file(self, path, out_path, chunk_seconds=5.0, hop_portion=0.5,
                     max_full_seconds=30.0, bwe=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (N, C)
        data = data.T  # (C, N)
        if bwe is not None:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=int(bwe), res_type="kaiser_best")
            sr = int(bwe)
        chunked = data.shape[1] / sr > max_full_seconds
        channels = []
        for c in range(data.shape[0]):  # each channel independently, like inference_chunk.py
            wav = torch.from_numpy(np.ascontiguousarray(data[c:c + 1]))
            if chunked:
                y = self.enhance_chunked(wav, sr, chunk_seconds, hop_portion)
            else:
                y = self.enhance_segment(wav, sr)
            channels.append(y[0].numpy())
        out = np.clip(np.stack(channels, axis=1), -1.0, 1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, sr, subtype="PCM_24")
        return sr, data.shape[1] / sr, chunked


def iter_inputs(paths):
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in AUDIO_EXTS:
                    yield p, f
        else:
            yield p.parent, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("--repo", default=None,
                    help="local copy of the HF model repo (default: download the pinned revision)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--chunk-seconds", type=float, default=5.0)
    ap.add_argument("--hop-portion", type=float, default=0.5)
    ap.add_argument("--max-full-seconds", type=float, default=30.0,
                    help="files longer than this are processed in chunks")
    ap.add_argument("--bwe", type=int, default=None,
                    help="bandwidth extension: resample input to this rate first (e.g. 32000)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    enh = Enhancer(args.repo, args.device)
    print(f"Loaded model on {enh.device} in {time.perf_counter() - t0:.1f}s")
    out_root = Path(args.output)
    for base, f in iter_inputs(args.inputs):
        out_path = out_root / f.relative_to(base).with_suffix(".flac")
        t = time.perf_counter()
        sr, dur, chunked = enh.enhance_file(f, out_path, args.chunk_seconds, args.hop_portion,
                                            args.max_full_seconds, args.bwe)
        dt = time.perf_counter() - t
        mode = "chunked" if chunked else "full"
        print(f"{f} -> {out_path}  ({dur:.1f}s @ {sr} Hz, {mode}, {dt:.1f}s, {dur / dt:.1f}x realtime)")


if __name__ == "__main__":
    main()
