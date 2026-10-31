# RE-USE for Mac (unofficial)

An unofficial port of [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) to macOS on Apple Silicon. RE-USE is a 9.6M-parameter universal speech enhancement model: it removes noise, reverb, clipping, codec artifacts and similar damage from speech recordings at 8 to 48 kHz, in any language.

**Not affiliated with or endorsed by NVIDIA.** For the official release, model card and demo, see NVIDIA's [model page](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE).

## Why a port is needed

The official release runs only on Linux with an NVIDIA GPU, because the model's Mamba layers come from [`mamba_ssm`](https://github.com/state-spaces/mamba), which ships CUDA kernels only. This port replaces that one layer with:

- a fused **Metal** kernel for the Apple GPU (via `torch.mps.compile_shader`), which works like the upstream CUDA kernel
- a pure **PyTorch** fallback for the CPU

Everything else is NVIDIA's code and weights, downloaded unchanged from Hugging Face at a pinned commit.

## Install

```bash
git clone https://github.com/yani-/re-use-mac-unofficial.git
cd re-use-mac-unofficial
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python enhance.py noisy.wav -o enhanced/
.venv/bin/python enhance.py recordings/ more.flac -o enhanced/   # folders are searched recursively
```

The first run downloads the model (about 40 MB) into the Hugging Face cache. Output files are written as 24-bit FLAC at the input's sample rate.

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `mps` (Apple GPU) when available, otherwise `cpu` |
| `--max-full-seconds` | `30` | Longer files are processed in overlapping chunks, as in NVIDIA's `inference_chunk.py` |
| `--chunk-seconds` | `5` | Chunk length for long files |
| `--hop-portion` | `0.5` | Chunk hop as a fraction of chunk length |
| `--bwe` | off | Bandwidth extension: resample to this rate (for example `32000`) before enhancing. Needs `pip install librosa resampy` |
| `--repo` | download | Use a local copy of the Hugging Face model repo instead |

## Performance

Measured on the 5.1-second, 44.1 kHz sample from the model repo, on an M5 Max running on battery:

| Device | Time | Speed |
|---|---|---|
| Apple GPU (MPS + Metal kernel) | 1.3 to 2.3 s | 2 to 4x faster than real time |
| CPU | about 40 s | about 8x slower than real time |

Use the GPU. The CPU path exists for correctness checks and for machines without MPS.

## Verification

- **Against NVIDIA's own output:** the sample file was enhanced by NVIDIA's [demo Space](https://huggingface.co/spaces/nvidia/NVIDIA-NemotronLabs-RE-USE) on an NVIDIA GPU and by this port. The outputs agree to 55.4 dB SNR with a correlation of 0.999999. The noisy input scores 1.7 dB against the same reference. The Space returns 16-bit audio, which caps the possible agreement at about 70 dB.
- **Against upstream Mamba:** the test suite compares the port with `mamba_ssm`'s own pure-PyTorch reference code (`selective_scan_ref` and the `Mamba` module's non-fused path), on CPU and on the Apple GPU, including inputs past the softplus threshold.
- **Weights:** the checkpoint loads with `strict=True` (all 9,607,747 parameters match).
- **Not verified:** chunked mode for long files mirrors NVIDIA's `inference_chunk.py`, but there was no official chunked output to compare against.

Run the tests:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/fetch_mamba_reference.py   # downloads and unpacks mamba-ssm source (hash-checked, not built)
.venv/bin/python -m pytest -q
```

## How it works

- NVIDIA's model code does `from mamba_ssm import Mamba`. `enhance.py` registers a stand-in `mamba_ssm` module, so that import resolves to `mamba_torch.Mamba`, which has the same parameter names and shapes as the original.
- **Apple GPU:** the selective scan runs as one Metal kernel. Each GPU thread owns one (batch, channel) pair, keeps the 16-value state in registers and steps through the sequence, so the large intermediate state tensor is never written to memory.
- **CPU:** the scan is a plain sequential loop. The layer's small depthwise convolution is computed as four shifted multiply-adds, because `nn.Conv1d` with 256 groups makes one NNPACK call per group on macOS CPUs, which took 80% of total CPU time.
- STFT and inverse STFT run on the CPU; only the network runs on the GPU.

## Limitations

- Inference only: no training or fine-tuning.
- Tested with Python 3.13, PyTorch 2.15 and macOS 27.2 on an M5 Max. `torch>=2.7` is required for `torch.mps.compile_shader`, but older versions are untested.
- Intel Macs have no MPS support, so they would use the slow CPU path.

## License

- The code in this repository is licensed under the [Apache License 2.0](LICENSE).
- `mamba_torch.py` is derived from [`mamba_ssm`](https://github.com/state-spaces/mamba) (Copyright 2023 Tri Dao, Albert Gu, Apache 2.0).
- `enhance.py` adapts parts of NVIDIA's inference scripts, which are licensed under the [OpenMDW License Agreement 1.1](LICENSES/OpenMDW-1.1.txt).
- NVIDIA's model weights and model code are **not** included. They are downloaded from Hugging Face at run time and governed by OpenMDW-1.1.

See [NOTICE](NOTICE) for full attributions.

## Citation

If you use the model in research, cite NVIDIA's papers:

- [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641) (NeurIPS 2026)
- [One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](https://arxiv.org/abs/2606.25621) (IEEE SLT 2026)
