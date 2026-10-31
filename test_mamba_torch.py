"""Checks the pure-PyTorch Mamba port against upstream mamba_ssm's own
reference code (selective_scan_ref and the Mamba module's non-fused path),
loaded from the unpacked PyPI source with its CUDA-only imports stubbed out.

Run once:  python scripts/fetch_mamba_reference.py
Then:      python -m pytest -q
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

import mamba_torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / ".mamba_ref" / "mamba_ssm-2.3.2.post1" / "mamba_ssm"
HAS_MPS = torch.backends.mps.is_available()

if not UPSTREAM.is_dir():
    raise RuntimeError(f"Upstream reference not found at {UPSTREAM}. "
                       "Run: python scripts/fetch_mamba_reference.py")


def _load_upstream():
    def passthrough(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)

    stubs = {
        "mamba_ssm": {}, "mamba_ssm.utils": {}, "mamba_ssm.ops": {}, "mamba_ssm.modules": {},
        "mamba_ssm.utils.torch": {"custom_fwd": passthrough, "custom_bwd": passthrough},
        "mamba_ssm.ops.triton": {},
        "mamba_ssm.ops.triton.layer_norm": {"_layer_norm_fwd": None, "RMSNorm": None,
                                            "layer_norm_fn": None, "rms_norm_fn": None},
        "selective_scan_cuda": {},
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        ssi = load("mamba_ssm.ops.selective_scan_interface", "ops/selective_scan_interface.py")
        simple = load("mamba_ssm.modules.mamba_simple", "modules/mamba_simple.py")
    finally:
        for k in list(stubs) + ["mamba_ssm.ops.selective_scan_interface",
                                "mamba_ssm.modules.mamba_simple"]:
            if saved.get(k) is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = saved[k]
    assert simple.causal_conv1d_fn is None  # forces upstream's non-fused path
    simple.selective_scan_fn = ssi.selective_scan_ref  # its pure-torch reference
    return ssi.selective_scan_ref, simple.Mamba


selective_scan_ref, UpstreamMamba = _load_upstream()


def _random_scan_inputs(b, d, l, n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    u, dt, z = r(b, d, l), r(b, d, l), r(b, d, l)
    A = -torch.exp(r(d, n) * 0.5 + 0.5)  # negative, varied magnitudes like trained A
    B, C = r(b, n, l), r(b, n, l)
    D, bias = r(d), r(d)
    return u, dt, A, B, C, D, z, bias


@pytest.mark.parametrize("l", [1, 5, 64, 97, 300])
def test_scan_matches_upstream_reference(l):
    # fp32: upstream's ref casts inputs to fp32 itself and fails on fp64.
    u, dt, A, B, C, D, z, bias = _random_scan_inputs(3, 8, l, 4, torch.float32)
    ref = selective_scan_ref(u, dt, A, B, C, D, z=z, delta_bias=bias, delta_softplus=True)
    t = lambda x: x.transpose(1, 2)  # upstream (b, d, l) -> port (b, l, d)
    out = mamba_torch.selective_scan(t(u), t(dt), A, t(B), t(C), D, z=t(z), delta_bias=bias,
                                     delta_softplus=True)
    torch.testing.assert_close(t(out), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not HAS_MPS, reason="no MPS")
@pytest.mark.parametrize("backend", ["metal", "torch"])
@pytest.mark.parametrize("l,n,scale", [(1, 16, 1.0), (97, 16, 1.0), (300, 16, 1.0),
                                       (64, 4, 1.0), (50, 16, 12.0)])
def test_scan_on_mps_matches_upstream_reference(backend, l, n, scale):
    # scale=12 pushes delta past softplus's threshold of 20 and z into saturation.
    u, dt, A, B, C, D, z, bias = _random_scan_inputs(3, 40, l, n, torch.float32, seed=l)
    dt, z = dt * scale, z * scale
    ref = selective_scan_ref(u, dt, A, B, C, D, z=z, delta_bias=bias, delta_softplus=True)
    t = lambda x: x.transpose(1, 2).to("mps")
    out = mamba_torch.selective_scan(t(u), t(dt), A.to("mps"), t(B), t(C), D.to("mps"),
                                     z=t(z), delta_bias=bias.to("mps"), delta_softplus=True,
                                     backend=backend)
    torch.testing.assert_close(out.cpu().transpose(1, 2), ref, rtol=1e-4, atol=1e-4)


def _paired_modules(seed=0, d_model=64):
    torch.manual_seed(seed)
    up = UpstreamMamba(d_model=d_model, d_state=16, d_conv=4, expand=4).eval()
    with torch.no_grad():  # move away from init so every parameter matters
        for p in up.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    port = mamba_torch.Mamba(d_model=d_model, d_state=16, d_conv=4, expand=4).eval()
    port.load_state_dict(up.state_dict(), strict=True)
    return up, port


@pytest.mark.parametrize("l", [7, 200])
def test_module_matches_upstream_cpu(l):
    up, port = _paired_modules()
    x = torch.randn(4, l, 64)
    with torch.no_grad():
        torch.testing.assert_close(port(x), up(x), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not HAS_MPS, reason="no MPS")
@pytest.mark.parametrize("l", [7, 200, 450])
def test_module_on_mps_matches_upstream_cpu(l):
    up, port = _paired_modules(seed=1)
    x = torch.randn(4, l, 64)
    with torch.no_grad():
        ref = up(x)
        out = port.to("mps")(x.to("mps")).cpu()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


def test_checkpoint_loads_strict():
    from enhance import Enhancer  # raises if any key is missing or unexpected
    enh = Enhancer(device="cpu")  # downloads the pinned model revision (~40 MB) once
    n = sum(p.numel() for p in enh.model.parameters())
    assert n == 9_607_747, n  # parameter count published on the HF model page
