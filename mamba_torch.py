"""Pure-PyTorch drop-in for ``mamba_ssm.Mamba`` (Mamba v1), for inference on
Apple Silicon (MPS) or CPU.

The upstream ``mamba_ssm`` package only ships CUDA kernels, so it cannot be
installed on macOS. This module reimplements the same layer. Parameter names
and shapes match upstream exactly, so checkpoints trained with ``mamba_ssm``
load with ``strict=True``.

Selective-scan backends:
- MPS (Apple GPU): a fused Metal kernel, the same strategy as upstream's CUDA
  kernel. One thread per (batch, channel) keeps the d_state hidden state in
  registers and walks the sequence, so the (b, l, d, n) state is never stored.
- Anywhere else (CPU): a sequential PyTorch loop over time. On CPU this beat a
  chunked parallel (Hillis-Steele) scan by 2x, since the latter does
  log(chunk) times more memory traffic.

Inference only: weight init is simplified (no dt init) because the weights are
expected to come from a checkpoint, and there is no backward pass.
"""

# Derived from mamba_ssm (https://github.com/state-spaces/mamba):
# mamba_ssm/modules/mamba_simple.py (Mamba) and
# mamba_ssm/ops/selective_scan_interface.py (selective_scan_ref).
#   Copyright 2023 Tri Dao, Albert Gu. Licensed under the Apache License 2.0.
# Changes: reimplemented without CUDA/Triton dependencies, (b, l, d) layout,
# added a fused Metal selective-scan kernel and a shifted-sum causal conv for
# CPU, removed training-only initialization and step-wise decoding.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#define DSTATE %(n)d

inline float softplus_f(float x) {  // matches F.softplus(beta=1, threshold=20)
    if (x > 20.0f) return x;
    const float e = precise::exp(x);
    return e < 1e-4f ? e * (1.0f - 0.5f * e) : precise::log(1.0f + e);
}

// All tensors contiguous fp32. u, delta, z, out: (b, l, d). B, C: (b, l, n).
// A: (d, n). D, bias: (d). dims: int32 [b, l, d].
kernel void selective_scan_fwd(
    device float* out [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* delta [[buffer(2)]],
    device const float* A [[buffer(3)]],
    device const float* Bm [[buffer(4)]],
    device const float* Cm [[buffer(5)]],
    device const float* D [[buffer(6)]],
    device const float* z [[buffer(7)]],
    device const float* bias [[buffer(8)]],
    device const int* dims [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const int nb = dims[0], L = dims[1], nd = dims[2];
    if ((int)gid >= nb * nd) return;
    const int bi = (int)gid / nd, di = (int)gid %% nd;
    float a[DSTATE], h[DSTATE];
    for (int k = 0; k < DSTATE; ++k) { a[k] = A[di * DSTATE + k]; h[k] = 0.0f; }
    const float Dd = D[di], bd = bias[di];
    for (int t = 0; t < L; ++t) {
        const long row = (long)bi * L + t;
        const long off = row * nd + di;
        const long offn = row * DSTATE;
        const float ut = u[off];
        const float dt = softplus_f(delta[off] + bd);
        const float dtu = dt * ut;
        float y = 0.0f;
        for (int k = 0; k < DSTATE; ++k) {
            h[k] = precise::exp(dt * a[k]) * h[k] + dtu * Bm[offn + k];
            y += h[k] * Cm[offn + k];
        }
        const float zt = z[off];
        out[off] = (y + ut * Dd) * (zt / (1.0f + precise::exp(-zt)));
    }
}
"""
_metal_libs = {}


def _selective_scan_metal(u, delta, A, B, C, D, z, delta_bias):
    n = A.shape[1]
    lib = _metal_libs.get(n)
    if lib is None:
        lib = _metal_libs[n] = torch.mps.compile_shader(_METAL_SRC % {"n": n})
    f = lambda t: t.float().contiguous()
    u, delta, A, B, C, D, z, delta_bias = map(f, (u, delta, A, B, C, D, z, delta_bias))
    b, l, d = u.shape
    out = torch.empty_like(u)
    dims = torch.tensor([b, l, d], dtype=torch.int32, device=u.device)
    lib.selective_scan_fwd(out, u, delta, A, B, C, D, z, delta_bias, dims, threads=b * d)
    return out


def _selective_scan_torch(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()
    if delta_softplus:
        delta = F.softplus(delta)
    A, B, C = A.float(), B.float(), C.float()
    b, l, d = u.shape
    du = delta * u
    state = u.new_zeros((b, d, A.shape[1]))
    y = torch.empty_like(u)
    for t in range(l):
        state = torch.exp(delta[:, t, :, None] * A) * state + du[:, t, :, None] * B[:, t, None, :]
        y[:, t] = torch.matmul(state, C[:, t, :, None]).squeeze(-1)
    if D is not None:
        y = y + u * D.float()
    if z is not None:
        y = y * F.silu(z.float())
    return y


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=True, backend="auto"):
    """Selective scan in (batch, length, channels) layout.

    u, delta, z: (b, l, d)   A: (d, n)   B, C: (b, l, n)   D, delta_bias: (d,)
    Returns y: (b, l, d), same math as mamba_ssm.selective_scan_ref.
    backend: "auto" (Metal on MPS when supported, else torch), "metal", "torch".
    """
    metal_ok = (u.device.type == "mps" and D is not None and z is not None
                and delta_bias is not None and delta_softplus)
    if backend == "metal" and not metal_ok:
        raise ValueError("metal backend needs MPS tensors with D, z, delta_bias and softplus")
    if backend == "metal" or (backend == "auto" and metal_ok):
        y = _selective_scan_metal(u, delta, A, B, C, D, z, delta_bias)
    else:
        y = _selective_scan_torch(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    return y.to(u.dtype)


class Mamba(nn.Module):
    """Same constructor and parameters as mamba_ssm.modules.mamba_simple.Mamba."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, device=None, dtype=None, **_ignored):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **fk)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1,
                                bias=conv_bias, **fk)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **fk)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **fk)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **fk)

    def _causal_conv(self, x):
        """Causal depthwise conv over time, x: (b, l, d_inner)."""
        l = x.shape[1]
        if x.device.type != "cpu":
            return self.conv1d(x.transpose(1, 2))[..., :l].transpose(1, 2)
        # On macOS CPUs nn.Conv1d with groups=d_inner runs one tiny NNPACK call per
        # group, which made it 80% of total CPU time. d_conv shifted multiply-adds
        # give the same result far faster.
        k = self.d_conv
        w = self.conv1d.weight[:, 0, :]                                # (d_inner, k)
        xp = F.pad(x, (0, 0, k - 1, 0))                                # left-pad time
        y = xp[:, 0:l] * w[:, 0]
        for i in range(1, k):
            y = y + xp[:, i:i + l] * w[:, i]
        return y if self.conv1d.bias is None else y + self.conv1d.bias

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("step-wise decoding is not implemented")
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)          # (b, l, d_inner)
        x = self.act(self._causal_conv(x))
        dt, B, C = torch.split(self.x_proj(x),
                               [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)  # bias is applied inside the scan
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B, C, self.D, z=z,
                           delta_bias=self.dt_proj.bias, delta_softplus=True)
        return self.out_proj(y)
