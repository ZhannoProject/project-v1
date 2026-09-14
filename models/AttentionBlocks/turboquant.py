"""TurboQuant reference codec for per-token, per-head K/V vectors.

Independent implementation of arXiv:2504.19874, sections 3.1 and 3.2:
Haar rotation + sphere-marginal Lloyd-Max scalar quantization; optionally
QJL correction of the residual for unbiased inner products over randomness.
This is a numerical reference, not a fused attention/cache kernel.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import nn


@lru_cache(maxsize=64)
def _codebook(dim, bits):
    # Fit the analytic sphere-coordinate distribution, never EEG samples.
    import numpy as np
    from scipy.special import betainc, gammaln
    a = (dim - 1) / 2
    c = np.exp(gammaln(dim / 2) - .5 * np.log(np.pi) - gammaln(a))
    levels = np.linspace(-1, 1, 2 ** bits + 2)[1:-1]
    for _ in range(500):
        edges = np.r_[-1., (levels[:-1] + levels[1:]) / 2, 1.]
        mass = np.diff(betainc(a, a, (edges + 1) / 2))
        primitive = -c / (dim - 1) * np.maximum(0, 1 - edges ** 2) ** a
        means = np.divide(np.diff(primitive), mass, out=levels.copy(), where=mass > 1e-14)
        means = (means - means[::-1]) / 2
        if np.max(np.abs(means - levels)) < 1e-10:
            levels = means
            break
        levels = means
    return torch.tensor(levels, dtype=torch.float32)


def _pack(codes, bits):
    """Pack each vector independently into ceil(dim * bits / 8) bytes."""
    dim = codes.shape[-1]
    positions = torch.arange(dim * bits, device=codes.device)
    binary = ((codes.to(torch.int64).unsqueeze(-1) >> torch.arange(bits, device=codes.device)) & 1)
    binary = binary.flatten(-2) << (positions % 8)
    packed = torch.zeros(*codes.shape[:-1], (dim * bits + 7) // 8,
                         dtype=torch.int64, device=codes.device)
    packed.scatter_add_(-1, (positions // 8).expand_as(binary), binary)
    return packed.to(torch.uint8)


def _unpack(packed, dim, bits):
    positions = torch.arange(dim * bits, device=packed.device)
    binary = (packed[..., positions // 8].to(torch.int64) >> (positions % 8)) & 1
    return (binary.reshape(*packed.shape[:-1], dim, bits) <<
            torch.arange(bits, device=packed.device)).sum(-1)


@dataclass
class QuantizedVectors:
    codes: torch.Tensor
    norms: torch.Tensor
    residual_signs: torch.Tensor | None = None
    residual_norms: torch.Tensor | None = None

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.codes, self.norms, self.residual_signs, self.residual_norms)
                   if x is not None)


class TurboQuant(nn.Module):
    """Fixed, seeded quantizer; no trainable parameters.

    bits is the total per-coordinate budget, excluding FP32 norm metadata
    and byte padding. 'prod' uses bits-1 scalar bits plus one QJL sign bit.
    Buffers are nonpersistent: a checkpoint's model_config records the seed.
    """
    def __init__(self, dim, bits=3, mode='mse', seed=42):
        super().__init__()
        if type(dim) is not int or dim < 2:
            raise ValueError('TurboQuant requires head_dim >= 2')
        if type(bits) is not int or not 2 <= bits <= 8:
            raise ValueError('turboquant_bits must be an integer from 2 to 8')
        if mode not in ('mse', 'prod'):
            raise ValueError('TurboQuant mode must be mse or prod')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('turboquant_seed must be an integer in [0, 2**63)')
        self.dim, self.bits, self.mode = dim, bits, mode
        self.scalar_bits = bits - (mode == 'prod')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
        rotation, r = torch.linalg.qr(matrix)
        rotation = rotation * torch.where(r.diag() >= 0, 1., -1.)
        self.register_buffer('rotation', rotation.float(), persistent=False)
        self.register_buffer('levels', _codebook(dim, self.scalar_bits).clone(), persistent=False)
        self.register_buffer('qjl', torch.randn(dim, dim, generator=generator), persistent=False)

    @torch.no_grad()
    def encode(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError('Input last dimension must match TurboQuant dim')
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            unit = x / norms.clamp_min(torch.finfo(torch.float32).tiny)
            rotated = unit @ self.rotation.float()
            levels = self.levels.float()
            codes = torch.bucketize(rotated.contiguous(), (levels[:-1] + levels[1:]) / 2)
            result = QuantizedVectors(_pack(codes, self.scalar_bits), norms)
            if self.mode == 'prod':
                coarse = (levels[codes] @ self.rotation.float().T) * norms
                residual = x - coarse
                result.residual_norms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
                signs = (residual @ self.qjl.float().T >= 0).to(torch.uint8)
                result.residual_signs = _pack(signs, 1)
            return result

    @torch.no_grad()
    def decode(self, quantized):
        with torch.autocast(device_type=quantized.codes.device.type, enabled=False):
            codes = _unpack(quantized.codes, self.dim, self.scalar_bits)
            result = (self.levels.float()[codes] @ self.rotation.float().T) * quantized.norms
            if self.mode == 'prod':
                signs = _unpack(quantized.residual_signs, self.dim, 1).float() * 2 - 1
                correction = (signs @ self.qjl.float()) * (math.sqrt(math.pi / 2) / self.dim)
                result = result + correction * quantized.residual_norms
            return result

    def forward(self, x, ste=False):
        reconstructed = self.decode(self.encode(x)).to(x.dtype)
        return x + (reconstructed - x).detach() if ste else reconstructed


class TurboQuantKV(nn.Module):
    """Quantize reconstructed K/V in either MHA or MLA before attention.

    train_mode='off' leaves training unchanged, enabling quantization only
    during evaluation. 'ste' enables experimental straight-through training.
    """
    def __init__(self, dim, bits=3, key_mode='prod', train_mode='off', seed=42):
        super().__init__()
        if train_mode not in ('off', 'ste'):
            raise ValueError('turboquant_train_mode must be off or ste')
        self.train_mode = train_mode
        self.keys = TurboQuant(dim, bits, key_mode, seed)
        self.values = TurboQuant(dim, bits, 'mse', (seed + 1) % 2**63)

    def forward(self, k, v):
        if self.training and self.train_mode == 'off':
            return k, v
        ste = self.training and self.train_mode == 'ste'
        return self.keys(k, ste), self.values(v, ste)


def attach_turboquant(model, enabled=False, bits=3, key_mode='prod', train_mode='off', seed=42):
    from .attention import CausalMHA, CausalMLA
    if type(enabled) is not bool:
        raise ValueError('turboquant must be a YAML boolean')
    if not enabled:
        return
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("turboquant_seed must be an integer in [0, 2**63)")
    attention = [m for m in model.modules() if isinstance(m, (CausalMHA, CausalMLA))]
    if not attention:
        raise ValueError('TurboQuant requires a transformer model; TCN-only has no K/V')
    for i, module in enumerate(attention):
        module.turboquant = TurboQuantKV(module.head_dim, bits, key_mode, train_mode,
                                        (seed + 2 * i) % 2**63)
