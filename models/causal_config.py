"""Explicit, checkpointed causal alternatives to the historical model blocks."""

import torch
from torch import nn


class ChannelNorm(nn.Module):
    """LayerNorm over channels of each sample, never over batch or time."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        if x.ndim == 2:  # incremental (B, C)
            return self.norm(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def configure_causal_blocks(model, tcn_norm="channel", se_window=None, attention_window=None):
    """Apply model options before optimization or checkpoint loading."""
    from models.TCN_backbone import CausalSE
    from models.AttentionBlocks.attention import CausalMHA, CausalMLA
    if tcn_norm not in ("batch", "channel", "none"):
        raise ValueError("tcn_norm must be batch, channel or none")
    for name, value in (("se_window", se_window), ("attention_window", attention_window)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if tcn_norm != "channel":
        for module in list(model.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, ChannelNorm):
                    channels = child.norm.normalized_shape[0]
                    replacement = nn.BatchNorm1d(channels) if tcn_norm == "batch" else nn.Identity()
                    setattr(module, name, replacement)
    for module in model.modules():
        if isinstance(module, CausalSE):
            if se_window is not None and module.lookahead:
                raise ValueError("finite SE pooling cannot use the legacy architectural lookahead")
            module.pool_window = se_window
        if isinstance(module, (CausalMHA, CausalMLA)):
            module.attention_window = attention_window
            if attention_window is not None:
                n = module.causal_mask.shape[-1]
                if attention_window > n:
                    raise ValueError("attention_window cannot exceed max_seq_len")
                i = torch.arange(n)
                distance = i[:, None] - i[None, :]
                allowed = ((distance >= 0) & (distance < attention_window) if module.causal
                           else distance.abs() < attention_window)
                module.causal_mask.copy_(allowed.to(module.causal_mask).view(1, 1, n, n))
