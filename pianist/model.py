"""Decoder-only transformer: pre-norm RMSNorm, rotary positions, QK-norm, SwiGLU MLPs,
fused scaled-dot-product (flash) attention, tied input/output embeddings and a KV cache."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pianist import tokenizer as tok


@dataclass
class ModelConfig:
    vocab_size: int = tok.VOCAB_SIZE
    seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    dropout: float = 0.1
    rope_base: float = 10_000.0

    @property
    def d_hidden(self) -> int:
        """SwiGLU width: 8/3 of d_model (parameter-matched to a 4x GELU MLP), rounded to 64."""
        return 64 * math.ceil(8 * self.d_model / 3 / 64)

    def to_dict(self) -> dict:
        return asdict(self)


class KVCache:
    """Preallocated keys and values for every layer, filled left to right."""

    def __init__(self, cfg: ModelConfig, batch: int, device, dtype):
        shape = (cfg.n_layers, batch, cfg.n_heads, cfg.seq_len, cfg.d_model // cfg.n_heads)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.layer, self.n_heads, self.dropout = layer, cfg.n_heads, cfg.dropout
        head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, -1).transpose(1, 3).unbind(2)
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        if cache is not None:
            # Either a prefill from an empty cache or one-token decoding, so the causal mask
            # below is always correct (SDPA's mask is aligned to the top-left corner).
            start = cache.length
            assert start == 0 or T == 1, "prefill must start from an empty cache"
            cache.k[self.layer, :, :, start:start + T] = k
            cache.v[self.layer, :, :, start:start + T] = v
            k = cache.k[self.layer, :, :, :start + T]
            v = cache.v[self.layer, :, :, :start + T]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_hidden, bias=False)
        self.down = nn.Linear(cfg.d_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer: int):
        super().__init__()
        self.attn_norm, self.mlp_norm = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg, layer), MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.mlp_norm(x)))


class Pianist(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight

        head_dim = cfg.d_model // cfg.n_heads
        inv_freq = cfg.rope_base ** -(torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(cfg.seq_len).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():  # scale residual-branch outputs (GPT-2)
            if name.endswith(("attn.out.weight", "mlp.down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Return next-token logits of shape (batch, time, vocab)."""
        T = idx.shape[1]
        start = cache.length if cache is not None else 0
        assert start + T <= self.cfg.seq_len, "sequence longer than the model's context"
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.length += T
        return self.head(self.norm(x))

    def loss(self, idx: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"):
        logits = self(idx)
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                               ignore_index=tok.PAD, reduction=reduction)


def load_checkpoint(path, device="cpu") -> tuple[Pianist, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Pianist(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
