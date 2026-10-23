"""Autoregressive sampling with a KV cache, nucleus sampling and a token grammar."""

from __future__ import annotations

import torch

from pianist import tokenizer as tok
from pianist.model import KVCache, Pianist

EOS_CHECK_EVERY = 32  # tokens between host syncs to see whether the piece has ended


def _note_aligned_start(tokens: list[int], earliest: int) -> int:
    """First index >= ``earliest`` where a new note group begins (right after a DURATION)."""
    for i in range(max(earliest, 1), len(tokens)):
        if tok.DURATION_OFFSET <= tokens[i - 1] < tok.VOCAB_SIZE:
            return i
    return earliest


@torch.inference_mode()
def generate(model: Pianist, prompt: list[int] | None = None, max_new_tokens: int = 4096,
             temperature: float = 1.0, top_p: float = 0.95, min_notes: int = 0,
             generator: torch.Generator | None = None) -> list[int]:
    """Continue ``prompt`` (default: just BOS) and return the full token list.

    Pieces longer than the model's context are produced with a sliding window: when the
    cache is full, the most recent half of the context (cut at a note boundary) is
    re-encoded and generation continues from there. ``min_notes`` suppresses EOS until
    that many notes have been generated. Sampling stays on the device; the host only
    syncs every ``EOS_CHECK_EVERY`` tokens.
    """
    device = model.embed.weight.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    grammar = torch.from_numpy(tok.grammar_table()).to(device)
    is_pitch = torch.zeros(tok.VOCAB_SIZE, dtype=torch.long, device=device)
    is_pitch[tok.PITCH_OFFSET:tok.VELOCITY_OFFSET] = 1
    eos_only = torch.zeros(tok.VOCAB_SIZE, dtype=torch.bool, device=device)
    eos_only[tok.EOS] = True

    prompt = list(prompt or [tok.BOS])
    buf = torch.full((len(prompt) + max_new_tokens,), tok.PAD, dtype=torch.long, device=device)
    buf[:len(prompt)] = torch.tensor(prompt, device=device)
    length, context = len(prompt), model.cfg.seq_len
    notes = torch.zeros((), dtype=torch.long, device=device)
    cache = None
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
        for step in range(max_new_tokens):
            if cache is None or cache.length == context:
                start = 0 if length < context else _note_aligned_start(
                    buf[:length].tolist(), length - context // 2)
                cache = KVCache(model.cfg, 1, device, dtype)
                logits = model(buf[start:length][None], cache)[0, -1]
            else:
                logits = model(buf[length - 1:length][None], cache)[0, -1]

            allowed = grammar[buf[length - 1]]
            if min_notes:  # EOS stays banned until enough notes exist (or the piece has ended)
                ban = (notes < min_notes) & (buf[length - 1] != tok.EOS)
                allowed = allowed & ~(ban & eos_only)
            probs = torch.softmax(logits.float().masked_fill(~allowed, float("-inf"))
                                  / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, order = probs.sort(descending=True)
                drop = sorted_probs.cumsum(0) - sorted_probs > top_p
                probs = probs.scatter(0, order, sorted_probs.masked_fill(drop, 0.0))
            next_token = torch.multinomial(probs, 1, generator=generator)[0]
            buf[length] = next_token
            notes += is_pitch[next_token]
            length += 1
            if (step + 1) % EOS_CHECK_EVERY == 0 and (buf[:length] == tok.EOS).any():
                break
    tokens = buf[:length].tolist()
    if tok.EOS in tokens:  # drop the forced EOS padding sampled after the piece ended
        tokens = tokens[:tokens.index(tok.EOS) + 1]
    return tokens
