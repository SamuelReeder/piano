"""Train Pianist on prepared corpora; validation always uses the MAESTRO validation split.

    pianist-train --run pretrain --train-data aria,train      # pretrain on Aria-MIDI + MAESTRO
    pianist-train --run finetune --init checkpoints/pretrain/best.pt --train-data train \\
        --steps 3000 --lr 1e-4 --min-lr 1e-5 --warmup 200 --eval-every 250 --patience 3
    pianist-train --run pretrain --resume checkpoints/pretrain/last.pt --steps 30000

The defaults are the final 85M-parameter configuration. The MAESTRO-only baseline in the
README used ``--d-model 512 --n-heads 8 --batch-size 16 --grad-accum 2``.

Every option below is a command-line flag (``--batch-size 16``, ``--no-compile``, ...).
Metrics are appended to ``<out-dir>/<run>/metrics.jsonl``; the checkpoint with the lowest
validation loss is kept as ``best.pt`` and the latest state (with optimizer) as ``last.pt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pianist import tokenizer as tok
from pianist.data import Corpus, RandomWindows, eval_windows
from pianist.model import ModelConfig, Pianist


@dataclass
class TrainConfig:
    run: str = "base"
    data_dir: str = "data/processed"
    out_dir: str = "checkpoints"
    train_data: str = "train"  # comma-separated corpus names in data_dir, e.g. "aria,train"
    resume: str = ""  # continue a run: weights, optimizer, step and best validation loss
    init: str = ""  # start a new run (e.g. fine-tuning) from these weights only
    # model
    seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    # optimization
    steps: int = 18_000
    batch_size: int = 8  # x grad_accum x seq_len = 64k tokens per step
    grad_accum: int = 4
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 1_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # augmentation
    max_transpose: int = 5
    max_stretch: float = 0.05
    # bookkeeping
    eval_every: int = 500
    eval_windows: int = 512
    patience: int = 0  # stop after this many evaluations without improvement (0: never)
    log_every: int = 50
    seed: int = 0
    workers: int = 10
    compile: bool = True
    extra: dict = field(default_factory=dict, repr=False)  # filled with run metadata

    def model_config(self) -> ModelConfig:
        return ModelConfig(seq_len=self.seq_len, d_model=self.d_model, n_layers=self.n_layers,
                           n_heads=self.n_heads, dropout=self.dropout)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to ``min_lr``."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, windows: torch.Tensor, batch_size: int, device) -> float:
    """Mean per-token cross-entropy (nats) over every non-padding target in ``windows``."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total += model.loss(batch[:, :-1], batch[:, 1:], reduction="sum").item()
        count += (batch[:, 1:] != tok.PAD).sum().item()
    model.train()
    return total / count


def configure_optimizer(model: Pianist, cfg: TrainConfig, fused: bool):
    """AdamW with weight decay on matrices only (not norms or the embedding)."""
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "embed" in n]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), fused=fused)


def save(path: Path, model: Pianist, cfg: TrainConfig, step: int, val_loss: float,
         optimizer=None) -> None:
    ckpt = {"model": model.state_dict(), "model_config": model.cfg.to_dict(),
            "train_config": dataclasses.asdict(cfg), "step": step, "val_loss": val_loss}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = Path(cfg.out_dir) / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data_dir)
    corpora = [Corpus.load(data_dir / f"{name}.npz") for name in cfg.train_data.split(",")]
    train_set = RandomWindows(Corpus.concat(corpora) if len(corpora) > 1 else corpora[0],
                              cfg.seq_len, cfg.seed, cfg.max_transpose, cfg.max_stretch)
    del corpora
    val = eval_windows(Corpus.load(data_dir / "validation.npz"), cfg.seq_len, cfg.eval_windows)
    loader = iter(DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.workers,
                             pin_memory=device.type == "cuda", persistent_workers=cfg.workers > 0))

    model = Pianist(cfg.model_config()).to(device)
    optimizer = configure_optimizer(model, cfg, fused=device.type == "cuda")
    step, best, stale = 0, float("inf"), 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step, best = ckpt["step"], ckpt["val_loss"]
        print(f"resumed from {cfg.resume} at step {step} (val loss {best:.4f})")
    elif cfg.init:
        model.load_state_dict(torch.load(cfg.init, map_location=device, weights_only=True)["model"])
        print(f"initialized weights from {cfg.init}")
    compiled = torch.compile(model) if cfg.compile else model
    n_params = model.num_params()
    print(f"{n_params / 1e6:.1f}M parameters | {len(train_set.corpus)} training pieces | "
          f"{len(val)} validation windows | device {device}")
    cfg.extra = {"n_params": n_params, "torch": str(torch.__version__),
                 "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    def log(record: dict) -> None:
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    tokens_seen, t0, running = 0, time.time(), 0.0
    while step < cfg.steps:
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for _ in range(cfg.grad_accum):
            batch = next(loader).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = compiled(batch[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1), batch[:, 1:].flatten(), ignore_index=tok.PAD)
            (loss / cfg.grad_accum).backward()
            running += loss.item() / cfg.grad_accum
            tokens_seen += batch[:, 1:].numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            record = {"step": step, "train_loss": running / cfg.log_every, "lr": lr,
                      "grad_norm": grad_norm.item(), "tokens_per_s": tokens_seen / elapsed}
            print(f"step {step:6d} | loss {record['train_loss']:.4f} | lr {lr:.2e} | "
                  f"|g| {record['grad_norm']:.2f} | {record['tokens_per_s'] / 1e3:.0f}k tok/s",
                  flush=True)
            log(record)
            tokens_seen, t0, running = 0, time.time(), 0.0

        if step % cfg.eval_every == 0 or step == cfg.steps:
            val_loss = evaluate(compiled, val, cfg.batch_size, device)
            print(f"step {step:6d} | val loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}",
                  flush=True)
            log({"step": step, "val_loss": val_loss})
            if val_loss < best:
                best, stale = val_loss, 0
                save(run_dir / "best.pt", model, cfg, step, val_loss)
            else:
                stale += 1
            save(run_dir / "last.pt", model, cfg, step, val_loss, optimizer)
            if cfg.patience and stale >= cfg.patience:
                print(f"stopping: no improvement in {stale} evaluations")
                break
            t0 = time.time()  # do not count evaluation time against throughput
    print(f"done: best validation loss {best:.4f} (ppl {math.exp(best):.2f})")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        if f.name == "extra":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
