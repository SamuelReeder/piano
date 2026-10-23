# Common workflows; see README.md for details. Requires uv (https://docs.astral.sh/uv/).
# TORCH=cpu for machines without an NVIDIA GPU
TORCH ?= cu128
CKPT ?= checkpoints/finetune/best.pt

.PHONY: setup data soundfont pretrain finetune samples evaluate figures test lint

setup:
	uv sync --extra $(TORCH) --extra viz

data:
	uv run pianist-prepare --aria

soundfont:
	uv run python -m pianist.render --download

pretrain:
	uv run pianist-train --run pretrain --train-data aria,train

finetune:
	uv run pianist-train --run finetune --init checkpoints/pretrain/best.pt --train-data train \
		--steps 3000 --lr 1e-4 --min-lr 1e-5 --warmup 200 --eval-every 250 --patience 3

samples:
	uv run pianist-generate $(CKPT) --num 16 --max-tokens 5000 --max-seconds 100 \
		--min-notes 300 --render mp3 --out-dir outputs/candidates
	uv run python scripts/select_sample.py outputs/candidates

evaluate:
	uv run pianist-evaluate $(CKPT) --samples outputs/candidates --out docs/results/final.json

figures:
	uv run python scripts/plot_figures.py samples/sample-15.mid \
		--run "MAESTRO only, 39M" checkpoints/maestro-only/metrics.jsonl \
		--run "Aria-MIDI pretraining, 85M" checkpoints/pretrain/metrics.jsonl

test:
	uv run pytest -q

lint:
	uv run ruff check .
