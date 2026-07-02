# WallzAI

A clean local ML pipeline for learning Wallz.gg-style 9x9 Quoridor gameplay. Training uses an internal simulator, legal action masks, self-play, baseline evaluation, reproducible checkpoints, and a separate browser dry-run adapter.

## What changed

The new pipeline lives under `wallz_ai/` and does not depend on the live website for training. It replaces stage-specific one-sided training scripts with:

- a standalone 9x9 Wallz simulator with legal pawn moves, jumps, diagonal jumps, wall legality, and BFS path validation;
- fixed 209-action encoding with masks;
- compact policy/value observations;
- Random, shortest-path, and wall-heuristic baselines;
- lightweight masked PPO self-play in PyTorch;
- CSV logging and checkpoint save/load;
- Playwright browser adapter in dry-run mode by default.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
playwright install chromium
```

## Tests

```bash
PYTHONPATH=. pytest -q
```

## Train

Fast smoke run:

```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --steps 2000 --games-per-iter 2 --max-moves 80
```

Local longer run on Apple Silicon:

```bash
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --device auto
```

The device resolver uses `mps -> cuda -> cpu` when `device: auto`.

## Evaluate

```bash
PYTHONPATH=. python scripts/evaluate.py --checkpoint runs/wallz_ppo/checkpoints/latest.pt --games 100 --device auto
```

## Browser dry-run

```bash
PYTHONPATH=. python scripts/play_browser.py --checkpoint runs/wallz_ppo/checkpoints/latest.pt --dry-run
```

Live clicking is intentionally not implemented by default. The adapter must validate a parsed state and action legality before any browser action.

## Action encoding

- `0..80`: pawn target square, `row * 9 + col`
- `81..144`: horizontal wall at 8x8 coordinate
- `145..208`: vertical wall at 8x8 coordinate

Total action size: 209. Policies always receive a `[209]` legal action mask.

## Observation encoding

Shape: `[11, 9, 9]`, player-centric for the side to move. Default model is intentionally small (`channels: 32`, `residual_blocks: 2`) for local Apple Silicon iteration.

Channels:

1. current-player pawn
2. opponent pawn
3. horizontal walls, padded to 9x9
4. vertical walls, padded to 9x9
5. current-player goal row
6. opponent goal row
7. current-player remaining walls / 10
8. opponent remaining walls / 10
9. side-to-move plane
10. current-player distance-to-goal map / 32
11. opponent distance-to-goal map / 32

## Known limitations

- Wallz.gg exact DOM/network schema still needs validation on the live site.
- Canvas visual parsing is intentionally not implemented until structured state extraction is ruled out.
- The trainer currently uses lightweight PPO, not full AlphaZero/MCTS.
- PPO self-play is single-process for reliability; parallel rollout workers can be added later.

## Next improvements

- Add AlphaZero-style MCTS policy improvement once rules are fully verified against the site.
- Add multiprocessing self-play workers.
- Add previous-checkpoint arena gating before replacing `best.pt`.
- Implement a site-specific private-game action executor after dry-run parsing is verified.
