# CLAUDE.md

## Project
YOLO26n / YOLO26s drone detector for **ground-to-air** and **air-to-air** scenarios.
The source of truth for decisions is `plan.md`. Read it before starting any task.

## Language
- Respond in Korean. Write code, comments, and commit messages in English.

## Environment
- Python dependencies are managed with `uv` (`uv add`, `uv run`). Do not use bare `pip install`.
- Training framework: Ultralytics YOLO26 (NMS-free, AGPL-3.0).

## Hard rules (never break)
1. **Test sets are untouchable.** Anti-UAV300 RGB frames and the held-out AOT sequences must never be used for training, validation, hyperparameter tuning, or threshold selection.
2. **Split by video / sequence / flight, never by frame.**
3. **Deduplicate before merging** datasets (exact hash + perceptual hash).
4. **Remove leakage**: drop any AOD-4 or seraphim image that matches a test-set frame.
5. **YOLO26n and YOLO26s use the same data and the same settings.** Only the model size differs.
6. Class map is fixed: `0 drone`, `1 bird`, `2 airplane`, `3 helicopter`. Inference reports `drone` only.
7. Do not commit datasets, weights, or run outputs (`data/`, `runs/`, `*.pt`).

## Workflow
1. Check the "확인 필요" items in `plan.md` with small samples before writing `spec.md`.
2. Write `spec.md` (folder layout, conversion rules, data.yaml, training configs, eval scripts).
3. Implement step by step and confirm each step before moving on.

## Agents (`.claude/agents/`)
- `dataset-auditor`: run after any data conversion/merge to check splits, duplicates, leakage, and class mapping.
- `eval-analyst`: run after training to report drone AP/recall by size bucket, viewpoint, and false alarms.
- `ml-engineer`, `mlops-engineer`, `data-scientist`, `python-pro`, `code-reviewer`, `debugger`: general-purpose agents (third-party, see `THIRD_PARTY_NOTICES.md`).
