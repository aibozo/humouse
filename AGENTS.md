# Agent Onboarding Guide

Welcome to the **1-layer mouse GAN** project. Use this guide at the start of every session to orient yourself and locate the documentation you must consult before taking action.

## Read Before Working
1. **Project overview:** [docs/project_overview.md](docs/project_overview.md)
2. **Timeline & milestones:** [docs/timeline.md](docs/timeline.md)
3. **Task scratchpad:** [docs/task_scratchpad.md](docs/task_scratchpad.md)

## Reference Library
- **Architecture details:** [docs/architecture_gan.md](docs/architecture_gan.md)
- **Data pipeline:** [docs/data_pipeline.md](docs/data_pipeline.md)
- **Neuromotor features:** [docs/neuromotor_features.md](docs/neuromotor_features.md)
- **Dataset inventory:** [docs/data_inventory.md](docs/data_inventory.md)
- **Experiment tracking:** [docs/experiment_tracking.md](docs/experiment_tracking.md)

## Environment
- Virtual environment: `.venv` (created via `python3 -m venv .venv`).
- Activate with `source .venv/bin/activate`; deactivate using `deactivate`.
- Install deps with `.venv/bin/pip install -r requirements.txt`; regenerate `requirements.lock` after changes.

## First-Session Checklist
- Confirm repository cleanliness (`git status`).
- Review scratchpad priorities.
- Ensure datasets exist under `datasets/` and run `pytest` to check loaders/segmenters/features/models.
- For experiments, review Hydra configs under `conf/` and adjust logging (W&B) credentials; outputs under `checkpoints/` now include metrics CSVs, JSON summaries, plots, and replay buffers for both GAN and detector runs.

## Working Norms
- Maintain reproducibility: deterministic seeds, config versioning, dataset hashes.
- Track task progress in scratchpad (move items between Backlog/In Progress/Done).
- Surface blockers promptly.

## Additional Notes
- Dataset storage conventions: raw archives under `data/raw/<dataset>`; processed outputs under `data/processed/<dataset>`.
- Experiment entry points: `src/train/train_gan.py`, `src/train/train_detector.py` (Hydra-based; override `experiment=` as needed).
- Ethics reminder: use datasets and generated traces responsibly and within licence terms.

Keep AGENTS.md and linked documents up to date as the project evolves.
