# MaMuJoCo Humanoid PPO Project

## Project Goal

Build a resume-ready reinforcement learning project around Farama Gymnasium-Robotics MaMuJoCo Humanoid:

- Stage 1: run a clear ordinary PPO baseline on Humanoid.
- Stage 2: add engineering improvements and ablations.
- Stage 3: compare single-agent PPO with MaMuJoCo multi-agent factorization ideas.
- Stage 4: write conclusions only from real experiment outputs.

The project should show algorithm understanding, implementation ability, experimental discipline, and a few thoughtful optimization/comparison points suitable for an algorithm-engineering resume.

## Current Stage

Stage 1: ordinary PPO baseline preparation.

- MaMuJoCo Humanoid smoke tests passed for `partitioning=None` and `partitioning="9|8"`.
- Current lesson: `notes/02_minimal_ppo_baseline.md`.
- AutoDL host baseline has been recorded in `AUTODL_HOST_BASELINE.md`.

## New AI Handoff

Before editing, read these files first:

1. `README.md`
2. `CHANGELOG.md`
3. `AUTODL_HOST_BASELINE.md` if it exists
4. `notes/00_project_roadmap.md`
5. The current lesson note in `notes/`
6. `autodl-project-manager/SKILL.md` if present

## Documents

- `notes/00_project_roadmap.md`: project route, experiment ladder, resume deliverables.
- `notes/01_project_start_and_env_check.md`: completed AutoDL and MaMuJoCo smoke test result.
- `notes/02_minimal_ppo_baseline.md`: current lesson, ordinary PPO minimal training loop.
- `AUTODL_HOST_BASELINE.md`: stable AutoDL hardware, CUDA, conda, and package baseline.
- `server/check_autodl_host.sh`: script for collecting stable AutoDL environment facts.
- `src/check_mamujoco_env.py`: quick MaMuJoCo Humanoid API smoke test.
- `requirements.txt`: starter dependencies, to be refined after AutoDL reports real versions.

## Directory Conventions

- `notes/`: tutorial lessons, decisions, questions, and stage summaries.
- `src/`: code that belongs to the project.
- `server/`: AutoDL helper scripts and lightweight environment reports.
- `scripts/`: local utility scripts when needed.
- `experiment_records/`: create later for repeated experiment plans and summaries.

Large runtime artifacts should stay on the AutoDL data disk, for example under `/root/autodl-tmp/Humanoid-runs/`.

## Local and AutoDL Split

- Local machine: editing, notes, Git, review, lightweight checks.
- AutoDL server: dependency installation, MuJoCo verification, PPO training, evaluation, rendering, TensorBoard, long-running jobs.

Do not infer AutoDL hardware or package versions from the local machine.

## Maintenance Rules

1. Keep this README as navigation and project rules only.
2. Put lesson details, concepts, questions, and feedback in `notes/`.
3. Before continuing after a lesson is complete, check `git status --short` and the relevant note.
4. Update `CHANGELOG.md` for meaningful changes.
5. Do not write metrics, curves, or performance conclusions without real AutoDL/server output.
6. Do not commit models, checkpoints, TensorBoard events, raw monitor logs, videos, secrets, private keys, or server access details.
7. Preserve user-written notes and comments unless they are wrong or misleading.

## Current Runnable Entry Points

After installing dependencies on AutoDL:

```bash
python src/check_mamujoco_env.py --partitioning none --steps 5
python src/check_mamujoco_env.py --partitioning "9|8" --steps 5
```

## Key References

- Farama MaMuJoCo Humanoid: <https://robotics.farama.org/envs/MaMuJoCo/ma_humanoid/>
- Farama MaMuJoCo overview: <https://robotics.farama.org/envs/MaMuJoCo/>
- Farama Gymnasium-Robotics installation: <https://robotics.farama.org/content/installation/>
