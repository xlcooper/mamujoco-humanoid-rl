# Project Rules Reference

Use this reference when creating or maintaining root project files.

## Root README Rules

Root `README.md` is navigation, not a notebook.

It should include:

- Project goal.
- Current stage.
- New AI handoff: which files to read first.
- Document navigation.
- Directory conventions.
- Local/server split.
- Important maintenance rules.
- Current runnable entry points if they are still current.

It should not include:

- Full experiment analysis.
- Long course explanations.
- Obsolete commands as current instructions.
- Raw logs.
- Secrets or server access details.

Recommended maintenance rules to include or adapt:

1. README only covers navigation: current stage, entry files, route, and important agreements. Put details in `notes/`, topic docs, or code comments.
2. If the project has lessons, write concepts, steps, questions, and feedback in `notes/`.
3. Before editing after a user says a lesson is complete, check `git status` and the relevant note content.
4. When a lesson/stage completes, prepare the next stage unless the user pauses.
5. Keep code readable for review and learning; prefer a few clear lines over dense one-liners.
6. Preserve user-written inline comments unless wrong or misleading.
7. Every update must be committed and pushed.
8. Do not write outputs, metrics, curves, or performance conclusions without real AutoDL/server output.
9. Treat pasted runtime errors as coming from AutoDL unless the user says otherwise.

## CHANGELOG Rules

`CHANGELOG.md` records project history:

```text
# 更新记录

## v0.1.0 - YYYY-MM-DD

- 做了什么：...
- 为什么：...
```

Keep entries concise. Do not store secrets or raw experiment logs.

## Code Style Rules

- Write for future review.
- Use explicit variables and straightforward control flow.
- Add abstractions only when they reduce real repetition or clarify ownership.
- Use comments for intent, assumptions, and non-obvious details.
- Do not delete user notes/comments just to make files look cleaner.

## Git Rules

Before edits:

```bash
git status --short
```

After edits:

```bash
git add <files>
git commit -m "<clear message>"
git pull --rebase
git push
```

Do not use destructive Git commands unless explicitly requested.

## Optional Experiment Module

Only create `experiment_records/` when the project enters repeated experiment tracking.

The design is project-specific. Keep these invariants:

- Plans and completed results are separate.
- Real output is required for conclusions.
- Git stores lightweight metadata and summaries.
- Server data disk stores large artifacts.
- Every result records enough context to reproduce: command/config, seed, environment, output path, and evaluation protocol.
