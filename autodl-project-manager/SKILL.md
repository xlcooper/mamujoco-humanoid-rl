---
name: autodl-project-manager
description: Set up or maintain a sustainable project-management structure for coding/research projects that use a local editor plus AutoDL server workflow. Use when asked to create README/CHANGELOG/project rules, design AI handoff rules, set up AutoDL environment reporting, generate server check scripts, summarize server reports into AUTODL_HOST_BASELINE.md, or package a reusable project-management workflow for future projects.
---

# AutoDL Project Manager

Use this skill to create and maintain a lightweight, durable management layer for projects that are edited locally and run on AutoDL.

## Core Workflow

When starting or reorganizing a project:

1. Inspect existing files before editing: `git status --short`, root files, `README.md`, `CHANGELOG.md`, `server/`, and any existing environment docs.
2. Create or update the management skeleton:
   - `README.md`
   - `CHANGELOG.md`
   - `AUTODL_HOST_BASELINE.md` after a real server report exists
   - `server/check_autodl_host.sh`
   - optional `notes/`, `scripts/`, `src/`, `experiment_records/`
3. Keep root README as navigation and rules only. Put details in notes, topic docs, code comments, or optional experiment docs.
4. Treat AutoDL as the default runtime unless the user says otherwise.
5. Ask the user to run the server check script on AutoDL and push its report.
6. After the report is pushed, summarize stable facts into `AUTODL_HOST_BASELINE.md`; do not store secrets or transient process state.
7. Update `CHANGELOG.md` for every meaningful change.
8. Commit, `git pull --rebase`, and push after each update.

## Required References

Read these only when needed:

- For README, CHANGELOG, Git, code-style, notes, and AI handoff rules: `references/project-rules.md`.
- For AutoDL assumptions, environment check flow, server/data-disk conventions, and environment baseline rules: `references/autodl.md`.

## Bundled Script

Use or copy:

```text
scripts/check_autodl_host.sh
```

into the target project as:

```text
server/check_autodl_host.sh
```

Then ask the user to run on AutoDL:

```bash
bash server/check_autodl_host.sh | tee server/autodl_host_report.txt
git add server/check_autodl_host.sh server/autodl_host_report.txt
git commit -m "Record AutoDL host report"
git pull --rebase
git push
```

After pull, read `server/autodl_host_report.txt` and write a concise `AUTODL_HOST_BASELINE.md`.

## Output Discipline

Never write experiment metrics, training curves, strategy performance, or hardware conclusions as facts unless they came from a real user-provided/server-produced output.

Do not commit large runtime artifacts or secrets:

```text
models, checkpoints, replay buffers, TensorBoard event files, raw monitor logs, videos, rendered images, VNC passwords, SSH host/port details, private keys, tokens
```

## New Conversation Handoff

Ensure the target project README tells future AI agents what to read first. A minimal handoff block:

```text
New AI handoff: read README.md, CHANGELOG.md, AUTODL_HOST_BASELINE.md if present, and PROJECT_MANAGEMENT_PLAYBOOK.md or this skill before editing.
```

The rule belongs in the project README because a future conversation may not automatically reload this skill.
