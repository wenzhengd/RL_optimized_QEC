# Codex Instructions (AGENTS.md)

## Project summary
- Purpose: <one paragraph>
- Tech stack: <languages/frameworks>
- Supported OS: <mac/linux/windows>
- Repo layout: <important folders/files>

## How to run (source of truth)
When you need to set up or validate the repo, prefer these scripts:

- Setup: `bash codex/setup.sh`
- Tests: `bash codex/test.sh`
- Lint: `bash codex/lint.sh`
- Sanity check: `bash codex/doctor.sh`

If something is missing, create/repair these scripts rather than inventing ad-hoc commands.

## Working rules
1. Make small, reviewable commits/patches.
2. Do not change public APIs unless asked. If you must, update docs + tests.
3. Prefer editing existing files over creating new ones.
4. After changes, run tests (or the closest available check).
5. If you need environment variables or secrets, ask the human and provide a `.env.example`.

## Coding standards
- Formatting: <prettier/black/gofmt/etc>
- Lint: <eslint/ruff/etc>
- Types: <tsc/mypy/etc>
- Testing: <pytest/jest/etc>

## Definition of done
A task is done only if:
- tests pass (or you explain why they can't run),
- lint/format passes (or you explain),
- README updated if behavior changes.

## What to do first in a new session
1. Read README.md (if needed for user-facing usage).
2. Run `bash codex/doctor.sh` (or create it if missing).
3. Propose a short plan before large edits.
