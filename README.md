## Using Codex (VS Code extension)

This repo includes `AGENTS.md`, which contains Codex instructions (coding standards, how to run tests, etc.).

Suggested workflow:
1. Open the repo in VS Code.
2. Open the Codex panel.
3. Start with: “Read AGENTS.md and codex/doctor.sh, then propose a plan.”
4. Approve runs/patches as needed.
5. For follow-up iterations, update AGENTS.md (preferred) or this README section, then ask Codex to re-read it.

Tip: when asking Codex to modify code, point it to the entrypoints:
- Setup: `bash codex/setup.sh`
- Test: `bash codex/test.sh`
