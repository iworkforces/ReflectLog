# Build and Development Scripts

**Generated:** 2026-10-11
**Commit:** 4159261
**Branch:** develop

## OVERVIEW

Local automation + focused CI entry. Git hooks in `scripts/git-hooks/` are **copied** (not symlinked) into `.git/hooks/`.

## STRUCTURE

```
scripts/
├── setup-git-hooks.sh
├── run_platform_gates.py       # Focused CI entry
├── benchmark_remediation.py
├── complexity-analysis.sh      # BROKEN: fails `bash -n`
├── profile-server.sh           # BROKEN: zsh `print`; flags unparsed
├── test_anthropic_*.py         # Manual live scripts; not pytest-collected
├── test_cats_dogs.py
└── git-hooks/
    ├── pre-push                # typecheck + lint --all; no pytest
    └── README.md
```

## WHERE TO LOOK

| Script | Purpose |
|--------|---------|
| `setup-git-hooks.sh` | Copy hooks; skip `*.sh` / `*.md` / `*.txt` |
| `run_platform_gates.py` | `--focused` gates; `main() -> int` |
| `git-hooks/pre-push` | `./start-type-check.sh` then `./start-lint.sh --all` |
| `complexity-analysis.sh` | Writes `reports/complexity` (its docstring says `.reports/`); fails `bash -n` at line 199, so do not rely on it |

## CONVENTIONS

```bash
./scripts/setup-git-hooks.sh
./start-type-check.sh || exit 1
./start-lint.sh --all || exit 1
```

- Edit `scripts/git-hooks/`, then re-run setup. `.git/hooks/` is not VCS.
- `--all` **mutates**. `--check` is read-only.
- ANN applies to scripts: `main() -> int`.
- Root wrappers: `./start-type-check.sh`, `./start-lint.sh`, `./start-unittest.sh`.

## ANTI-PATTERNS

- Never edit `.git/hooks/` directly.
- `pr-quality.yml` is the Ubuntu typecheck, lint `--check`, and coverage workflow. `platform-storage.yml` is the OS-matrix storage gate.
- Never treat pre-push lint as `--check` — `--all` writes files, including trailing-whitespace strips in `scripts/` and `stubs/`.
- Never treat pre-push lint as a gate: `--all` Step 4 only echoes the final check, so remaining violations do not block the push. CI `--check` is the real gate.
- Pre-push still no pytest.
- `start-type-check.sh` getattr scan uses regex `ggetattr` (typo), so it matches nothing. Ruff B009 is the only guard.
- `profile-server.sh` calls zsh `print` under bash and never parses `--profile` / `--queries`.

## NOTES

`pr-quality.yml`: PRs to `develop`/`main`, and push to `main`. `platform-storage.yml`: push to `develop`/`main`, every PR, `workflow_dispatch`. Push to `develop` does not run `pr-quality`.

`git push --no-verify` exists but is not the workflow. After hook edits, re-run setup.

`cd.yml` runs via `workflow_run` after `pr-quality` and `platform-storage` succeed on a `main` push. It tags `v<pyproject version>` and creates a release; it exits 1 if that tag already points at a different SHA, so bump `pyproject.toml` before merging.

`test_cats_dogs.py` and `test_anthropic_add_flow.py` `shutil.rmtree` a relative `indexes/<workspace>` before and after. `test_anthropic_provider.py` takes credentials from the OS keychain. Default pytest never collects `scripts/`; `pytest scripts/` would.
