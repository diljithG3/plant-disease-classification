# CLAUDE.md

Guidance for Claude Code (or any coding agent) working in this repository. Read
README.md for the full roadmap and setup walkthrough — this file only covers what
README doesn't: decisions already made that shouldn't be re-litigated, and where
things currently stand.

## Design decisions (settled — don't relitigate without a good reason)

- **Dataset lives outside this project folder, never in version control.** Local:
  `~/plant_disease_dataset` by default (home-dir-relative, resolved via
  `Path.expanduser()` in `src/utils/config.py` — portable to any machine/OS with zero
  setup), overridable per-machine via the `DATA_ROOT` env var — set via a gitignored
  `.env` at the project root (loaded by `src/utils/config.py` via `python-dotenv`;
  template in `.env.example`) rather than exporting it in the shell each session.
  This dev machine's `.env` sets `DATA_ROOT=C:\Diljith_AK\projects\plant_dataset`, a
  sibling of the project folder, so the resolved dataset path is
  `plant_dataset\PlantVillage\color`. Colab: `/content/dataset` (downloaded fresh each
  session on purpose — Colab runtimes are ephemeral anyway, and local disk beats
  Drive-mounted I/O for training). Never write dataset files into this project
  folder, and never hardcode a machine-specific path in `configs/config.yaml` or
  anywhere in `src/` — that's exactly the bug this design avoids.
- **Local Python env is a venv at `.venv/` inside this project folder** (gitignored).
  Use that venv's `python`/`pip` for everything local — not any system Python. No
  config-driven override exists for this path (unlike the dataset root), it's just
  wherever each machine chooses to create it — substitute your own if it lives
  elsewhere. Registered as Jupyter kernel name `plant_disease` (display name "Plant
  Disease (venv)").
- **All paths are config-driven.** `configs/config.yaml` + `src/utils/config.py` +
  `src/utils/env.py` resolve local-vs-Colab paths automatically via `load_config()`.
  Never hardcode an absolute, machine-specific path (`/home/...`, `D:\...`,
  `/content/...`) in new code — use `load_config()` or the `DATA_ROOT`/`RUN_ENV` env
  vars (directly, or via the gitignored `.env` file) instead.
- **Uses git/GitHub, not Drive sync, for code distribution.** Colab is pull-only
  (`git clone`/`git pull`, no push credentials in Colab) — see README's "Version
  control" section for the full workflow and why (Drive sync silently corrupted files
  more than once during development).
- **Logic lives in `src/`, notebooks only call into it.** Reusable data/model/training
  code belongs in a `src/` module; notebook cells import from it and handle
  plotting/display. This keeps every phase's results reproducible outside the notebook
  UI and testable headlessly (see below).
- **The training/eval harness (Phase 3) must stay identical across Phases 4-7.** The
  entire point of building it once is that the from-scratch CNN and the three
  transfer-learning variants become fairly comparable in Phase 8. Don't fork it
  per-phase.

## Git workflow

- **Never `git commit` or `git push` without asking first, every time.** Approval
  given earlier in a session (even "yes, push this") does not carry forward to the
  next change — ask again before each commit/push. This applies regardless of how
  small or how obviously-correct the change seems, and regardless of how many times
  in a row the user has said yes before. Staging/diffing locally needs no
  permission; only the actual commit/push does.

## Commands

```powershell
$VENV = ".venv\Scripts"   # this machine's venv path (inside the project folder) -- adjust for yours

# Verify config resolution (prints resolved paths + detected env as JSON)
& $VENV\python.exe -m src.utils.config

# Fetch the dataset (idempotent — skips if already present)
& $VENV\python.exe -m src.data.fetch_data
```

Test a notebook headlessly end-to-end before considering it done (this is how every
notebook in this repo has been validated so far — catches errors that only show up on
a full top-to-bottom run). Run from the project root:

```powershell
& $VENV\python.exe scripts/run_notebook.py notebooks/<name>.ipynb
```

This also coalesces tqdm's per-update stream output into one block per run --
without it, a real training run (Phase 4+, hundreds of batches per epoch) bloats the
saved notebook with dozens of near-duplicate output fragments. Don't execute a
notebook via a raw `nbclient` one-liner instead of this script; that's how the bloat
happened once already (Phase 3's first pass) before this script existed.

(`nbclient`/`nbformat` aren't in `requirements.txt` since they're a dev/testing tool,
not a project dependency — install with `& $VENV\pip.exe install nbclient nbformat` if
missing.)

## Status

See PHASES.md for the full plan, current status, and findings for every phase —
don't duplicate that detail here. It's the single source of truth; update it whenever
a phase's status changes.

## Known quirks

- `src/utils/env.py`'s detection function is named `current__env` (double
  underscore) — that's intentional, not a typo. `src/utils/config.py` imports it by
  that exact name.
