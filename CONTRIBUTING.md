# Contributing

A personal learning project, structured so it's easy to pick up from another machine
or hand off to another contributor (human or AI agent) without re-deriving context.

## Setup

See README.md's "Setup -- local" / "Setup -- Colab" sections.

## Conventions

- **Reusable logic goes in `src/`.** Data loading, model definitions, training/eval
  loops — anything that should behave identically whether called from a notebook, a
  script, or another phase — belongs in a `src/` module. Notebooks under `notebooks/`
  only import from `src/` and handle plotting/exploration.
- **All paths are config-driven.** Read them from `load_config()`
  (`src/utils/config.py`) — never hardcode an absolute path in new code, since the
  same code needs to run unmodified locally and in Colab.
- **One notebook per roadmap phase:** `notebooks/0N_<phase-name>.ipynb` (see
  PHASES.md for the phase list).
- **Before calling a notebook "done," execute it headlessly top-to-bottom** and
  confirm zero error outputs — see CLAUDE.md's "Commands" section for the exact
  snippet. This catches ordering bugs and stale-state issues that "run all cells
  once interactively and never re-run" hides.
- **The Phase 3 training/eval harness must stay identical through Phases 4-7.** If a
  later phase needs the harness to do something new, extend it for every phase, don't
  fork a phase-specific copy — otherwise the Phase 8 comparison across approaches
  stops being apples-to-apples.

## Adding a new phase

1. Create `notebooks/0N_name.ipynb`.
2. Put any reusable function in a new or existing `src/` module, not inline in the
   notebook.
3. Run the headless execution check.
4. Update PHASES.md with concrete findings/numbers, not just "done" — future-you (or
   the next agent) should be able to read PHASES.md alone and know what was learned,
   not just what was completed.
