# Batch files - quick reference

All batch files live at the repo root, run from there. Each one auto-locates
the conda `marltest` env's Python (where Ray + RLlib are installed). If your
marltest env lives elsewhere, edit the `PYTHON=` line in `_env.bat`.

## Pre-flight

| File | What it does | Wall time |
|---|---|---|
| `validate.bat` | Run 3 Stage 3 tests + invariants suite (pytest) + benchmark (5 steps total). Use before training or after any code change in `marl_economy/env/`. | ~30 s |

## Training

| File | What it does | Wall time |
|---|---|---|
| `quick_train.bat [REGIME] [ITERS]` | Stage-1-only sanity training. Defaults: regime LME, 10 iters. Use to verify the training pipeline works end-to-end. | ~5-15 min |
| `train.bat [REGIME] [SEED]` | Full 4-stage curriculum for ONE regime. Defaults: regime LME, seed 0. Iter counts: 50 + 50 + 100 + 200 = 400 iters. | ~12-18 h |
| `run_full_training_fixed.cmd` | Full 4-stage curriculum for **ALL FOUR** regimes (LME, CME, SDWE, SME), then evaluate + plot each. 1600 iters total. | ~48-72 h |

Examples:
```
quick_train.bat                  # LME, 10 iters
quick_train.bat SDWE 30          # SDWE, 30 iters
train.bat                        # LME seed 0, full 4 stages
train.bat CME 42                 # CME seed 42, full 4 stages
run_full_training_fixed.cmd      # everything, all regimes
```

## Inspection / debugging

| File | What it does |
|---|---|
| `episode.bat [REGIME] [WEEKS] [SEED]` | Run a scripted random-action episode. Writes a tidy `runs/episode_<REGIME>_seed<N>_<ts>/` with metrics CSV, summary JSON, config JSON, and a self-describing README.md. Run `plots.bat` on the output dir to generate plot images. Use to sanity check the env without training. |
| `plots.bat <RUN_DIR>` | Regenerate plots from a run directory's `metrics_step.csv`. |

Examples:
```
episode.bat                       # LME, 156 weeks, seed 0
episode.bat CME 100 7             # CME, 100 weeks, seed 7
plots.bat runs\LME_stage4_seed0_eval
```

## Post-training evaluation + visualization

| File | What it does |
|---|---|
| `evaluate.bat <CHECKPOINT> <REGIME> [STAGE]` | Roll out a trained checkpoint, write metrics + snapshots + plots into an eval dir. |
| `unity.bat <RUN_DIR> [OUTPUT_JSON]` | Build a Unity-ready playback JSON from an eval dir that already has `snapshots.json`. |

Typical sequence after training one regime end-to-end:
```
train.bat LME 0
evaluate.bat runs\LME_stage4_seed0\final LME 4
unity.bat   runs\LME_stage4_seed0_eval
plots.bat   runs\LME_stage4_seed0_eval
```

## Shared bits

| File | Purpose |
|---|---|
| `_env.bat` | Sourced by every other batch file. Locates the marltest Python and `cd`s into the repo root. Edit `PYTHON=` here if you change envs. |

## Tips

- `runs\` is created at runtime on first use; it is not included in the
  source distribution.
- All long-running batch files print their start + finish timestamps and
  pause on error so you can read the message.
- Output directory names always include `<regime>`, `<stage>` (when relevant),
  `seed<N>`, and a `<YYYYMMDD_HHMMSS>` timestamp, so you can identify a run
  just from the directory name.
- Each generated run dir contains a `README.md` that lists every file it
  contains plus the run's config - the directories are self-describing.
- Plots in `<run_dir>/plots/` ship with an `INFO.md` so PNGs can be traced
  back to their source run even if you copy them somewhere else.
