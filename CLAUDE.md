# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout — read this first (two repos, duplicated source tree)

This working directory contains **two independent git repositories** with overlapping, near-identical
source. Know which one you are touching before you edit or commit:

| Path | Git repo | Branch / remote | What it is |
|------|----------|-----------------|------------|
| `/home/emrama/Development/MARL-Economy/` | **outer** | `fatih` (this repo) | Unity visualization work + a flattened copy of the Python source |
| `/home/emrama/Development/MARL-Economy/marl_economy/` | **inner** | `main` → `idilbagubek/marl_economy` | Canonical Python simulation (its own `.git`) |

The Python source exists in **two copies** that are byte-identical today but tracked by *different* repos:

- `marl_economy/marl_economy/<pkg>` — the **nested** copy. Tracked by the **inner** repo. **This is canonical.**
- `marl_economy/env/`, `marl_economy/config/`, … — the **flattened** copy at the inner-repo root. Tracked by
  the **outer** `fatih` repo.

Which copy Python imports depends on your CWD (both define `marl_economy/__init__.py`):
- CWD = `marl_economy/` (inner repo root) → `import marl_economy.*` resolves to the **nested** copy. **Run Python here.**
- CWD = outer root → resolves to the **flattened** copy.

**Practical rule:** do Python work from inside `marl_economy/`, edit files under `marl_economy/marl_economy/…`, and
commit them in the inner repo. If you edit one copy, the other goes stale. Verify with
`git -C marl_economy status` vs. `git status`.

**Active Unity project:** `unity-MARL-EXPERIMENT/MARL-Experiment/` (Unity 6000.4.11f1 + URP). The older `unity/` and
`marl_economy/unity/` folders are **reference-only** — do not build there.

## Common commands

Run all Python commands from the inner repo root: `cd marl_economy/`.

```bash
# Install (base sim + tests). Or `pip install -r requirements.txt` for the full stack (adds torch/ray/pygame/scipy).
pip install numpy matplotlib gymnasium scipy

# Smoke test — all regimes + invariant checks (run before any long job)
python -m marl_economy.scripts.run_smoke_test

# One logged episode
python -m marl_economy.scripts.run_one_episode --regime LME --weeks 156

# Tests (pytest). Single test:
pytest tests/
pytest tests/test_invariants.py::test_food_shopping_reduces_hunger

# Evaluate all regimes (random-policy baselines + comparison tables)
python -m marl_economy.eval.eval_regimes --weeks 156 --output runs/regime_baselines

# Plots from any run dir
python -m marl_economy.eval.plot_run runs/<run_id>/
```

Training is **Ray RLlib PPO**, run as a 4-stage curriculum chain (each stage `--load-from` the previous `final`):

```bash
pip install "ray[rllib]" torch
python -m marl_economy.train.train_rllib --stage 1 --regime LME --iterations 50  --checkpoint runs/LME_stage1
python -m marl_economy.train.train_rllib --stage 2 --regime LME --iterations 50  --load-from runs/LME_stage1/final --checkpoint runs/LME_stage2
# …stages 3 (100 iters) and 4 (200 iters) follow the same pattern.

# Frozen inference on a checkpoint → writes metrics_step.csv, summary.json, snapshots.json, plots/, unity_playback.json
python -m marl_economy.eval.evaluate_checkpoint --checkpoint runs/LME_stage4/final --output runs/LME_stage4_eval
```

The `*.bat` / `*.cmd` files at both repo roots are **Windows launchers** (e.g. `train.bat LME 42` chains all four
stages). On this Linux box, call the `python -m …` commands above directly. `--regime` accepts `LME|CME|SDWE|SME`
(see Regimes below); `--seed` is supported throughout.

## Architecture (the parts that span multiple files)

A multi-agent RL simulator of a synthetic economy: **80 households, ≤12 firms, a government, a central bank**,
over ~156 weekly steps, under one of four economic regimes.

**Four agents on different timescales** (this multi-timescale cadence drives the whole step loop and the
per-policy discount/LR scaling):

| Agent | Update period | Action space | RLlib id prefix |
|-------|---------------|--------------|-----------------|
| Household | weekly (1 step) | `Discrete(6)`: Work/Shop/Steal/Rest/Invest/Clinic | `h_*` |
| Firm | every 4 weeks | `MultiDiscrete(5,5,3)` price/wage/capacity | `f_*` |
| Government | every 12 weeks | `MultiDiscrete(5,5,5,5,5,3)` 6 policy dims | `gov` |
| Central Bank | every 12 weeks | `Discrete(5)` rate delta | `cb` |

**`World` is a thin facade over `BatchedWorld(K=1)`** (`env/world.py`). All economic dynamics are implemented
**exactly once** in `env/batched_world.py` plus the vectorized `env/dynamics_*.py`, `env/routing.py`, and the
small physics modules. **Do not add simulation logic to `world.py`** — it only unwraps row 0 of the batched state.
Edit `batched_world.py` / the `dynamics_*` modules.

**State is struct-of-arrays, never `list[dict]`.** `BatchedHouseholdState` is shape `(K, N)`,
`BatchedFirmState` is `(K, F_max)` (`env/batched_state.py`). `World.h` / `World.f` are row-0 views. Per-seed
trajectories are identical across any `K` because each world gets its own `np.random.Generator` via
`SeedSequence.spawn(K)`.

**All tunable numbers live in `config/constants.py`** (~1050 lines: coefficients, bounds, thresholds, action
metadata, curriculum schedule, regime tables, PPO hyperparameters). `config/presets.py` is a thin regime
accessor over those tables. Change behavior there, not with magic numbers in the dynamics code.

**Layered subsystems** (each its own module under `env/`), composed by `batched_world.step()`:
human-condition stack — hunger (`dynamics_hunger.py`), fatigue (`dynamics_fatigue.py`), 5-state illness machine
(`illness.py`); place-based epidemic transmission `P = 1-(1-α)^k` (`epidemics.py`); exogenous disaster generator,
5 types (`disasters.py`); severity-triage clinic allocator (`clinic_capacity.py`); multi-signal economic-collapse
detector (`collapse_detector.py`); firm solvency hire-gate (`firm_solvency.py`); and a typed `EventLedger` with
~20 event types (`events.py`) whose per-week counts surface as `events_*` columns in `metrics_step.csv`.

**RLlib adapters** wrap the env two ways: `env/marl_env_rllib.py` (`MARLEconomyEnv`, single world) and
`env/marl_env_rllib_batched.py` (K-parallel inside one runner). `policy_mapping_fn` maps `h_*`→households policy,
`f_*`→firms, `gov`, `cb`.

### Regimes

Four models, each a row in `REGIME_CONFIGS` (`config/constants.py`): **LME** (Liberal Market), **CME**
(Coordinated Market), **SDWE** (Social-Democratic Welfare), **SME** (Socialist Market). The old free-form names
(`FreeMarket`/`Restricted`/`SocialDem`/`Command`) are **aliases** that still resolve, so legacy checkpoint paths
and playback files keep working. `REGIME_CONFIGS[...]["instruments"]` (`price_cap`, `subsidies`) is **declarative
metadata only** — no code path enforces it yet.

## Python → Unity bridge

Pipeline: train → checkpoint → `evaluate_checkpoint.py` (frozen inference) → a run dir →
`viz/export_unity.py::export_unity_playback()` bundles `snapshots.json` + `metrics_step.csv` + `summary.json` +
`config.json` into a single **`unity_playback.json`**. Per-week snapshots come from `World.get_snapshot()`
(`env/world.py:545`): top-level `week`, `regime`, `households[]` (position, action_label, wealth, health, hunger,
fatigue, illness, wealth_class, …), `firms[]` (position, price, cash, capacity, spawned_this_step), `institutions`
(government/central_bank/market anchor positions), `macro`, `policy`, `interest_rate`, `events`. Sample output
lives in `LME_stage4_v3_eval/`; the Unity project reads a copy at
`unity-MARL-EXPERIMENT/MARL-Experiment/Assets/Resources/unity_playback.json`.

**Hard constraint (current task):** the MARL model is slow to retrain, so **make no Python changes** — the Unity
layer must consume the existing, buggy `unity_playback.json` as-is. The exported data has known quirks the Unity
side compensates for (it does **not** use Newtonsoft — `JsonUtility` only):
- **Event-name set** is `firm_created`/`firm_closed`/`central_bank_rate_changed`/`government_policy_changed`/
  `disaster_*`/`household_death`/`household_*_crisis` (an older Unity processor listened for names that never fire).
- **Event-accumulation bug:** each week's `events` list also contains every prior week's events. Silver lining:
  the final snapshot is effectively the full history → used to build the timeline (de-duped by
  `type|week|firm_id|household_id`).
- **Disasters carry no position** (emitted at origin) → the Unity layer infers a site and a duration from sim
  constants (fire 4wk, flood 6wk, supply-shock 8wk, epidemic 12wk).
- `get_snapshot()`'s `visual_events` is always empty in the `K=1` wrapper; the sample data's events predate the
  Python-side fix.

Unity scripts live in `unity-MARL-EXPERIMENT/MARL-Experiment/Assets/Scripts/MarlEconomy/` (namespace
`MarlEconomy.UnityViz`, Assembly-CSharp, no asmdef). The scene is **asset-driven** (DenysAlmaral CityPeople =
households, Toon City Pack = buildings), wired through `MarlEconomyPrefabConfig.asset` and bootstrapped at Play by
`MarlEconomyAutoSetup` (gated to scene `SampleScene`). All runtime materials must use URP shaders
(`Universal Render Pipeline/Lit` / `/Unlit`), never Built-in `Standard`; editor menus under **MARL Economy** fix
imported pack materials for URP.
