# MARL Economy Simulation

Multi-agent reinforcement learning economy with Households, Firms, Government, and a Central Bank.

## Architecture

4 agent types operate on different timescales in a shared economy:

| Agent | Update Period | Obs Dim | Action Space |
|-------|---------------|---------|--------------|
| Household | Weekly (1 step) | 13D | 6 discrete (Work/Shop/Steal/Rest/Invest/Clinic) |
| Firm | Monthly (4 steps) | 10D | `MultiDiscrete(5,5,3)` for price/wage/capacity deltas |
| Government | Quarterly (12 steps) | 9D | `MultiDiscrete(5,5,5,5,5,3)` for 6 policy dimensions |
| Central Bank | Quarterly (12 steps) | 4D | `Discrete(5)` for interest-rate deltas |

The RL stack uses factorized action heads for firms and government, realistic central-bank rate moves (`+/- 25-50 bps`), and per-policy discount factors that match each agent's decision cadence.

### Action routing

`Work`, `Shop`, and `Clinic` actions all route to a specific live firm. `Work` additionally requires the firm to be solvent (firm cash + bounded credit can cover one more wage). `Steal` does **not** route to a firm; it samples a victim only on success in the firm-side phase. `Rest` and `Invest` are local-only actions.

### Household state and mechanics

- **Hunger:** Households accumulate hunger (`+0.06/week`) and only `Shop` reduces it (`−0.35` per successful purchase). Above the penalty threshold (0.5) hunger amplifies health decay; above the productivity threshold (0.6) it scales work income down toward a floor of 0.3; at the fatal threshold (0.95) hunger drives health rapidly to the death threshold. This forces real demand for goods even when alternative health-restoring actions exist.
- **Firm solvency gate on hiring:** `route_work` filters out firms whose cash + `FIRM_HIRE_CREDIT_LIMIT` cannot cover one more wage. Refused-hire households become unemployment events.
- **Multi-signal collapse detection:** terminates the episode with `economic_collapse:hidden_insolvency` or `:blocked_exit_cascade` when GDP has been zero for a configurable streak and either every alive firm is deeply negative or `F_MIN`-blocked exits keep cascading.
- **Typed event ledger:** every meaningful state transition (firm entry, exit, blocked exit, emergency entry, hire refused, household death, hunger crisis, displacement) is recorded with week + actor id. Per-week counts are exposed in `metrics_step.csv` (`events_*` columns) and episode totals in `summary.json`.
- **Cost-conservative shop / clinic:** `Shop` charges `SHOP_COST_FRACTION × price` (default 0.57, calibrated to the real-world labor-share target), `Clinic` charges `CLINIC_COST_MULTIPLIER × price`. The matched firm is credited the *actual amount paid* - the cash leak that previously made clinic visits silently destroy money has been closed.

### Regime instrument flags

`REGIME_CONFIGS["<regime>"]["instruments"]` exposes `price_cap` and `subsidies` flags per regime. These are **declarative metadata** - no code path currently enforces them. They're useful for documentation and as hooks for future policy mechanics; treat them as advisory until a price-cap module ships.

## Regimes

4 economic models with academic taxonomy + reference countries (calibrated against Table IV of the FatBar paper):

| Regime | Name | Reference Countries | Source |
|---|---|---|---|
| **LME** | Liberal Market Economy | USA, Brazil, Singapore | Hall & Soskice 2001 |
| **CME** | Coordinated Market Economy | Germany, Japan | Hall & Soskice 2001 |
| **SDWE** | Social Democratic Welfare Economy | Sweden, Norway | Esping-Andersen 1990 |
| **SME** | Socialist Market Economy | China, Vietnam | Bekkevold et al 2020 |

Anchor values (tax %, welfare, healthcare coverage %, education %, policing intensity, base wage) are sourced directly from Table IV of the paper, derived from World Bank / OECD / WHO / UNESCO data for the reference countries.

Old free-form names (`FreeMarket`, `Restricted`, `SocialDem`, `Command`) are kept as aliases that resolve to the canonical names above, so existing checkpoint paths, run dirs, and Unity playback files keep working.

## Quick Start

```bash
# Install
pip install numpy matplotlib gymnasium

# 1. Smoke test (all regimes, invariant checks)
python -m marl_economy.scripts.run_smoke_test

# 2. Run one episode with logging
python -m marl_economy.scripts.run_one_episode --regime LME --weeks 156

# 3. Evaluate all regimes with comparison tables
python -m marl_economy.eval.eval_regimes

# 4. Generate plots from a run
python -m marl_economy.eval.plot_run runs/<run_id>/

# 5. Evaluate a trained RLlib checkpoint
python -m marl_economy.eval.evaluate_checkpoint --checkpoint runs/<checkpoint_dir>/final

# 6. Train households (standalone prototype)
python -m marl_economy.train.train_standalone --stage 1 --regime LME --episodes 500

# 7. Visual demo (requires pygame)
python -m marl_economy.viz.viewer_pygame
```

Regime names are centralized in `marl_economy/config/constants.py` via:

- `REGIME_FREE_MARKET`
- `REGIME_RESTRICTED`
- `REGIME_SOCIAL_DEM`
- `REGIME_COMMAND`
- `REGIME_NAMES`

## Training Plan (4 Stages)

| Stage | Phase | Agent(s) Training | Notes |
|-------|-------|-------------------|-------|
| 1 | Household Foundation | Households | Firms/Gov/CB scripted |
| 2 | Micro Market Adaptation | Households, Firms | Households keep adapting at a reduced LR |
| 3 | Macro Policy Introduction | Households, Firms, Gov, CB | Macro agents are added without freezing prior populations |
| 4 | Joint Fine-Tuning | All | Reduced LR, increased smoothness |

## Curriculum Schedule

| Phase | Episodes | Ep Length | Description |
|-------|----------|-----------|-------------|
| 1 | 0-500 | 78 weeks | Households learn micro-behavior |
| 2 | 500-1500 | 104 weeks | Firms enter while households keep adapting |
| 3 | 1500-3000 | 156 weeks | Gov/CB join a still-learning private sector |
| 4 | 3000-5000+ | 156 weeks | Joint equilibrium |

## Key Files

```text
docs/                         # Reference documentation
  architecture.md
  BATCH_FILES.md
  notebooklm_presentation.md
  unity_integration.md
scripts/                      # Standalone validation harnesses
  stage3_test1_k1_equivalence.py
  stage3_test2_distribution.py
  stage3_test3_smoke.py
  sanity_run_v4.py
  (and other benchmark_stage*.py scripts)
tests/                        # pytest suite
  test_invariants.py
unity/                        # Unity C# integration scripts
marl_economy/
  config/
    constants.py         # Coefficients, bounds, curriculum, action metadata
    presets.py           # Thin regime accessor layer over the canonical config tables
  env/                   # Full simulation stack (20 files)
    world.py             # Core simulation (weekly tick loop)
    batched_state.py
    batched_world.py     # Vectorized K-world engine
    clinic_capacity.py
    collapse_detector.py
    disasters.py
    dynamics_fatigue.py
    dynamics_firm.py
    dynamics_household.py
    dynamics_hunger.py
    entry_exit.py
    epidemics.py
    events.py            # EventLedger and typed event records
    firm_solvency.py
    illness.py
    marl_env_rllib.py    # RLlib MultiAgentEnv wrapper (single-world)
    marl_env_rllib_batched.py  # RLlib wrapper (batched)
    metrics.py
    routing.py
    state.py
  train/
    train_standalone.py  # Lightweight REINFORCE prototype
    train_rllib.py       # Full RLlib PPO training
  eval/
    eval_regimes.py
    evaluate_checkpoint.py
    plot_run.py
    run_metadata.py
  viz/
    viewer_pygame.py
  scripts/
    run_smoke_test.py
    run_one_episode.py
```

## Default Coefficients

- Government: `a=1.2`, `b=0.9`, `c=0.8`, `d=0.6`, `e=0.9`, `u=0.5`, `zeta=1.5`
- Firms: `chi=0.35`, `eta=8.0`
- Central Bank: `rho=2.5`, `alpha_r=0.15`, `beta_r=0.10`

## RLlib Training (Full MARL)

```bash
pip install "ray[rllib]" torch
python -m marl_economy.train.train_rllib --stage 1 --regime LME --iterations 500
```

## Current Handoff / Next Work

Most of the bugs that the original handoff flagged have now been addressed in code:

1. Bankrupt firms can no longer hire (`firm_solvency.can_hire`).
2. Hidden collapse is detected by `collapse_detector.CollapseDetector` independent of `F_MIN`.
3. Real event counters live in `events.EventLedger` (module `marl_economy.env.events`), exposed both in CSV `events_*` columns and `summary.json` totals.
4. Shop / clinic accounting is cash-conservative and shopping cost rebalanced; hunger forces real consumption.
5. Government reward includes a GDP-collapse penalty so the "dead but equal" optimum is no longer rewarded.

Remaining work (rough order):

1. **Retrain.** The household observation has grown to 13D (added hunger, fatigue, and illness state), so old Stage-4 checkpoints need to be re-run.
2. **Tune hunger constants** if random play dies too fast or trained policies ignore food.
3. **Phase 2+ remaining:** food vs. luxury split and any further calibration of the fatigue/illness/epidemic/disaster modules now present in env/.

### Windows CMD Commands

Open `cmd.exe`, then run:

```cmd
cd /d C:\Users\Baris Kaan\Desktop\marl_economy_source_code
call C:\miniconda\Scripts\activate.bat marltest
```

Optional safety check before long training:

```cmd
python -m marl_economy.scripts.run_smoke_test
```

### Real Training Chain: FreeMarket First

Stage 1:

```cmd
python -m marl_economy.train.train_rllib --stage 1 --regime LME --iterations 50 --checkpoint runs\fm_stage1_real
```

Stage 2:

```cmd
python -m marl_economy.train.train_rllib --stage 2 --regime LME --iterations 50 --load-from runs\fm_stage1_real\final --checkpoint runs\fm_stage2_real
```

Stage 3:

```cmd
python -m marl_economy.train.train_rllib --stage 3 --regime LME --iterations 100 --load-from runs\fm_stage2_real\final --checkpoint runs\fm_stage3_real
```

Stage 4:

```cmd
python -m marl_economy.train.train_rllib --stage 4 --regime LME --iterations 200 --load-from runs\fm_stage3_real\final --checkpoint runs\fm_stage4_real
```

### Evaluate a Trained Checkpoint

This runs frozen inference only. It does not keep learning.

```cmd
python -m marl_economy.eval.evaluate_checkpoint --checkpoint runs\fm_stage4_real\final --output runs\fm_stage4_eval
```

Generated outputs include:

- `metrics_step.csv`
- `summary.json`
- `snapshots.json`
- plot PNGs under `plots\`

### Generate Plots Manually

If you want to regenerate plots from any run folder:

```cmd
python -m marl_economy.eval.plot_run runs\fm_stage4_eval
```

The plot set now includes reward graphs when reward columns are present in `metrics_step.csv`, including:

- `reward_households_mean.png`
- `reward_firms_mean.png`
- `reward_gov.png`
- `reward_cb.png`
- `reward_total.png`
- `rewards.png`

### Compare Regimes After the First Successful FreeMarket Run

Once the workflow looks good in one regime, repeat the training/evaluation chain for:

- `CME`
- `SDWE`
- `SME`

You can also compare random-policy baselines with:

```cmd
python -m marl_economy.eval.eval_regimes --weeks 156 --output runs\regime_baselines
```

### Unity Preparation Path

After trained evaluation is working well:

1. Use `evaluate_checkpoint.py` to generate snapshots from frozen policies.
2. Extend `world.py::get_snapshot()` with positions and event metadata.
3. Feed those snapshots into Unity using the plan in `docs/unity_integration.md`.
4. Keep Unity playback in inference mode, not online learning mode.
