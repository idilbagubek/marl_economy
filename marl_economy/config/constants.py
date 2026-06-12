"""Global constants for the simulation: action spaces, coefficients, regime configs, RL hyperparameters."""

import numpy as np

# --- Numerical stability ---
EPSILON = 1e-6

# --- Global runtime defaults ---
DEFAULT_RANDOM_SEED = 42

# --- Discount factor ---
WEEKLY_GAMMA = 0.995
GAMMA = WEEKLY_GAMMA  # backward-compatible alias for weekly discounting

# --- Population defaults (Section 12.2) ---
DEFAULT_NUM_HOUSEHOLDS = 80
DEFAULT_NUM_FIRMS = 6
F_MIN = 2
F_MAX = 12

# Regime taxonomy follows Hall & Soskice (varieties of capitalism), Esping-Andersen
# (welfare regimes), and Bekkevold et al on the socialist market economy.
REGIME_LME = "LME"            # Liberal Market Economy: USA, Brazil, Singapore
REGIME_CME = "CME"            # Coordinated Market Economy: Germany, Japan
REGIME_SDWE = "SDWE"          # Social Democratic Welfare Economy: Sweden, Norway
REGIME_SME = "SME"            # Socialist Market Economy: China, Vietnam
REGIME_NAMES = (
    REGIME_LME,
    REGIME_CME,
    REGIME_SDWE,
    REGIME_SME,
)
DEFAULT_REGIME_NAME = REGIME_LME

# Old free-form names kept as aliases so existing run dirs and CLI flags still resolve.
REGIME_FREE_MARKET = "FreeMarket"
REGIME_RESTRICTED = "Restricted"
REGIME_SOCIAL_DEM = "SocialDem"
REGIME_COMMAND = "Command"
REGIME_ALIAS_MAP = {
    REGIME_FREE_MARKET: REGIME_LME,
    REGIME_RESTRICTED: REGIME_CME,
    REGIME_SOCIAL_DEM: REGIME_SDWE,
    REGIME_COMMAND: REGIME_SME,
    # Identity entries so callers can normalize freely.
    REGIME_LME: REGIME_LME,
    REGIME_CME: REGIME_CME,
    REGIME_SDWE: REGIME_SDWE,
    REGIME_SME: REGIME_SME,
}
REGIME_CLI_CHOICES = tuple(sorted(set(
    list(REGIME_NAMES) + [REGIME_FREE_MARKET, REGIME_RESTRICTED, REGIME_SOCIAL_DEM, REGIME_COMMAND]
)))

# Per-regime starting state.
# CALIBRATED 2026-05-16 to the user-supplied real-world dataset
# (things_to_work_on/2026-05-16_calibration_dataset_tier1-3.md). Mapping rule:
#   - tax_rate: true fraction in-sim (wages*=(1-tax)); set to envelope midpoint,
#     bounds = the dataset's effective-avg-tax envelope. (HIGH-impact correction:
#     LME was 0.08 vs real ~18-34%.)
#   - healthcare_access / education_quality: abstract [0,1] quality indices -
#     normalized across the 9-country span from beds+physicians+public-health%GDP
#     and public-education%GDP respectively.
#   - welfare: LOW-confidence (benefit replacement) - only a modest ordered nudge.
#   - policing/min_wage_floor/price_level/wage_base/interest_rate: deliberately
#     UNCHANGED here - lower data confidence and wage/price must be co-tuned to
#     the labor-share target during the structural firm-economics fix.
REGIME_CONFIGS = {
    REGIME_LME: {
        # Low tax, low welfare, broad wage spread.
        "anchors": {
            "tax_rate": 0.25,             # real LME effective tax ~18-34% (was 0.08)
            "welfare": 0.06,              # low-confidence nudge (was 0.02)
            "policing": 0.20,
            "healthcare_access": 0.55,    # beds 2.3-2.8 + US-high spend, unequal (was 0.35)
            "education_quality": 0.50,    # public edu ~3-5% GDP (was 0.45)
            "min_wage_floor": 0.10,
            "price_level": 0.50,
            "wage_base": 0.18,
            "interest_rate": 0.04,
        },
        "bounds": {
            "tax_rate": (0.18, 0.34),
            "welfare": (0.0, 0.18),
            "policing": (0.10, 0.45),
            "healthcare_access": (0.35, 0.75),
            "education_quality": (0.35, 0.65),
            "min_wage_floor": (0.05, 0.25),
            "price_level": (0.1, 2.0),
            "wage_base": (0.1, 2.0),
            "interest_rate": (0.0, 0.10),
        },
        "instruments": {"price_cap": False, "subsidies": False},
        "reference_countries": ("USA", "Brazil", "Singapore"),
        "description": "Liberal Market Economy: low tax, low welfare, wide wage spread.",
    },
    REGIME_CME: {
        # Coordinated wage bargaining, moderate tax, moderate welfare.
        "anchors": {
            "tax_rate": 0.40,             # real CME effective tax ~33-50% (was 0.22)
            "welfare": 0.10,              # low-confidence nudge (was 0.06)
            "policing": 0.45,
            "healthcare_access": 0.85,    # beds 7.8-12.6 + health spend ~9% (was 0.55)
            "education_quality": 0.55,
            "min_wage_floor": 0.11,
            "price_level": 0.55,
            "wage_base": 0.15,
            "interest_rate": 0.04,
        },
        "bounds": {
            "tax_rate": (0.33, 0.50),
            "welfare": (0.05, 0.22),
            "policing": (0.30, 0.60),
            "healthcare_access": (0.70, 0.95),
            "education_quality": (0.40, 0.70),
            "min_wage_floor": (0.07, 0.25),
            "price_level": (0.1, 2.0),
            "wage_base": (0.1, 2.0),
            "interest_rate": (0.0, 0.10),
        },
        "instruments": {"price_cap": False, "subsidies": True},
        "reference_countries": ("Germany", "Japan"),
        "description": "Coordinated Market Economy: institutional bargaining, moderate redistribution.",
    },
    REGIME_SDWE: {
        # High welfare, narrow wage spread, very high public services.
        "anchors": {
            "tax_rate": 0.40,             # real SDWE effective tax ~35-45% (kept)
            "welfare": 0.15,              # low-confidence nudge (was 0.12)
            "policing": 0.60,
            "healthcare_access": 0.88,    # physicians 4.4-5.2 + health spend ~8.5%
            "education_quality": 0.85,    # public edu 6.5-7.6% GDP - highest (kept)
            "min_wage_floor": 0.15,
            "price_level": 0.60,
            "wage_base": 0.16,
            "interest_rate": 0.025,
        },
        "bounds": {
            "tax_rate": (0.35, 0.45),
            "welfare": (0.08, 0.28),
            "policing": (0.40, 0.75),
            "healthcare_access": (0.70, 0.95),
            "education_quality": (0.65, 0.95),
            "min_wage_floor": (0.10, 0.30),
            "price_level": (0.1, 2.0),
            "wage_base": (0.1, 2.0),
            "interest_rate": (0.0, 0.10),
        },
        "instruments": {"price_cap": False, "subsidies": True},
        "reference_countries": ("Sweden", "Norway"),
        "description": "Social Democratic Welfare Economy: high redistribution, narrow wage spread.",
    },
    REGIME_SME: {
        # High tax, high policing, narrow wage spread, state-led allocation with market signals.
        "anchors": {
            "tax_rate": 0.22,             # real SME effective tax ~15-28% (was 0.55)
            "welfare": 0.12,              # low-confidence nudge (was 0.14)
            "policing": 0.85,
            "healthcare_access": 0.50,    # beds 3.1-7.2 but public health spend low ~3% (was 0.70)
            "education_quality": 0.55,
            "min_wage_floor": 0.12,
            "price_level": 0.45,
            "wage_base": 0.12,
            "interest_rate": 0.05,
        },
        "bounds": {
            "tax_rate": (0.15, 0.28),
            "welfare": (0.06, 0.25),
            "policing": (0.65, 0.95),
            "healthcare_access": (0.35, 0.70),
            "education_quality": (0.40, 0.70),
            "min_wage_floor": (0.08, 0.25),
            "price_level": (0.1, 2.0),
            "wage_base": (0.1, 2.0),
            "interest_rate": (0.0, 0.20),
        },
        "instruments": {"price_cap": True, "subsidies": True},
        "reference_countries": ("China", "Vietnam"),
        "description": "Socialist Market Economy: state-driven allocation with market price signals.",
    },
}
REGIME_ANCHORS = {
    regime_name: dict(config["anchors"])
    for regime_name, config in REGIME_CONFIGS.items()
}
REGIME_BOUNDS = {
    regime_name: dict(config["bounds"])
    for regime_name, config in REGIME_CONFIGS.items()
}
REGIME_INSTRUMENTS = {
    regime_name: dict(config["instruments"])
    for regime_name, config in REGIME_CONFIGS.items()
}
REGIME_DESCRIPTIONS = {
    regime_name: config["description"]
    for regime_name, config in REGIME_CONFIGS.items()
}
REGIME_REFERENCE_COUNTRIES = {
    regime_name: tuple(config.get("reference_countries", ()))
    for regime_name, config in REGIME_CONFIGS.items()
}

# --- Multi-timescale update periods ---
HOUSEHOLD_UPDATE_PERIOD = 1
FIRM_UPDATE_PERIOD = 4
GOV_UPDATE_PERIOD = 12
CB_UPDATE_PERIOD = 12
COMPRESSED_QUARTERLY_UPDATE_PERIOD = 6

# Discount each policy by its true decision interval, not raw weekly ticks.
HOUSEHOLD_GAMMA = WEEKLY_GAMMA ** HOUSEHOLD_UPDATE_PERIOD
FIRM_GAMMA = WEEKLY_GAMMA ** FIRM_UPDATE_PERIOD
GOV_GAMMA = WEEKLY_GAMMA ** GOV_UPDATE_PERIOD
CB_GAMMA = WEEKLY_GAMMA ** CB_UPDATE_PERIOD
POLICY_GAMMAS = {
    "pi_household": HOUSEHOLD_GAMMA,
    "pi_firm": FIRM_GAMMA,
    "pi_gov": GOV_GAMMA,
    "pi_cb": CB_GAMMA,
}

# --- Episode configuration (Addendum Section 16) ---
DEFAULT_EPISODE_LENGTH = 156
DEFAULT_EPISODE_WEEKS = DEFAULT_EPISODE_LENGTH  # alias
DEFAULT_WARMUP_WEEKS = 12
WARMUP_WEEKS = DEFAULT_WARMUP_WEEKS  # alias
RUN_PROGRESS_INTERVAL_WEEKS = 26
MASS_DEATH_THRESHOLD = 0.60
HYPERINFLATION_THRESHOLD = 0.50
HYPERINFLATION_CONSECUTIVE = 4
EARLY_TERM_PENALTY_SCALE = -10.0
# Phase 3: symmetric collapse/mass-death terminal penalty for households (gov/cb
# already get EARLY_TERM_PENALTY_SCALE). Closes the "drive collapse to end the
# episode early" exploit. Scaled to the household per-episode dense prize (~+18).
# Firm terminal penalty is DEFERRED (firm-reward cadence makes terminal delivery
# unreliable; firms are already hammered by collapsed firm rewards) - flagged.
HOUSEHOLD_EARLY_TERM_SCALE = -5.0
FIRM_EARLY_TERM_SCALE = -3.0   # reserved; not yet wired (see deferral note)
# Option C: firms running negative cash pay interest at the CB rate into the
# treasury (conserved). Makes the CB rate causally affect firms -> inflation/
# GDP/unemployment, so the CB's macro reward becomes meaningful (kept, not
# descoped). The CB rate is an ANNUAL macro rate; the sim ticks weekly, so the
# per-week charge is rate*|debt|/WEEKS_PER_YEAR. (Harsh review fix: charging
# the full annual rate every week caused ~64%/yr exponential runaway debt.)
WEEKS_PER_YEAR = 52.0
FIRM_BORROW_RATE_SCALE = 1.0 / WEEKS_PER_YEAR

# --- Script/eval defaults ---
SMOKE_TEST_DEFAULT_STEPS = 200
SMOKE_TEST_EPISODE_BUFFER_WEEKS = 20
SMOKE_TEST_MIN_EPISODE_WEEKS = 200
SMOKE_TEST_LOG_INTERVAL_STEPS = 50
SMOKE_TEST_MAX_ABS_REWARD = 100000.0
TRAIN_STAGE1_LOG_INTERVAL_EPISODES = 50
TRAIN_STAGE2_LOG_INTERVAL_EPISODES = 100
RLLIB_LOG_INTERVAL_ITERATIONS = 10
RLLIB_CHECKPOINT_INTERVAL_ITERATIONS = 25

# --- RLlib runtime defaults ---
RLLIB_TRAIN_NUM_CPUS = 15
RLLIB_EVAL_NUM_CPUS = 1
# Fewer Ray actor processes (8) each running 2 vectorized envs gives the same
# total throughput as 12x1 with lower baseline RAM.
RLLIB_TRAIN_NUM_ENV_RUNNERS = 8
RLLIB_EVAL_NUM_ENV_RUNNERS = 0
RLLIB_NUM_ENVS_PER_ENV_RUNNER = 2
RLLIB_TRAIN_NUM_GPUS_PER_LEARNER = 1
RLLIB_EVAL_NUM_GPUS_PER_LEARNER = 0
RLLIB_MAX_FIRM_AGENT_SLOTS = 200

# Firm IDs increase monotonically across an episode, so the slot table has to
# stay ahead of F_MAX even though the live firm count never exceeds it.
assert RLLIB_MAX_FIRM_AGENT_SLOTS >= F_MAX, (
    f"RLLIB_MAX_FIRM_AGENT_SLOTS ({RLLIB_MAX_FIRM_AGENT_SLOTS}) must be >= "
    f"F_MAX ({F_MAX})."
)

# --- Household action space (6 actions) ---
ACTION_WORK = 0
ACTION_SHOP = 1
ACTION_STEAL = 2
ACTION_REST = 3
ACTION_INVEST = 4
ACTION_CLINIC = 5
NUM_HOUSEHOLD_ACTIONS = ACTION_CLINIC + 1
HOUSEHOLD_ACTION_NORMALIZER = float(NUM_HOUSEHOLD_ACTIONS)
HOUSEHOLD_ACTION_LABELS = {
    ACTION_WORK: "work",
    ACTION_SHOP: "shop",
    ACTION_STEAL: "steal",
    ACTION_REST: "rest",
    ACTION_INVEST: "invest",
    ACTION_CLINIC: "clinic",
}

# Household observation layout (13D):
#   0  wealth                  7  education_quality
#   1  health                  8  age / AGE_NORMALIZER
#   2  policing                9  last_action / NUM_HOUSEHOLD_ACTIONS
#   3  tax_rate               10  hunger
#   4  mean_price             11  fatigue
#   5  mean_wage              12  illness state / NUM_ILLNESS_STATES
#   6  healthcare_access
HOUSEHOLD_OBS_DIM = 13
FIRM_OBS_DIM = 10
# Government obs adds an active-disaster count as slot 9 so policy can react to shocks.
GOV_OBS_DIM = 9
CB_OBS_DIM = 4
AGE_NORMALIZER = 1000.0

# --- Household dynamics ---
HEALTH_DECAY_BASE = 0.0004
HEALTH_DECAY_ACCESS_OFFSET = 1.5
HEALTH_DECAY_FACTOR = HEALTH_DECAY_ACCESS_OFFSET  # backward-compatible alias
HEALTH_DECAY_HEALTHCARE_FACTOR = HEALTH_DECAY_ACCESS_OFFSET  # backward-compatible alias

HEALTH_SAFETY_NET = 0.15
HEALTH_SAFETY_NET_THRESHOLD = HEALTH_SAFETY_NET  # alias

WEALTH_DRAG_CONST = 0.002
WEALTH_DRAG_BASE = WEALTH_DRAG_CONST  # alias
WEALTH_DRAG_LOW_SLOPE = 0.015
WEALTH_DRAG_LOW_COEFF = WEALTH_DRAG_LOW_SLOPE  # alias
WEALTH_DRAG_MID_SLOPE = 0.006
WEALTH_DRAG_HIGH_FIXED = WEALTH_DRAG_MID_SLOPE  # alias
WEALTH_DRAG_HIGH_EXP = 1.5
WEALTH_DRAG_HIGH_COEFF = 0.12
WEALTH_DRAG_THRESHOLD = 0.4

EXPERIENCE_CAP = 3.5
WORK_EXPERIENCE_INCREMENT = 0.01
MOTIVATION_DECAY = 0.3
MOTIVATION_WEALTH_FACTOR = MOTIVATION_DECAY  # alias
WAGE_GAP_EXPERIENCE_SCALE = 1.5
WAGE_GAP_EXP_FACTOR = WAGE_GAP_EXPERIENCE_SCALE  # alias
# Slope of the experience -> wage premium curve. Not a regime knob today.
WAGE_GAP_DEFAULT = 0.1
WORK_NOISE_STD = 0.02
WORK_NOISE_MIN = 0.9
WORK_NOISE_MAX = 1.1

INVEST_RETURN_BASE_CONST = 0.008
INVEST_RETURN_BASE = INVEST_RETURN_BASE_CONST  # alias
INVEST_RETURN_EDUCATION_COEFF = 0.03
INVEST_RETURN_EDU_COEFF = INVEST_RETURN_EDUCATION_COEFF  # alias
INVEST_RETURN_WEALTH_DECAY = 0.8
INVEST_RETURN_WEALTH_FACTOR = INVEST_RETURN_WEALTH_DECAY  # alias
INVEST_CRASH_PROB_BASE = 0.35
INVEST_CRASH_PROB_MIN = 0.05
INVEST_CRASH_PROB_MAX = 0.80
INVEST_CRASH_LOSS = 0.15

CRIME_LOGISTIC_SCALE = 6.0
CRIME_LOGISTIC_CENTER = 0.5
CRIME_WELFARE_EFFECT = 0.3
CRIME_WELFARE_FACTOR = CRIME_WELFARE_EFFECT  # alias
CRIME_STEAL_ATTEMPT_MIN = 0.02
CRIME_STEAL_ATTEMPT_MAX = 0.08

WELFARE_THRESHOLD = 0.08
WELFARE_MULTIPLIER = 1.2
POVERTY_RATE_THRESHOLD = WELFARE_THRESHOLD

# --- Treasury Citizen Dividend (welfare-valve recycle) -----------------------
# The per-world gov treasury has multiple inflows (Option-B wage-tax wedge +
# Option-C firm interest) but its ONLY outflow, welfare, is hard-gated at
# hh wealth < WELFARE_THRESHOLD. In a healthy economy poverty->0 so welfare->0
# while inflow persists => treasury hoards unrecyclably (3-Opus + 5-Opus
# reviewed; see things_to_work_on/2026-05-18_WELFARE_VALVE_fix_v2_adjudicated).
# Fix: recycle treasury to ALIVE households every step as a fixed per-hh floor
# D plus kappa*(surplus above a ceiling), affordability-clamped, never draining
# below a reserve R. Pure treasury<->hh transfer (conservation-exact, residual
# unchanged). Steady state (welfare~=0): treasury* = CEILING + (tau - N*D)/kappa -
# FINITE and distribution-INDEPENDENT (D is the tau-independent behavioral knob;
# kappa only sets the surplus-drain half-life ln2/kappa). All four are economic
# operating-point knobs; conservation/determinism are invariant to them.
# DISABLED 2026-05-18 pending welfare/treasury RE-VALIDATION: the
# welfare-valve premise was REFUTED by the pre-retrain RE-review (2nd-F3 -
# the welfare gate is intra-step so poverty_rate==0 =/=> welfare~=0; treasury
# deficits/humps, not a monotone hoard; treasury invisible to all
# rewards/obs). FLOOR=KAPPA=0 => payout==0 => the phase-11b TCD block is an
# exact no-op (early-continue on payout<=0); the per-week treasury/tax/
# welfare/dividend instrumentation is RETAINED (now the trustworthy signal
# for the re-validation). Restore these (D~=0.010, kappa~=0.10) only if a binding,
# objective-relevant treasury defect is PROVEN on direct post-Fix-B data.
TREASURY_DIVIDEND_FLOOR = 0.0       # D (DISABLED - see note above)
TREASURY_DIVIDEND_KAPPA = 0.0      # kappa (DISABLED - see note above)
TREASURY_DIVIDEND_CEILING = 6.0     # surplus above this is bled out via kappa
TREASURY_DIVIDEND_RESERVE = 1.0     # never dividend treasury below this buffer
# ---------------------------------------------------------------------------
CLINIC_COST_MULTIPLIER = 1.5
# Shopping = basic food at a fraction of the firm's posted price. This fraction
# is ALSO the firm's revenue capture per unit, so under wage conservation it
# equals the model's labor share at competitive break-even (proof: at break-even
# q*P*phi = gross_comp, so labor_share = gross_comp/(q*P) = phi, any production
# exponent). Set to the real-world labor-share target (dataset: 0.55-0.60).
# CALIBRATION/KNOWN TENSION: this constant is OVERLOADED - it also sets the
# household food price (shop_cost = price*phi). Raising 0.30->0.57 ~doubles
# household food spend; cross-check vs the Engel coefficient at retrain. If the
# labor-share and Engel targets cannot both be met by one constant, split into
# two (firm_revenue_capture vs household_food_price_frac) - only if Engel fails.
SHOP_COST_FRACTION = 0.57
SHOP_HEALTH_GAIN = 0.02
CLINIC_HEALTH_GAIN_SCALE = 0.05
HEALTH_SAFETY_NET_BOOST_SCALE = 0.02
FAILED_STEAL_HEALTH_PENALTY = 0.01
# A3 complement: tainted-wealth removes theft's WEALTH-channel reward, but the
# consumption-survival value (stolen cash -> food/clinic -> fed/health level
# reward + avoided death) flows through the health/hunger channel which no
# wealth-accounting can reach. This flat reward penalty on a SUCCESSFUL steal
# prices that act directly. ~0.6 dominates the per-steal consumption value
# (~0.05 wealth) while staying within the rescaled cliff budget (death -2.0).
# Harsh-review: 0.6 was far below the avoided death/bankruptcy a desperate
# thief dodges, and serial theft compounded faster than a linear 0.6. Raised
# to 1.5 (~ the bankruptcy it can dodge, within the death-cliff budget).
# FOLLOW-UP (flagged, NOT done): an ESCALATING/recidivist penalty (grows with
# prior steals) is the principled fix for serial theft - needs per-household
# steal-count state. Final value is a retrain-calibration question.
STEAL_REWARD_PENALTY = 1.5
# A3 carry penalty (lambda): subtracted as lambda*(currently-held tainted/stolen wealth)
# from the household reward EVERY step the stolen money is still held. A3
# excludes stolen money from the WEALTH channel, but tainted money still buys
# real SHOP/CLINIC -> multi-week health+hunger reward; the one-time
# STEAL_REWARD_PENALTY does not price that. This term claws back that value
# over exactly the weeks the money is held (taint falls only via genuine
# consumption_outflow). Calibration rationale: a steal yields tainted in [0.02,
# 0.08]; a stolen-funded SHOP's per-step consumption reward ~= FED(0.30*Delta hunger)
# +health ~= 0.1-0.3. lambda*tainted >~ that per-step value => lambda~=0.2/0.064~=3; the
# suicide-safety bound is lambda < |DEAD_STATE_REWARD|(0.45)/typical_tainted(0.08)
# ~= 5.6 (else dying could beat carrying). 2.0 sits safely below the bound and
# claws back most multi-week value; final value is a retrain-calibration knob
# (safe range ~= [1.5, 3.5]). Zero effect on non-thieves (tainted==0).
STEAL_TAINT_CARRY_PENALTY = 2.0
REST_HEALTH_RECOVERY = 0.005
LOG_UTILITY_HEALTH_WEIGHT = 2.0

# HEALTH_BONUS_* retained for any external importer but NO LONGER APPLIED in the
# household reward (Phase 3): it was redundant with / dominated by the level term.
HEALTH_BONUS_THRESHOLD = 0.7
HEALTH_BONUS = 0.01
DEATH_THRESHOLD = 0.01
# Phase 3 death model: a ONE-TIME transition penalty when health crosses below
# DEATH_THRESHOLD, then a bounded PERSISTENT dead-state reward (death is not
# absorbing in dynamics - the safety net can revive). Replaces the old
# "-8..-5 EVERY week dead" which created a ~-740 tail dominating all learning.
# No-suicide-incentive constraint: DEAD_STATE_REWARD < worst sustainable-alive
# per-step = -0.5*(HEALTH_LEVEL_WEIGHT+FED_WEIGHT) = -0.30 at 0.30/0.30 -> use
# -0.45 (0.15 margin). episode_progress scaling dropped.
DEATH_TRANSITION_PENALTY = -2.0
DEAD_STATE_REWARD = -0.45
BANKRUPTCY_THRESHOLD = 0.005
# Phase 3: one-time transition penalty (Delta log-utility already penalizes ongoing
# low wealth; no persistent bankruptcy term needed).
BANKRUPTCY_TRANSITION_PENALTY = -1.5
DISPLACEMENT_PENALTY = -0.05

# --- Household wellbeing "level" reward terms (added 2026-05-16) ---
# The base household reward is a telescoping log-utility *delta*, so sustaining
# a healthy/fed state earns ~0. These add a per-week LEVEL signal so a genuinely
# healthy population scores positive and a starving one scores negative. Bounded
# in [-(W_HEALTH+W_FED)*0.5, +(W_HEALTH+W_FED)*0.5] = [-0.30, +0.30] per week
# at the Phase-3 weights. Raised 0.20->0.30 so the dense healthy gradient is
# within ~1 order of the (now rescaled) catastrophe penalties and learnable.
# Applied BEFORE the bankruptcy/death overrides so those remain hard overrides.
HOUSEHOLD_HEALTH_LEVEL_WEIGHT = 0.30   # reward per unit health above neutral
HEALTH_LEVEL_NEUTRAL = 0.5             # health at/below this contributes <= 0
HOUSEHOLD_FED_WEIGHT = 0.30            # reward for being fed (low hunger)
HUNGER_NEUTRAL = 0.5                   # hunger above this contributes < 0
# CALIBRATION TARGET: HUNGER_NEUTRAL / HOUSEHOLD_FED_WEIGHT should track the
# real food share of household budget (Engel coefficient) once data arrives.

# Hunger drives the demand for goods. Households accumulate hunger every week and
# only Shop reduces it; rest and clinic do not feed.
HUNGER_INIT_MIN = 0.10
HUNGER_INIT_MAX = 0.30
HUNGER_RISE_PER_WEEK = 0.06
HUNGER_DROP_FROM_FOOD = 0.35
HUNGER_HEALTH_PENALTY_THRESHOLD = 0.50
HUNGER_HEALTH_PENALTY_RATE = 0.02
HUNGER_PRODUCTIVITY_THRESHOLD = 0.60
HUNGER_PRODUCTIVITY_FLOOR = 0.30
HUNGER_FATAL_THRESHOLD = 0.95
# FIX C (2026-05-18 v2, 5-Opus-reviewed): was 0.50 - a DISCONTINUOUS 28x jump
# at hunger>=0.95 (0.018->0.50/wk) made starvation an absorbing death state
# (health->0 in 1-2 wk, irreversible even if food later restored). Now the
# max of a CONTINUOUS linear ramp over (0.95,1.0] (see dynamics_hunger.py).
# 0.08/wk: chronic starvation still lethal (~7 wk) but a re-fed household
# recovers (verified). Health-only (conservation-neutral).
HUNGER_FATAL_HEALTH_DECAY = 0.08
HUNGER_CRISIS_THRESHOLD = 0.85

# Firm hiring eligibility. A bounded credit line lets a firm ride out one slow
# week; past the insolvency threshold it stops hiring entirely.
FIRM_HIRE_CREDIT_LIMIT = 5.0
FIRM_HIRE_MIN_CAPACITY = 0.5
FIRM_INSOLVENT_CASH_THRESHOLD = -2.0

# Fatigue rises with work and falls with rest. Calibration target is the 5-on-2-off
# work week: one rest week clears about three work weeks of accumulated fatigue.
FATIGUE_INIT_MIN = 0.05
FATIGUE_INIT_MAX = 0.20
FATIGUE_RISE_FROM_WORK = 0.12         # one work week ~= +12% fatigue
FATIGUE_DROP_FROM_REST = 0.40         # one rest week clears ~3 work weeks
FATIGUE_DROP_FROM_CLINIC = 0.10       # clinic helps modestly
FATIGUE_PRODUCTIVITY_THRESHOLD = 0.55
FATIGUE_PRODUCTIVITY_FLOOR = 0.40     # at fatigue=1.0, work earns 40% of normal
FATIGUE_HEALTH_PENALTY_THRESHOLD = 0.70
FATIGUE_HEALTH_PENALTY_RATE = 0.005
FATIGUE_ILLNESS_RISK_MULT = 2.0

# Illness state machine. Transition probabilities below are roughly tuned to
# common-cold attack rate (~2-3 episodes/adult/year) and untreated severe
# respiratory mortality (5-15%/wk in low-resource settings).
ILLNESS_HEALTHY = 0
ILLNESS_EXPOSED = 1
ILLNESS_MILD = 2
ILLNESS_SEVERE = 3
ILLNESS_RECOVERING = 4
NUM_ILLNESS_STATES = 5
ILLNESS_LABELS = {
    ILLNESS_HEALTHY: "healthy",
    ILLNESS_EXPOSED: "exposed",
    ILLNESS_MILD: "mild",
    ILLNESS_SEVERE: "severe",
    ILLNESS_RECOVERING: "recovering",
}

# Per-week transition probabilities.
ILLNESS_BASE_INFECTION_RATE = 0.005
ILLNESS_EXPOSED_TO_MILD = 0.55
ILLNESS_EXPOSED_TO_RECOVERED = 0.30   # asymptomatic resolution
ILLNESS_MILD_TO_SEVERE = 0.08
ILLNESS_MILD_TO_RECOVERING = 0.40
ILLNESS_SEVERE_TO_RECOVERING_NO_CARE = 0.10
ILLNESS_SEVERE_TO_RECOVERING_WITH_CARE = 0.40
ILLNESS_SEVERE_DEATH_NO_CARE = 0.10
ILLNESS_SEVERE_DEATH_WITH_CARE = 0.02
ILLNESS_RECOVERING_TO_HEALTHY = 0.50
ILLNESS_REST_RECOVERY_BONUS = 0.15    # rest action while sick adds to recovery prob

ILLNESS_MILD_PRODUCTIVITY = 0.70
ILLNESS_SEVERE_PRODUCTIVITY = 0.20
ILLNESS_RECOVERING_PRODUCTIVITY = 0.85

ILLNESS_MILD_HEALTH_DECAY = 0.003
ILLNESS_SEVERE_HEALTH_DECAY = 0.025

ILLNESS_HUNGER_RISK_MULT = 2.0
ILLNESS_FATIGUE_RISK_MULT = 2.0
ILLNESS_LOW_HEALTH_RISK_MULT = 3.0
ILLNESS_LOW_HEALTH_THRESHOLD = 0.30

# Place-based transmission. P(infection at place | k infectious attendees) = 1 - (1 - alpha)^k.
# alpha is tuned so a workplace with one infectious colleague gives an effective R0 in [1.5, 2].
EPIDEMIC_PLACE_HOME = "home"
EPIDEMIC_PLACE_WORKPLACE = "workplace"
EPIDEMIC_PLACE_MARKET = "market"
EPIDEMIC_PLACE_CLINIC = "clinic"
EPIDEMIC_PLACE_PUBLIC = "public"
EPIDEMIC_PLACES = (
    EPIDEMIC_PLACE_HOME,
    EPIDEMIC_PLACE_WORKPLACE,
    EPIDEMIC_PLACE_MARKET,
    EPIDEMIC_PLACE_CLINIC,
    EPIDEMIC_PLACE_PUBLIC,
)
EPIDEMIC_TRANSMISSION_ALPHA = {
    EPIDEMIC_PLACE_HOME: 0.005,         # low - small household exposure
    EPIDEMIC_PLACE_WORKPLACE: 0.020,    # medium
    EPIDEMIC_PLACE_MARKET: 0.015,       # medium
    EPIDEMIC_PLACE_CLINIC: 0.040,       # HIGH - sick agents concentrate here
    EPIDEMIC_PLACE_PUBLIC: 0.008,       # low-medium
}
EPIDEMIC_HEALTHCARE_DAMP = 0.5         # higher healthcare_access damps transmission

# Clinic capacity, scaled by healthcare_access. Anchored to OECD hospital-bed data
# (4.4 beds/1000 average, ~5-day stay) but compressed to a tractable per-week budget.
CLINIC_CAPACITY_BASE = 30                  # baseline weekly slots
CLINIC_CAPACITY_HEALTHCARE_BONUS = 60      # extra slots, scaled by healthcare_access
CLINIC_OVERCROWD_TRANSMISSION_MULT = 1.5   # overcrowded clinic boosts disease transmission
CLINIC_OVERCROWD_THRESHOLD = 1.0           # used > capacity * threshold

# Per-week disaster probabilities. Tuned so a 156-week episode sees ~1-2 events
# of mixed type on average - enough to test resilience without dominating the run.
DISASTER_TYPES = ("earthquake", "fire", "flood", "supply_shock", "epidemic_wave")
DISASTER_PROB_PER_WEEK = {
    "earthquake": 0.0008,
    "fire": 0.0025,
    "flood": 0.0015,
    "supply_shock": 0.0035,
    "epidemic_wave": 0.0020,
}
# Earthquake: damages firm capacity + inventory + injures households.
EARTHQUAKE_FIRM_CAPACITY_LOSS_FRAC = 0.30
EARTHQUAKE_FIRM_INVENTORY_LOSS_FRAC = 0.50
EARTHQUAKE_FIRM_CASH_LOSS = 1.0
EARTHQUAKE_HOUSEHOLD_INJURY_PROB = 0.20
EARTHQUAKE_HOUSEHOLD_HEALTH_LOSS = 0.30
# Fire: closes 1-2 firms for several weeks.
FIRE_FIRMS_AFFECTED_RANGE = (1, 3)
FIRE_FIRM_DOWNTIME_WEEKS = 4
# Flood: closes more firms for longer; displaces a fraction of households.
FLOOD_FIRMS_AFFECTED_RANGE = (2, 4)
FLOOD_FIRM_DOWNTIME_WEEKS = 6
FLOOD_HOUSEHOLD_DISPLACE_PROB = 0.10
# Supply shock: temporary price spike + capacity drag.
SUPPLY_SHOCK_DURATION_WEEKS = 8
SUPPLY_SHOCK_PRICE_MULT = 1.40
SUPPLY_SHOCK_CAPACITY_PENALTY = 0.70
# Epidemic wave: external seeding of infectious agents + transmission boost.
EPIDEMIC_WAVE_DURATION_WEEKS = 12
EPIDEMIC_WAVE_INITIAL_INFECTED_FRAC = 0.05
EPIDEMIC_WAVE_INFECTION_BOOST = 4.0

# Disaster firm-side bookkeeping.
FIRM_DOWNTIME_DEFAULT = 0  # firm field; 0 = operating, >0 = closed weeks left

# Hidden-collapse detector. Triggers when GDP has been zero for a full quarter
# and either every alive firm is deeply negative or F_MIN keeps blocking exits.
COLLAPSE_GDP_ZERO_WEEKS = 12
COLLAPSE_NEGATIVE_CASH_THRESHOLD = -10.0
COLLAPSE_BLOCKED_EXIT_RECENT_WEEKS = 12
COLLAPSE_BLOCKED_EXIT_THRESHOLD = 8

# Humanitarian-collapse trigger (added 2026-05-16). The GDP/cash/blocked-exit
# triggers above are blind to a "mass-death subsistence" economy where firms
# churn (so cash never deep-negative) and intermittent tiny GDP keeps the
# zero-streak from accumulating. A 2026-05-16 eval ran 156 wk with mean health
# ~0.12 and 4220 deaths yet tripped nothing. This independent trigger fires when
# mean household health stays catastrophically low for a sustained period,
# regardless of GDP/firm state.
COLLAPSE_LOW_HEALTH_THRESHOLD = 0.25   # mean household health <= this = crisis
COLLAPSE_LOW_HEALTH_WEEKS = 12         # sustained for a full quarter -> collapse
# CALIBRATION TARGET: tie COLLAPSE_LOW_HEALTH_THRESHOLD to a real population
# health/mortality floor (e.g. crude death rate / life-expectancy) once data
# arrives. 0.25 is a conservative "unambiguous catastrophe" placeholder.

# --- Firm dynamics (Section 5) ---
# FIX B (2026-05-18 v2, 5-Opus-reviewed): productivity x3. Diagnosis: firms
# cannot break even by construction (q=min(prod*sqrt(hires),cap) -> revenue proportional to sqrt(h)
# while gross wage bill proportional to h), so the economy under-supplies food -> mass
# starvation. Break-even h_max=(prod*P*SHOP_COST_FRACTION/(W*kappa))^2; at prod=1
# no hires>=1 is profitable. prod median 1.0->3.0 reopens a wide profitable
# band (h~=1-6) and lifts sustainable supply ~=20 u/wk vs ~14 subsistence.
# Conservation-neutral (productivity sets goods/stock, not money).
FIRM_PRODUCTIVITY_DEFAULT = 3.0
FIRM_PRODUCTIVITY_MIN = 1.5
FIRM_PRODUCTIVITY_LOGNORMAL_MEAN = 1.0986122886681098  # ln(3): spawned-firm median = 3.0
FIRM_PRODUCTIVITY_LOGNORMAL_SIGMA = 0.2
FIRM_FIXED_COST = 0.005
FIRM_CAP_COST = 0.002
# FIX D (2026-05-18 v2): capacity up so Fix-B output (prod*sqrt(h), up to ~6 at
# h>=4) is NOT clipped by cap. FIRM_CAPACITY_MIN 1.0->3.0 is the load-bearing
# change (capacity smooths toward a policy target with floor MIN; a shrunk
# firm must still supply ~3 u/wk). Overhead = FIRM_FIXED_COST+FIRM_CAP_COST*
# cap -> at cap 8 = 0.021/wk, negligible vs ~1.1/wk revenue. Conserved
# (capacity != money); overhead stays a C1-ledgered DESTROYED site.
FIRM_CAPACITY_MIN = 3.0
FIRM_CAPACITY_SMOOTH = 0.3
FIRM_CAPACITY_BETA = FIRM_CAPACITY_SMOOTH  # alias
FIRM_DEFAULT_SMOOTHNESS_SCALE = 1.0
FIRM_JOINT_FINE_TUNING_SMOOTHNESS_SCALE = 1.5
FIRM_BANKRUPTCY_PENALTY = 1.5   # Phase 3: was 10.0 - a single -10 spike dwarfed
# the dense +/-0.5 firm-activity signal so PPO only learned bankruptcy-avoidance.
# 1.5 keeps a clear deterrent within ~3x the activity reward -> learnable.
FIRM_PRICE_FLOOR = 0.01
FIRM_WAGE_FLOOR = 0.01
FIRM_STOCK_TRANSACTION_UNITS = 1.0
FIRM_REWARD_MIN_SALES_NORMALIZER = FIRM_STOCK_TRANSACTION_UNITS
FIRM_MARKET_NOISE_STD = 0.02

FIRM_EXIT_CONSECUTIVE_WEEKS = 8
# S1 FIX (2026-05-18, 5-Opus re-review CRITICAL): the firm bankruptcy penalty
# must fire on SUSTAINED insolvency, not an instantaneous cash<0 snapshot. Due
# to firm cash timing (one big monthly gross wage vs a weekly trickle of sale
# revenue, small buffer, Option-C interest), a cash-flow-PROFITABLE firm sits
# transiently cash<0 most weeks -> the old instantaneous penalty made firm
# reward ANTI-correlated with firm health (~=94% of monthly firm rewards
# negative while the economy thrived) -> a retrained firm policy would learn
# cash-hoarding/under-hiring and re-throttle food supply. Gate the penalty on
# `negative_cash_weeks` (the same counter the exit predicate maintains every
# step). S1 RE-REVIEW (2 Opus converged): a HARD STEP at a single threshold
# is exploitable - a firm running neg,neg,neg,pos resets the shared counter
# and evades BOTH the penalty AND the 8-wk exit forever (sawtooth), and a
# step aliases against the monthly (period-4) reward sampling. FIX: a MONOTONE
# LINEAR RAMP, penalty = FIRM_BANKRUPTCY_PENALTY * clip((ncw - GRACE)/(FULL -
# GRACE), 0, 1). GRACE=3 = the benign monthly-wage trough (a healthy firm is
# mechanically cash<0 for <=3 consecutive wks: monthly gross wage debit vs
# weekly revenue trickle) -> 0 penalty at/below 3. FULL=7 = one week before
# FIRM_EXIT_CONSECUTIVE_WEEKS(8) -> full penalty just before exit. The ramp
# has no cliff for a sawtooth to straddle, gives a smooth gradient toward
# solvency, and ramps IN SYNC with the exit countdown. Reward-only ->
# money-conservation-neutral, deterministic. Endpoints = retrain-calib.
FIRM_BANKRUPTCY_GRACE_WEEKS = 3
FIRM_BANKRUPTCY_FULL_WEEKS = 7
FIRM_ENTRY_THRESHOLD = 0.3
FIRM_ENTRY_PROB = 0.5
FIRM_ENTRY_LOOKBACK = 8
FIRM_ENTRY_WINDOW_WEEKS = FIRM_ENTRY_LOOKBACK  # alias

FIRM_INIT_CASH = 0.5
FIRM_INIT_CAPACITY = 8.0   # FIX D: was 5.0 - headroom for Fix-B output (entry firms)
FIRM_INIT_STOCK = 3.0
FIRM_INIT_NOISE = 0.05
EMERGENCY_ENTRY_BUFFER = 2
WORLD_INIT_FIRM_CASH_JITTER = 0.3
WORLD_INIT_FIRM_CAPACITY = 8.0   # FIX D: was 3.0 - headroom for Fix-B output (world-init firms)
WORLD_INIT_FIRM_STOCK = 5.0
WORLD_INIT_HOUSEHOLD_WEALTH_MIN = 0.2
WORLD_INIT_HOUSEHOLD_WEALTH_MAX = 0.8
WORLD_INIT_HOUSEHOLD_HEALTH_MIN = 0.5
WORLD_INIT_HOUSEHOLD_HEALTH_MAX = 1.0
FIRM_NEUTRAL_ACTION = (2, 2, 1)

# --- Observation / classification defaults ---
OBS_FALLBACK_POLICING = 0.3
OBS_FALLBACK_TAX_RATE = 0.1
OBS_FALLBACK_HEALTHCARE_ACCESS = 0.3
OBS_FALLBACK_EDUCATION_QUALITY = 0.3
WEALTH_CLASS_POOR_MAX = 0.15
WEALTH_CLASS_MIDDLE_MAX = 0.4
WEALTH_CLASS_RICH_MAX = 0.8
MARKET_DEFAULT_PRICE = 0.5
MARKET_DEFAULT_WAGE = 0.4

# --- Routing layer ---
KAPPA_W = 5.0
KAPPA_WAGE = KAPPA_W  # alias
KAPPA_P = 5.0
KAPPA_PRICE = KAPPA_P  # alias
ROUTING_MIN_STOCK = FIRM_STOCK_TRANSACTION_UNITS

# --- Reward coefficients ---
# Government
GOV_COEFF_A = 1.2
GOV_A = GOV_COEFF_A
GOV_COEFF_B = 0.9
GOV_B = GOV_COEFF_B
GOV_COEFF_C = 0.8
GOV_C = GOV_COEFF_C
GOV_COEFF_D = 0.6
GOV_D = GOV_COEFF_D
GOV_COEFF_E = 0.9
GOV_E = GOV_COEFF_E
GOV_COEFF_U = 0.5
GOV_U = GOV_COEFF_U
GOV_COEFF_ZETA = 1.5
GOV_ZETA = GOV_COEFF_ZETA
# Phase 3 (#4 RELATIVE): the absolute floor 0.02 sat ~4x below healthy gdp_pc so
# the penalty was identically 0 through the whole collapse. Now RELATIVE: a slow-
# decaying per-world peak of gdp_pc is the reference; penalty ramps as gdp_pc
# falls below GDP_COLLAPSE_RELATIVE_FRAC * peak (robust to the unknown absolute
# scale, which also shifts under Option B). GDP_COLLAPSE_FLOOR kept as a tiny
# absolute backstop. Plus a CONTINUOUS per-gov-step health-floor penalty so a
# sick economy is punished every decision, not only at the (inert) floor.
GDP_COLLAPSE_FLOOR = 0.02
GDP_COLLAPSE_PENALTY = 8.0
GDP_COLLAPSE_RELATIVE_FRAC = 0.5    # collapse = gdp_pc < 0.5 * trailing peak
# Decay is applied PER GOV CALL (~every 12 wk, ~13x/episode), NOT per week.
# 0.98 (harsh-review bug) barely decayed in an episode so one transient spike
# poisoned the whole run; 0.85 -> peak ~halves in ~4-5 quarters. NOTE: a pure
# relative reference cannot, by construction, penalize a permanently-flat-low
# (never-grew) economy - the health/poverty/unemployment terms + a post-retrain
# absolute calibration are the mitigations (chicken-egg: needs healthy gdp_pc).
GDP_PC_PEAK_DECAY = 0.85           # per-gov-call decay of the peak reference
GOV_HEALTH_FLOOR = 0.40            # mean health below this is punished every step
# Harsh-review fix: 3.0 dwarfed every other gov term (~0.1) with a sharp kink;
# 1.5 keeps it a strong but not totally-dominant continuous signal.
GOV_HEALTH_FLOOR_PEN = 1.5         # continuous: -PEN * max(0, FLOOR - avg_health)

# Firm
FIRM_COEFF_CHI = 0.35
FIRM_CHI = FIRM_COEFF_CHI
FIRM_COEFF_ETA = 8.0
FIRM_ETA = FIRM_COEFF_ETA
# Firm "dead-economy" shaping (added 2026-05-16). The old reward was
# profit/max(sales,1) with no term for output scale, so "shrink to F_MIN and
# idle" was a flat local optimum indistinguishable from a healthy firm. These
# reward real sales volume and explicitly penalize a firm that sells ~nothing.
FIRM_ACTIVITY_BONUS = 0.5             # max monthly reward for full sales activity
FIRM_ACTIVITY_REF_SALES = 12.0        # monthly sales at which activity saturates
FIRM_DEAD_ECONOMY_PENALTY = 1.0       # max penalty when a firm sells ~nothing/month
# Smooth ramp instead of a binary <1-sale cliff: a 2026-05-16 red-team found the
# binary form was gamed by selling exactly 1 token unit/month to zero the whole
# penalty (+1.0 swing). Penalty now fades linearly from full at 0 sales to 0 at
# FIRM_DEAD_SALES_RAMP, so a token sale only buys ~1/RAMP of relief.
FIRM_DEAD_SALES_RAMP = 12.0           # sales at which the dead-economy penalty hits 0
FIRM_DEAD_SALES_MIN = 1.0             # (legacy, retained; superseded by the ramp)
# CALIBRATION TARGET: FIRM_ACTIVITY_REF_SALES should track real demand implied
# by the Engel food share x population; FIRM_FIXED_COST/FIRM_CAP_COST vs labor
# share once data arrives.

# Central Bank
CB_COEFF_RHO = 2.5
CB_RHO = CB_COEFF_RHO
CB_INFLATION_TARGET = 0.02
CB_COEFF_LAMBDA = 0.5
CB_LAMBDA = CB_COEFF_LAMBDA
CB_COEFF_NU = 0.3
CB_NU = CB_COEFF_NU

CB_ALPHA_R = 0.15
CB_BETA_R = 0.10
CB_RATE_MIN = 0.0
CB_RATE_MAX = 0.20
MACRO_DEFAULT_SMOOTHNESS_SCALE = 1.0

# --- Firm action space ---
FIRM_PRICE_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
FIRM_WAGE_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
FIRM_CAPACITY_DELTAS = np.array([-1.0, 0.0, 1.0])
FIRM_ACTION_DIMS = (
    len(FIRM_PRICE_DELTAS),
    len(FIRM_WAGE_DELTAS),
    len(FIRM_CAPACITY_DELTAS),
)
NUM_FIRM_ACTION_LOGITS = int(sum(FIRM_ACTION_DIMS))
NUM_FIRM_ACTION_COMBINATIONS = int(np.prod(FIRM_ACTION_DIMS))

# --- Government action space ---
GOV_TAX_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
GOV_WELFARE_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
GOV_POLICING_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
GOV_HEALTHCARE_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
GOV_EDUCATION_DELTAS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
GOV_MINWAGE_DELTAS = np.array([-0.01, 0.0, 0.01])
GOV_ACTION_DIMS = (
    len(GOV_TAX_DELTAS),
    len(GOV_WELFARE_DELTAS),
    len(GOV_POLICING_DELTAS),
    len(GOV_HEALTHCARE_DELTAS),
    len(GOV_EDUCATION_DELTAS),
    len(GOV_MINWAGE_DELTAS),
)
NUM_GOV_ACTION_LOGITS = int(sum(GOV_ACTION_DIMS))
NUM_GOV_ACTION_COMBINATIONS = int(np.prod(GOV_ACTION_DIMS))

# --- Central Bank action space ---
CB_RATE_DELTAS = np.array([-0.005, -0.0025, 0.0, 0.0025, 0.005])
NUM_CB_ACTIONS = len(CB_RATE_DELTAS)

# --- Curriculum schedule defaults (Addendum Section 16.2) ---
CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION = 1
CURRICULUM_STAGE_MICRO_MARKET = 2
CURRICULUM_STAGE_MACRO_POLICY = 3
CURRICULUM_STAGE_JOINT_FINE_TUNING = 4
CURRICULUM_STAGE_IDS = (
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION,
    CURRICULUM_STAGE_MICRO_MARKET,
    CURRICULUM_STAGE_MACRO_POLICY,
    CURRICULUM_STAGE_JOINT_FINE_TUNING,
)
CURRICULUM_STAGE_NAMES = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: "Household Foundation",
    CURRICULUM_STAGE_MICRO_MARKET: "Micro Market Adaptation",
    CURRICULUM_STAGE_MACRO_POLICY: "Macro Policy Introduction",
    CURRICULUM_STAGE_JOINT_FINE_TUNING: "Joint Fine-Tuning",
}
CURRICULUM_STAGE_DEFAULT_EPISODE_LENGTHS = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: 78,
    CURRICULUM_STAGE_MICRO_MARKET: 104,
    CURRICULUM_STAGE_MACRO_POLICY: 156,
    CURRICULUM_STAGE_JOINT_FINE_TUNING: 156,
}
CURRICULUM_STAGE_EPISODE_RANGES = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: (0, 500),
    CURRICULUM_STAGE_MICRO_MARKET: (500, 1500),
    CURRICULUM_STAGE_MACRO_POLICY: (1500, 3000),
    CURRICULUM_STAGE_JOINT_FINE_TUNING: (3000, 5000),
}
CURRICULUM_STAGE_POLICIES_TO_TRAIN = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: ["pi_household"],
    CURRICULUM_STAGE_MICRO_MARKET: ["pi_household", "pi_firm"],
    CURRICULUM_STAGE_MACRO_POLICY: ["pi_household", "pi_firm", "pi_gov", "pi_cb"],
    CURRICULUM_STAGE_JOINT_FINE_TUNING: ["pi_household", "pi_firm", "pi_gov", "pi_cb"],
}
CURRICULUM_STAGE_POLICY_LR_SCALES = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: {
        "pi_household": 1.0,
    },
    CURRICULUM_STAGE_MICRO_MARKET: {
        "pi_household": 0.35,
        "pi_firm": 1.0,
    },
    CURRICULUM_STAGE_MACRO_POLICY: {
        "pi_household": 0.20,
        "pi_firm": 0.30,
        "pi_gov": 1.0,
        "pi_cb": 1.0,
    },
    CURRICULUM_STAGE_JOINT_FINE_TUNING: {
        "pi_household": 0.15,
        "pi_firm": 0.20,
        "pi_gov": 0.30,
        "pi_cb": 0.30,
    },
}
CURRICULUM_STAGE_CONFIG = {
    CURRICULUM_STAGE_HOUSEHOLD_FOUNDATION: {
        "firms_fixed": True,
        "gov_scripted": True,
        "cb_scripted": True,
        "households_learning": True,
    },
    CURRICULUM_STAGE_MICRO_MARKET: {
        "firms_fixed": False,
        "gov_scripted": True,
        "cb_scripted": True,
        "households_learning": True,
    },
    CURRICULUM_STAGE_MACRO_POLICY: {
        "firms_fixed": False,
        "gov_scripted": False,
        "cb_scripted": False,
        "households_learning": True,
    },
    CURRICULUM_STAGE_JOINT_FINE_TUNING: {
        "firms_fixed": False,
        "gov_scripted": False,
        "cb_scripted": False,
        "households_learning": True,
        "lr_scale": 0.4,
        "smoothness_scale": 1.5,
    },
}

# --- Curriculum phases (Addendum Section 16.2) ---
CURRICULUM_PHASES = {
    stage_id: {
        "name": CURRICULUM_STAGE_NAMES[stage_id],
        "episode_length": CURRICULUM_STAGE_DEFAULT_EPISODE_LENGTHS[stage_id],
        "episodes": CURRICULUM_STAGE_EPISODE_RANGES[stage_id],
        "policies_to_train": list(CURRICULUM_STAGE_POLICIES_TO_TRAIN[stage_id]),
        "policy_lr_scales": dict(CURRICULUM_STAGE_POLICY_LR_SCALES[stage_id]),
        **CURRICULUM_STAGE_CONFIG[stage_id],
    }
    for stage_id in CURRICULUM_STAGE_IDS
}

CURRICULUM = {
    k: {"episode_weeks": v["episode_length"], **v}
    for k, v in CURRICULUM_PHASES.items()
}

# --- PPO hyperparameters (Section 12.2) ---
PPO_NETWORK_HIDDENS = [256, 256]
PPO_LR = 3e-4
PPO_CLIP = 0.2
PPO_ENTROPY_COEFF = 0.01
PPO_NUM_SGD_ITER = 10
PPO_GAE_LAMBDA = 0.95
PPO_VF_LOSS_COEFF = 0.5
PPO_GRAD_CLIP = 0.5
PPO_MINIBATCH_SIZE = 1024
PPO_TRAIN_BATCH_SIZE = 8192

PPO_DEFAULTS = {
    "lr": PPO_LR,
    "gamma": HOUSEHOLD_GAMMA,
    "clip_param": PPO_CLIP,
    "entropy_coeff": PPO_ENTROPY_COEFF,
    "vf_loss_coeff": PPO_VF_LOSS_COEFF,
    "num_sgd_iter": PPO_NUM_SGD_ITER,
    "sgd_minibatch_size": PPO_MINIBATCH_SIZE,
    "train_batch_size": PPO_TRAIN_BATCH_SIZE,
    "model": {"fcnet_hiddens": PPO_NETWORK_HIDDENS},
}

NUM_ENV_WORKERS = 8

# --- Unity/export visualization defaults ---
UNITY_EXPORT_FILENAME = "unity_playback.json"
UNITY_SCENE_SCALE = 1.0
UNITY_HOUSEHOLD_GRID_COLUMNS = 10
UNITY_HOUSEHOLD_GRID_SPACING = 2.25
UNITY_HOUSEHOLD_DISTRICT_Z = -14.0
UNITY_FIRM_RING_RADIUS = 10.0
UNITY_FIRM_RING_STEP = 2.5
UNITY_FIRM_RING_LAYERS = 3
UNITY_GOV_POSITION = (-19.0, 0.0, 22.0)
UNITY_CB_POSITION = (19.0, 0.0, 22.0)
UNITY_MARKET_POSITION = (0.0, 0.0, 0.0)
UNITY_CLINIC_POSITION = (-14.0, 0.0, 5.0)
UNITY_BANK_POSITION = (14.0, 0.0, 5.0)
UNITY_CRIME_POSITION = (14.0, 0.0, -8.0)
UNITY_REST_POSITION = (-14.0, 0.0, -8.0)
UNITY_REPLAY_SECONDS_PER_WEEK = 0.35

# --- Metrics defaults ---
METRICS_EMA_ALPHA = 0.1
METRICS_TOP_SHARE_FRACTION = 0.01
METRICS_BOTTOM_SHARE_FRACTION = 0.50
MACRO_SUMMARY_WINDOW_WEEKS = GOV_UPDATE_PERIOD

# --- Plot styling ---
PLOT_TIME_SERIES_FIGSIZE = (10, 4)
PLOT_HISTOGRAM_FIGSIZE = (8, 5)
PLOT_LORENZ_FIGSIZE = (6, 6)
PLOT_SCATTER_FIGSIZE = (6, 5)
PLOT_DPI = 150
PLOT_GRID_ALPHA = 0.3
PLOT_HISTOGRAM_BINS = 30
PLOT_HISTOGRAM_ALPHA = 0.7
PLOT_REFERENCE_LINE_ALPHA = 0.5
PLOT_FILL_ALPHA = 0.1
PLOT_SCATTER_ALPHA = 0.3
PLOT_SCATTER_SIZE = 10

# --- Viewer styling/layout ---
VIEWER_SPEED_SLOW = 0.5
VIEWER_SPEED_DEFAULT = 1.0
VIEWER_SPEED_FAST = 2.0
VIEWER_BASE_FPS = 20
VIEWER_WINDOW_WIDTH = 1200
VIEWER_WINDOW_HEIGHT = 800
VIEWER_FONT_SMALL_SIZE = 12
VIEWER_FONT_MED_SIZE = 16
VIEWER_PANEL_MARGIN = 5
VIEWER_HOUSEHOLD_AREA_RIGHT_MARGIN = 120
VIEWER_HOUSEHOLD_AREA_BOTTOM_MARGIN = 70
VIEWER_HOUSEHOLD_COLS = 10
VIEWER_HOUSEHOLD_DOT_RADIUS = 4
VIEWER_HEALTH_RING_RADIUS = 6
VIEWER_HEALTH_RING_THRESHOLD = 0.3
VIEWER_FIRM_X_OFFSET = 100
VIEWER_FIRM_Y_START = 30
VIEWER_FIRM_Y_SPACING = 30
VIEWER_FIRM_SIZE_SCALE = 2
VIEWER_FIRM_SIZE_MIN = 4
VIEWER_FIRM_SIZE_MAX = 15
VIEWER_FIRM_COLOR_SCALE = 400
VIEWER_OVERLAY_OFFSET_Y = 45
VIEWER_STATUS_OFFSET_Y = 25
VIEWER_COLORS = {
    "poor": (220, 50, 50),
    "middle": (220, 180, 50),
    "rich": (50, 180, 50),
    "elite": (50, 100, 220),
    "bg": (30, 30, 40),
    "panel_bg": (45, 45, 55),
    "text": (220, 220, 220),
    "grid": (70, 70, 80),
    "firm_outline": (200, 200, 200),
    "health_ring": (255, 100, 100),
    "fallback_regime": (200, 200, 200),
    "fallback_household": (150, 150, 150),
}
VIEWER_REGIME_COLORS = {
    REGIME_FREE_MARKET: (100, 200, 100),
    REGIME_RESTRICTED: (200, 200, 100),
    REGIME_SOCIAL_DEM: (100, 150, 250),
    REGIME_COMMAND: (250, 100, 100),
}
