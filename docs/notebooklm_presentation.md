# MARL Economy: A Multi-Agent Reinforcement Learning Simulation of National Economies

*A presentation script for NotebookLM. Designed to be read aloud or summarized into an audio overview. Tone: academic but accessible - the level of a graduate seminar that does not assume prior reinforcement-learning expertise but does assume the listener is intellectually serious. Estimated runtime when read at a moderate pace: forty to forty-five minutes.*

---

## 1. What This Project Is, In One Paragraph

The `marl_economy` project is a self-contained multi-agent reinforcement learning simulator of a small national economy. It does not try to replicate any single real country. Instead, it implements a synthetic but internally consistent economy populated by four distinct kinds of decision-making agent - households, firms, a government, and a central bank - and trains each of them, simultaneously and from scratch, to pursue its own objective inside the same shared world. Layered on top of those four learning agents is a stack of human-condition mechanics - hunger, fatigue, illness, and place-based disease transmission - together with a generator of exogenous disasters: earthquakes, fires, floods, supply shocks, and epidemic waves. The simulator is written in Python on top of `numpy`, `gymnasium`, and Ray RLlib. It produces logged trajectories, plots, and exported snapshots that can be replayed in Unity for three-dimensional visualization. The intellectual ambition of the project is to ask, in concrete computational form, the question: *what happens when economic actors with conflicting time horizons and conflicting incentives are forced to learn how to live together inside a world that can also hurt them?*

## 2. Why This Question Is Interesting

Traditional economic models tend to make one of two simplifications. They either assume that agents are perfectly rational optimizers who can solve fully specified expected-utility problems, or they specify behavior in advance through rule-based heuristics. Both approaches sacrifice something important. The first sacrifices realism, because nobody actually solves a Bellman equation when deciding whether to go to work or rest. The second sacrifices generality, because the rules have to be hand-tuned for every new scenario.

Multi-agent reinforcement learning offers a third path. It lets each agent class learn its policy through trial-and-error interaction with the others, with no closed-form solution required. The result is a kind of computational political economy: instead of asserting how a household should behave, the simulator asks what behaviors emerge when many households, firms, a government, and a central bank are all simultaneously learning to optimize their own different objectives. The agents' policies coevolve. Equilibria, if they exist, are discovered rather than assumed.

This particular implementation is small enough to read end-to-end in a few hours, but rich enough to exhibit the phenomena that any macro-scale toy model needs to be able to display: business cycles, inflation, unemployment, inequality, crime, firm entry and exit, monetary policy, infectious-disease outbreaks, natural disasters, and full economic collapse. It does not just simulate decision-making. It simulates the conditions under which decision-making takes place.

## 3. The Four Agent Classes

Every agent in this simulator belongs to one of four classes. Each class operates on a different timescale, has its own observation vector, its own action space, and its own reward function.

**Households** act every simulated week. They are the workers, consumers, patients, and occasional criminals in the system. A household has six discrete actions available to it: work at a firm, shop for goods, attempt to steal from a firm, rest, invest, or visit a clinic. Its observation vector is thirteen-dimensional, encoding its personal state - wealth, health, hunger, fatigue, current illness state, age, last action - together with the macro context it can perceive: current tax rate, mean market price, mean market wage, healthcare access, education quality, and policing intensity. The default population is eighty households.

**Firms** act every four simulated weeks, which the project treats as one month. A firm chooses three things at every action step: a discrete delta to its price, a discrete delta to its wage offer, and a discrete delta to its production capacity. The price and wage deltas can move by two percent up or down in one-percent increments, and the capacity delta is a three-way choice: shrink, hold, or grow. The firm population is dynamic. A firm exits after eight consecutive weeks of negative cash; that exit is processed in weakest-first order, and is blocked only when alive-firm count is at the floor of two. New firms enter when the recent eight-week unmet-demand rate exceeds thirty percent, but entry is probabilistic - even when conditions warrant it, the actual spawn happens with fifty percent probability per check. When a mass exit pushes firms below an emergency buffer of two above the floor, the simulator force-spawns replacements until the buffer is restored. The hard maximum firm count is twelve. The default starting firm count is six. The firm observation vector is ten-dimensional.

**The government** acts every twelve simulated weeks, treated as a quarter. It controls six policy levers: tax rate, welfare generosity, policing intensity, healthcare access, education quality, and the minimum-wage floor. Each lever moves by a small discrete delta on a quarterly cadence - five-way for the first five levers (plus or minus two percent in one-percent steps), and a narrower three-way move for the minimum-wage floor. The government's nine-dimensional observation includes quarterly aggregates of GDP per capita, Gini, health, poverty, crime, unemployment, and inflation, the current tax rate, and a count of currently-active disasters, which lets policy react to shocks rather than only to steady-state metrics. Its reward integrates the same macro outcomes with a smoothness penalty that punishes erratic policy.

**The central bank** also acts quarterly. Its single decision is a discrete delta to the policy interest rate, drawn from the set {−0.5, −0.25, 0, +0.25, +0.5} percentage points - twenty-five to fifty basis points per move, exactly the kind of increment a real central bank deliberates over. Its four-dimensional observation includes quarterly inflation, GDP growth, unemployment, and the current interest rate. Its reward follows a stylized loss function: it penalizes squared deviations of inflation from a two-percent target, rewards GDP growth, and penalizes both unemployment and the squared change in the rate itself, which discourages whiplash monetary policy.

The reason these timescales differ matters more than it might first appear. Households experience the world in weeks because consumption and labor decisions really do operate at that frequency. Firms reset their pricing and wage decisions monthly, which approximates a payroll cycle. Government and central bank act quarterly because that is roughly the cadence of real fiscal and monetary deliberation. Aligning the agents' learning frequencies with their natural decision cadences is part of what makes the simulator feel like a small economy rather than an abstract game.

## 4. The Shared World

All four agent classes inhabit a single object - a Python class called `World`. This object owns the simulator's state and exposes one main method, `step`, which advances time by exactly one simulated week. Every action by every agent must pass through this method, and every state update emerges from it. The orchestration is deliberately concentrated in one place so that the order of operations is unambiguous.

Inside `step`, the weekly tick proceeds in a fixed sequence of fifteen distinct phases. The order matters, because each phase consumes state that earlier phases produced.

The first phase is the **disaster roll**. Before any agent acts, the simulator consults the exogenous shock generator. Five independent rolls - earthquake, fire, flood, supply shock, epidemic wave - are made, and any that fire mutate firm capacity, household health, household displacement, and ongoing disaster counters. Putting disasters first matters: it means a household choosing to work cannot avoid being injured by an earthquake; the disaster sets up the conditions under which the household decides.

The second phase **applies slow-agent actions**, but only on the weeks those agents are scheduled to act. Firm price, wage, and capacity deltas are applied on monthly weeks. Government policy deltas are applied on quarterly weeks. Central-bank rate deltas are applied on their quarterly weeks.

The third phase processes **firm entry and exit**. Insolvent firms shut down. New firms spawn under favorable conditions, subject to the hard minimum and maximum bounds. Mean market price and mean market wage are then recomputed so that every downstream agent reads a consistent macro reference point.

The fourth phase is **routing**. Each household action that requires a firm - work, shop, or clinic - is matched to a specific firm through the softmax routing rules, gated by the firm solvency predicates. Refused hires are recorded as typed events.

The fifth phase is the **vectorized household step** - the hot path. All households across all parallel worlds advance their state in a single batched numpy operation. Wealth changes, health changes, hunger and fatigue updates, experience accumulation, and base reward computation all happen here.

The sixth phase combines **clinic capacity allocation, place-based epidemic transmission, and illness state transitions**. The simulator computes how many clinic slots are available this week, allocates them to the visiting households with severe cases prioritized, and records who was treated and who was denied. Then it determines where each household physically was - at a workplace, market, clinic, home, or in public - and computes per-place infection probabilities. Newly-exposed households enter the EXPOSED state. The illness state machine then advances every household, with treated severe cases moving toward recovery and untreated severe cases facing higher mortality.

The seventh phase **records per-household events and updates the typed ledger**: hunger crises, fatigue crises, infection onsets, severe illness onsets, recoveries, and deaths. The displacement penalty is applied to households that were displaced by floods or firm exits.

The eighth phase applies **firm-side sale effects**. For each shop or clinic match, the firm's cash is credited by the actual amount paid - fifty-seven percent of the posted price for shopping, one hundred and fifty percent for clinic visits.

The ninth phase resolves **theft**. Successful thieves transfer cash from a randomly-targeted alive firm, capped at the victim firm's cash. The household reward is recomputed after the transfer so that the reward reflects the actual post-conservation wealth change.

The tenth phase runs **firm production and overhead**: each alive firm produces output bounded by capacity, pays wages for hired workers, and pays its weekly overhead. Firms in disaster downtime pay overhead but produce nothing.

The eleventh phase, on firm-update weeks only, **updates firm capacity smoothing and computes monthly firm rewards**.

The twelfth phase updates **macro metrics**: GDP, inflation, Gini, poverty, unemployment, crime, average health, plus the human-condition aggregates over hunger, fatigue, and illness, plus firm solvency diagnostics, plus the per-week event counts from the ledger.

The thirteenth phase computes **government and central-bank rewards**, on their respective quarterly weeks.

The fourteenth phase advances the week counter and **checks termination**. Five distinct paths can end the episode: reaching the configured episode length, mass death (more than sixty percent of households below the death threshold), the multi-signal collapse detector firing, hyperinflation persisting for four consecutive weeks, or any numerical NaN appearing in the latest metrics dict. Each path produces a distinct reason string. After the termination check, the simulator runs a set of cheap always-on invariant assertions: no negative wealth, all biological variables in the unit interval, all illness states in range, no NaN, finite firm cash, and firm count within bounds. These assertions cost a few tens of microseconds per step and catch vectorized-arithmetic bugs the instant they appear, which is one of the most important debuggability properties of the refactored code.

The fifteenth phase **assembles the return payload**: filtered observations for the agents scheduled to act next week, the rewards delivered this week, the per-world done flags, and the info dict containing this week's metrics and event counts.

This ordering is not accidental. It guarantees that within a single weekly tick, all agents see a consistent snapshot of the world, and that no agent's reward is contaminated by stale state from the previous week. It also guarantees that disasters constrain choices rather than retroactively invalidating them.

## 5. The Four Economic Regimes

A key design choice is that the simulator does not commit to a single political-economic configuration. Instead, it ships four canonical regimes, each calibrated against published economic data for representative countries, sourced from Table IV of the FatBar comparative-political-economy paper using underlying World Bank, OECD, WHO, and UNESCO figures.

The **Liberal Market Economy**, abbreviated LME, corresponds to the United States, Brazil, and Singapore in the comparative-political-economy literature. Taxes are low, welfare is modest, policing is moderate, and the minimum wage floor is permissive. The intellectual lineage here is Hall and Soskice's 2001 framework for varieties of capitalism.

The **Coordinated Market Economy**, or CME, corresponds to Germany and Japan. Taxes and welfare sit higher than in the LME, healthcare and education are more universal, and wage-setting institutions exert more influence on firm behavior.

The **Social Democratic Welfare Economy**, abbreviated SDWE, corresponds to Sweden and Norway. The state takes the largest share of GDP through taxation, redistributes aggressively through welfare and universal healthcare, and maintains very high education quality. The lineage here is Esping-Andersen's 1990 classification of welfare-state regimes.

The **Socialist Market Economy**, or SME, corresponds to China and Vietnam. Policing is intense (anchor 0.85), and taxes are calibrated to the real SME range (anchor 0.22, bounds 0.15-0.28), which is lower than the CME and SDWE regimes. The state retains broad authority to set policy bounds. The reference here is Bekkevold et al's 2020 analysis of contemporary socialist-market hybrids.

Each regime is encoded as two things: a set of *anchor values* that initialize the policy state at the start of an episode, and a set of *hard bounds* that constrain how far the government's learned actions can drift. A government policy in the LME regime literally cannot push tax rates as high as it could under the SDWE regime, because the regime bounds clamp the action before it takes effect. This means the comparison across regimes is not "which policy did the government discover," but rather "given these institutional bounds, what equilibrium emerges?"

The four regimes also expose declarative metadata fields for instruments like price caps and subsidies. These flags are currently advisory hooks rather than enforced mechanics - they document intended future policy modules and remain available for extension without disturbing the rest of the system. Older free-form regime names - FreeMarket, Restricted, SocialDem, Command - remain as aliases that resolve to the canonical names above, so older checkpoints, run directories, and Unity playback files keep working.

## 6. The Reward Functions: How Each Agent Defines Success

The reward functions are where the simulator's economic theory is most explicit. Each agent class optimizes a different objective.

**Households** optimize a log-utility difference. The reward at each weekly step is the change in the logarithm of wealth plus twice the change in the logarithm of health. The doubled weighting on health reflects the assumption that survival matters more than money - wealth without health is worthless because death terminates the agent. A one-time transition penalty of -2.0 is applied at the moment of death, followed by a bounded persistent dead-state reward of -0.45 per week while dead. Bankruptcy triggers a one-time penalty of -1.5. Episode-progress scaling was removed in the Phase-3 reward redesign. A small additional penalty is applied for displacement events triggered by firm exits or floods. A small per-week health bonus is also granted for the Rest action, and a small health cost is imposed for failed steal attempts. The intuition is direct: a household learns to keep itself alive, fed, healthy, and solvent, and to value those things in roughly the proportions a real household would.

**Firms** optimize a monthly reward that combines normalized profit, an unmet-demand penalty, a smoothness penalty on price and wage changes, and a bankruptcy penalty when cash goes negative. Profit is computed as revenue minus wage bill minus four weeks of overhead. The normalization divides by total sales - guarded against division by zero - so that small firms and large firms are scored on comparable scales. The unmet-demand coefficient is 0.35, the smoothness coefficient is 8.0, and the bankruptcy penalty ramps linearly from zero at three or fewer consecutive weeks of negative cash to a maximum of 1.5 at seven weeks, just before the exit threshold. The smoothness term, weighted heavily, prevents firms from learning oscillatory pricing policies that would be unrealistic and unstable. During the joint fine-tuning stage the smoothness coefficient is further scaled up by a factor of 1.5 to discourage oscillation in the equilibrium-search phase.

**The government** optimizes a weighted sum of macro outcomes. The reward is a times log of one plus GDP per capita, minus b times Gini, plus c times average health, minus d times crime rate, minus e times poverty rate, minus u times unemployment, minus zeta times the smoothness sum. The coefficients are explicit: a equals 1.2, b equals 0.9, c equals 0.8, d equals 0.6, e equals 0.9, u equals 0.5, and zeta equals 1.5. The smoothness term sums the squared deltas of all six policy levers - tax rate, welfare, policing, healthcare access, education quality, and minimum wage floor - so that erratic motion in any of them is penalized. These coefficients are tuned so that no single dimension can dominate, and so that the government cannot satisfy its objective simply by collapsing the economy into perfect equality of misery. A specific reward modification added during development includes an explicit GDP-collapse penalty with a continuous severity scale: the penalty equals 8.0 times severity, where severity is computed relative to a decaying trailing peak: severity = max(0, (0.5*peak - gdp_pc) / (0.5*peak)), clamped to [0,1]. A separate continuous health-floor penalty of 1.5 * max(0, 0.40 - avg_health) is also applied every government step. Together these forms close the "dead but equal" loophole that earlier reward formulations accidentally rewarded.

**The central bank** follows a stylized loss function that economists will recognize as a relative of the Taylor rule. It minimizes squared deviation of inflation from a two-percent target (coefficient 1.0), rewards GDP growth with weight lambda equal to 0.5, penalizes unemployment with weight nu equal to 0.3, and penalizes squared changes in the rate itself with weight rho equal to 2.5, which discourages whiplash monetary policy. This formulation captures the standard dual mandate - price stability and full employment - together with the institutional preference for gradualism in monetary policy.

The cross-agent design pressure deserves emphasis. The household reward, firm reward, government reward, and central-bank reward are not aligned. Households want low taxes and high welfare. Firms want low minimum wages and high consumer demand. The government wants equality and growth simultaneously. The central bank wants low inflation, which may conflict with government's growth objective. Learning takes place inside this conflict.

## 7. The Human Condition Stack: Hunger, Fatigue, and Illness

A reinforcement-learning agent will exploit any unmodeled corner of its environment. If the only thing keeping a household alive is wealth, the optimal policy is to accumulate cash and ignore everything else. To prevent this, the simulator layers a sequence of biological pressures on top of the basic wealth-health dynamics - each one calibrated so that ignoring it produces measurable harm.

**Hunger** rises every week by 0.06 and is reduced by 0.35 each time a household successfully shops. Hunger is bounded in the unit interval. It has three threshold regimes. Below 0.50 - the health-penalty threshold - hunger has no effect. Between 0.50 and 0.60 - the productivity threshold - hunger imposes an extra weekly health decay that scales linearly with how far past the threshold it has climbed. Above 0.60, hunger additionally scales work income down toward a productivity floor of 0.30. A household at maximum hunger keeps only thirty percent of its work income. Above the fatal threshold of 0.95, hunger produces a large health decay of 0.50 per week that quickly drives the household past the death threshold of 0.01. A separate crisis threshold at 0.85 triggers a typed event in the ledger. The intent is direct: a household that never shops cannot survive, even if it has infinite wealth. With a weekly accumulation of 0.06 and a shopping drop of 0.35, the household must shop at least once every roughly five to six weeks just to stay below the penalty zone.

**Fatigue** is the symmetric pressure. It rises by 0.12 every time a household works, falls by 0.40 every time it rests, and falls by 0.10 every time it visits a clinic. The calibration target is a familiar one: one rest week clears roughly three work weeks of accumulated fatigue, which approximates a five-on-two-off schedule. Like hunger, fatigue lives in the unit interval. Above the productivity threshold of 0.55, work income is dampened toward a floor of 0.40 - a fully exhausted household earns only forty percent of normal. Above the health-penalty threshold of 0.70, fatigue imposes additional weekly health decay. A household that works every week without resting will eventually exhaust itself into illness.

**Illness** is the most elaborate piece of the human-condition stack. It is a five-state machine: HEALTHY, EXPOSED, MILD, SEVERE, and RECOVERING. Transitions happen weekly with explicit probabilities. A HEALTHY household becomes EXPOSED through the epidemic layer, which the next section will describe in detail. An EXPOSED household either progresses to MILD with probability 0.55 or returns to HEALTHY asymptomatically with probability 0.30. A MILD household progresses to SEVERE with probability 0.08 or recovers to RECOVERING with probability 0.40, and the Rest action that week adds 0.15 to the recovery probability through an explicit rest-bonus term. A SEVERE household either recovers, dies, or stays severe - and the difference between treated and untreated outcomes is dramatic. Untreated, a severe household recovers with probability 0.10 and dies with probability 0.10. Treated at a clinic, a severe household recovers with probability 0.40 and dies with probability 0.02. Healthcare access changes a severe-illness mortality rate by a factor of five. A RECOVERING household returns to HEALTHY with probability 0.50 plus a rest bonus.

Each illness state has its own consequences. MILD illness imposes a small extra health decay of 0.003 per week and dampens work productivity to seventy percent. SEVERE illness imposes a larger decay of 0.025 per week and dampens productivity to twenty percent. RECOVERING dampens productivity slightly to eighty-five percent. Severe untreated illness can be lethal, and the simulator implements that lethality by directly pushing health below the death threshold for households unlucky enough to die from disease - a deliberate design choice that funnels all causes of death through the same termination machinery.

These three pressures - hunger, fatigue, and illness - also interact through an explicit per-household susceptibility multiplier in the illness module. Hunger past its penalty threshold doubles infection probability. Fatigue past its penalty threshold doubles it. Low health, below 0.30, triples it. These multipliers compose: a starving exhausted low-health household can be twelve times more likely to catch whatever disease is circulating than a healthy fed rested one. This is not just a numerical detail. It is the simulator's implementation of the well-documented epidemiological reality that poverty, malnutrition, and chronic exhaustion are infection risk factors. The macro consequence is that institutional choices - welfare generosity, healthcare access, minimum wage floors - affect disease dynamics through the bodies of individual households.

## 8. Place-Based Epidemic Transmission

A simple SIR model treats infection as a population-level mean-field process: a single transmission rate times the number of infectious people. This simulator does something more granular. It models *places*.

Every week, every household is at exactly one place, determined by its action. A household that chose to work is at the workplace of the firm it was routed to. A household that chose to shop is at the market of the firm it was routed to. A household that chose to visit a clinic is at the clinic of the firm it was routed to. A household that chose to rest is at home. A household that chose to invest or steal is in public.

The transmission rule is then per-place. Within a single place, with k infectious attendees, the base infection probability for any susceptible attendee is `P = 1 - (1 - α)^k`. The transmission coefficient α is place-specific. Home is the lowest at 0.005, public spaces are next at 0.008, then markets at 0.015, workplaces at 0.020, and - strikingly - clinics are the highest at 0.040. The clinic α is double the workplace value precisely because the sick concentrate at clinics. A household visiting a clinic is more likely to catch something there than at any other named place in the simulator, before any modifiers. Healthcare access enters as a multiplicative damp on α - a regime with full healthcare can halve transmission everywhere - and clinic overcrowding multiplies α at clinics by 1.5 when demand exceeds capacity, which models the well-known phenomenon that overwhelmed hospitals become disease vectors. The per-household susceptibility multiplier from the previous section then scales this place-base probability before the actual infection roll.

A small base background infection rate also applies, covering contacts not modeled by named places. The base rate is healthcare-damped and boosted during active epidemic waves.

The implications are non-trivial. A government that underfunds healthcare not only reduces direct clinic treatment, it raises α at every place where infection can occur. A government that overconcentrates demand at clinics through aggressive public-health messaging risks pushing clinic capacity past the overcrowding threshold, which makes clinics themselves dangerous. A regime that produces high unemployment may inadvertently reduce workplace transmission, but it may also reduce wealth-health buffers and raise individual susceptibility. The epidemic layer makes these tradeoffs measurable.

## 9. Exogenous Disasters

The simulator includes five disaster types, each rolled for independently every week with a small per-week probability. The probabilities are calibrated together so that a typical 156-week episode experiences roughly one or two events of mixed type - enough to test resilience without dominating the simulation.

**Earthquakes** strike with probability 0.08 percent per week. When they fire, they affect all alive firms simultaneously. Each affected firm loses thirty percent of its capacity, half of its inventory, and a fixed cash amount. Each household rolls independently for injury with probability twenty percent - if injured, it loses thirty percent of its current health. An earthquake is a sudden negative supply shock that also damages the workforce.

**Fires** strike with probability 0.25 percent per week. They affect a random subset of one or two firms, drawn uniformly. Affected firms enter a four-week downtime period during which they cannot hire or produce. The downtime is recorded per firm and decays each week.

**Floods** strike with probability 0.15 percent per week. They affect two or three firms with a six-week downtime - longer and broader than fires. Each household additionally rolls independently for displacement with probability ten percent. Displaced households are temporarily relocated, which interacts with the routing layer and incurs a small reward penalty.

**Supply shocks** strike with probability 0.35 percent per week - the most common disaster. They are not localized. They impose a thirty-percent capacity drag across all firms for eight weeks, which dampens production economy-wide. The configuration also defines a price multiplier of 1.40 intended to model the corresponding cost-push inflation, but the current build does not wire that multiplier into the firm pricing path; only the capacity drag is active. Supply shocks therefore manifest as a temporary aggregate-supply contraction, which the inflation metric picks up indirectly through reduced output and changing market prices.

**Epidemic waves** strike with probability 0.20 percent per week. They seed five percent of the healthy population into the EXPOSED state immediately and apply a four-times boost to the transmission coefficient for twelve weeks. They model pandemics that arrive from outside the simulated economy.

All five disasters share an important property: they are exogenous. No agent can prevent them, and no agent's policy choice causes them. Their function is to keep the trained policies honest. A government policy that looks optimal under steady-state conditions may be brittle to the introduction of a fire or a flood; a central bank's gradualism may be tested by a supply shock. The disasters are stress tests that the trained system cannot anticipate by learning, only by maintaining buffers.

Every disaster occurrence is recorded in the event ledger with its week, type, and effect counts, so post-hoc analysis can quantify exactly how much of an episode's macroeconomic trajectory is attributable to exogenous shocks versus learned policy choices.

## 10. Clinic Capacity and Medical Triage

Healthcare is not infinite. The simulator models clinic capacity explicitly. Each week, the maximum number of households that can be treated is a base value of thirty slots plus a bonus of up to sixty slots that scales with healthcare access. A regime with full healthcare access provides ninety clinic slots per week. A Liberal Market regime starting from a healthcare-access anchor of 0.55 provides about sixty-three clinic slots per week. With a population of eighty households, this means a Social Democratic regime can hospitalize the entire population in a single week, while a Liberal Market regime cannot.

When the number of households who choose to visit a clinic in a given week is at or below capacity, every visitor is treated. When demand exceeds capacity, the simulator applies a simple medical triage: SEVERE illness gets a priority score of 3.0; MILD, RECOVERING, and otherwise non-severe states get a priority score of 1.0; HEALTHY visitors get 0.0. The allocator sorts by score with a small random jitter to break ties and assigns the available slots in priority order. In practice, this means severe cases are reliably treated first, but among non-severe visitors the allocation is effectively random. Untreated visitors still pay the firm - they were physically at the clinic - but they are routed through the no-treatment branch of the illness state machine, which uses lower recovery probabilities and higher mortality.

Clinic overcrowding has a second consequence already mentioned: it boosts the transmission coefficient at clinics, so when clinics are at capacity they also become more dangerous as infection venues. The combination produces a realistic hospital-system collapse dynamic in which an epidemic wave can saturate clinic capacity, which then both reduces treatment quality for the sickest and increases new infections among the worried-well, which then pushes the next week's demand even higher.

The triage and overcrowding mechanics together close a feedback loop that no static rule-based simulator could capture in a single equation. The same policy parameter - healthcare access - controls clinic capacity, transmission damping, and treatment probability simultaneously, and the consequences of changing it depend on what the rest of the system is doing.

## 11. The Firm Solvency Gate

A subtle but consequential mechanic in the simulator is the multi-predicate hire gate. When a household chooses to work, the routing layer must select a firm. Not every firm is eligible. The simulator enforces five predicates simultaneously.

First, the firm must be alive. Exited firms are filtered out.

Second, the firm must not be in disaster downtime. A firm that just suffered a fire or flood cannot hire while its downtime counter is positive.

Third, the firm must be solvent. Its cash must be above an insolvency threshold. A firm whose cash has dropped past that floor cannot take on additional wage obligations.

Fourth, the firm must be able to afford one more worker. The total expected wage bill - current hires plus one, multiplied by the wage offer - must fit within cash plus a bounded credit limit. The credit limit is a configurable cushion that allows brief solvency dips without immediate cessation of hiring, modeling the fact that real firms can borrow short-term.

Fifth, the firm must have at least minimum capacity. A firm at the capacity floor has no room for an additional worker even if it could afford one.

This gate matters historically. An earlier version of the simulator allowed bankrupt firms to continue hiring, and households learned to keep working at those firms indefinitely, producing artificial GDP, accumulating private wealth, and hiding the underlying economic collapse. The plots looked fine; the economy was dead. The fix was to make the hire decision a function of firm solvency, and the consequence is that work attempts can now fail - which surfaces as a real unemployment signal in the macro metrics, exactly as it should.

When a hire is refused, the routing layer can attempt to reroute to another eligible firm. When no eligible firm exists, the work attempt fails and a typed event is recorded in the ledger so that post-hoc analysis can count exactly how many household-weeks were spent unable to find a willing employer.

## 12. The Routing System

Households cannot simply "work" or "shop" in the abstract. They have to be matched to a specific firm. The routing module handles this matching, and the matching rules are economically meaningful.

For **work routing**, the simulator considers every firm that passes the multi-predicate solvency gate described above and offers a wage above the minimum-wage floor. It then samples a firm with probability proportional to the exponential of the wage offer multiplied by a sensitivity coefficient. This is a softmax matching rule, which produces realistic-looking labor-market behavior: workers are biased toward higher-paying firms but do not always pick the single best one, because the world has frictions and information limits.

For **shop routing**, the simulator considers every firm that is alive and has positive stock, and samples a firm with probability proportional to the exponential of negative price times a sensitivity coefficient. This is the dual of the work rule: consumers prefer cheaper firms but do not always pick the absolute cheapest one.

For **clinic routing**, the simulator reuses the shop logic with a different cost multiplier - a clinic visit costs 1.5 times the firm's posted price, which represents the elevated cost of medical services relative to ordinary consumption. By contrast, shopping itself only charges fifty-seven percent of the posted price, calibrated to a real-world labor-share target. This calibration matters historically: an earlier version charged full price for food and trained policies stopped shopping entirely, which is why hunger emerged as a forced-demand mechanic in the first place.

For **steal targeting**, the simulator samples a firm uniformly at random from alive firms with positive cash, but only after the theft attempt itself succeeds in the firm-side phase. Theft is cash-conservative: a successful thief gains exactly what the victim firm loses, no money is created or destroyed, and the household reward is recomputed after the transfer.

These routing rules are simple, but they matter for emergent behavior. The softmax temperature on price sensitivity, for example, controls how aggressively consumers punish firms for raising prices, which in turn shapes the firm's incentive to keep prices low. The solvency gate on work routing, by removing bankrupt firms from the eligible set, prevents the bankruptcy-fueled fake-GDP exploit that earlier versions of the simulator suffered from.

## 13. The Training Pipeline: A Four-Stage Curriculum

The simulator is not trained all at once. It uses a four-stage curriculum that progressively introduces complexity, which is essential for getting the system to converge at all in a multi-agent setting.

**Stage One** is the household foundation, running from episode zero through episode five hundred. Households are the only learning agents, at full learning rate. Firms, government, and central bank act according to fixed scripted policies that produce reasonable but suboptimal behavior. The household population learns to balance work, rest, shopping, and clinic visits inside this scripted backdrop. Episodes are seventy-eight weeks long - eighteen simulated months. The goal is for households to discover the basic survival policy: work to earn wealth, shop to eat, visit the clinic when sick, rest when exhausted.

**Stage Two** introduces firm learning, running from episode five hundred through fifteen hundred. Households continue to train but at thirty-five percent of their original learning rate. Firms now optimize their own pricing, wage, and capacity decisions through PPO at full learning rate. Episode length increases to one hundred and four weeks. The two populations coevolve. Firms learn that aggressive price hikes lose customers; households learn to reroute to the surviving firms.

**Stage Three** adds the macro policy layer, running from episode fifteen hundred through three thousand. The government and the central bank become learning agents at full learning rate. Households continue at twenty percent of their original rate; firms at thirty percent. None of the earlier populations are frozen - they continue to learn - but their learning rates are dialed down so that the macro policy can stabilize against a roughly known private sector. Episode length grows to one hundred and fifty-six weeks, which is three years of simulated time.

**Stage Four** is joint fine-tuning, running from episode three thousand through five thousand. All four agent classes train simultaneously at low learning rates - households at fifteen percent, firms at twenty percent, government and central bank at thirty percent - with the firm smoothness coefficient scaled up by 1.5 to discourage policy oscillations. This is where the simulator searches for a stable equilibrium in which all four policies are locally optimal given each other's behavior. Episode length stays at three years.

The curriculum is not just pedagogical. It is computationally necessary. Multi-agent learning is notoriously unstable: every agent's environment is non-stationary because the other agents are simultaneously learning. Without staging, all four populations would chase a moving target and never converge. The curriculum gives each new agent class a relatively stable scaffold to learn against, and then loosens that scaffold once the new class has discovered a baseline policy.

## 14. The Underlying Equations

The economic dynamics inside the simulator are governed by a compact set of equations. Listing them in full is unnecessary, but the structure is worth understanding.

**Health decay** for each household at each week is a function of healthcare access - better access reduces decay - augmented by an extra decay term from hunger past the penalty threshold, another from fatigue past its penalty threshold, and another from illness in the MILD or SEVERE states.

**Wealth drag** is a piecewise function that penalizes accumulation at high wealth levels, which prevents indefinite wealth concentration and approximates the cost of maintaining a stock of assets.

**Work income** combines the firm's wage offer with an experience multiplier capped to prevent runaway compounding, a motivation factor that diminishes when wealth is already high, multiplicative noise, a tax adjustment, and now three productivity multipliers: one from hunger, one from fatigue, and one from illness state. A household that is hungry, exhausted, and severely ill earns a fraction of what a healthy well-fed worker earns at the same firm.

**Investment returns** are influenced by education quality and dampened by interest rates. There is also a non-trivial crash probability: investing is risky, and the risk increases with interest rates, modeling the way that monetary tightening can puncture asset bubbles.

**Crime success probability** is a sigmoid function of policing intensity, attenuated by welfare. Specifically, the success probability equals the sigmoid of six times the difference between half and the policing level, multiplied by one minus thirty percent of the welfare level. A successful theft transfers a uniformly-drawn amount between 0.02 and 0.08 from the victim firm to the thief. A failed steal attempt costs the household a small fixed health penalty of 0.01. The intuition is that strong policing makes crime less likely to succeed, but generous welfare reduces the temptation to attempt it in the first place - a soft version of the deterrence-versus-opportunity-cost debate that runs through real criminological literature.

**Firm production** is a square-root function of hires capped by capacity: weekly output equals the minimum of productivity times the square root of hires and the firm's current capacity. The capacity cap matters - it makes investment in capacity meaningful, because a firm cannot exceed its capacity even if it hires more workers.

**Inflation, GDP, Gini, poverty, unemployment, crime rate, and average health** are computed at the macro level by aggregating across the household and firm populations. The Gini coefficient uses the standard formula for sorted wealth vectors. GDP is a revenue proxy, summed across all firms in slot order using compensated summation to preserve bit-exact parity with the legacy implementation. Inflation and GDP growth are additionally smoothed by exponential moving averages with smoothing coefficient 0.1, which dampens single-week shocks when the macro agents consume the metric. A subtle but important distinction: the simulator's reported crime rate is the rate of *successful* thefts per household, not the rate of attempts. A separate crime-attempts rate is also tracked. The government's reward penalizes successful crime, not attempts, which means policing reduces the punishment by lowering the success probability rather than by deterring the action. New aggregates over the human-condition stack - mean hunger, fraction of population in hunger crisis, mean fatigue, fraction exhausted, fractions of population in each illness state - are also computed each week and exposed both in the per-step metrics CSV and in the episode summary JSON.

These equations do not claim to replicate real economic theory. They are stylized. What they are is *consistent*. Each equation respects the others - no money is created or destroyed, no production exceeds capacity, no firm hires beyond its solvency, and no household consumes what it cannot afford. The simulator was extensively debugged to enforce these conservation properties, because earlier versions had subtle leaks that produced artificially favorable outcomes.

## 15. Multi-Signal Collapse Detection

A non-trivial fraction of the engineering effort in the project went into deciding when an episode should end. The naive rule - "end after a fixed number of weeks" - is insufficient, because it can mask collapse. The simulator can keep ticking forward in time even after the economy has effectively died.

The current termination logic checks for five distinct conditions on every weekly tick. The episode ends when the configured length - typically one hundred and fifty-six weeks - is reached. It ends if more than sixty percent of the household population reaches near-zero health, which the simulator calls **mass death**. It ends if the multi-signal collapse detector fires, with one of three sub-reasons described below. It ends after four consecutive weeks of inflation above fifty percent, which captures **hyperinflation**. And it ends if any metric becomes numerically invalid, recorded with a specific reason like `nan_detected:gdp` so the failing field is known. Each termination reason is a distinct string, which makes it trivial to filter and group episodes by failure mode during analysis.

A more subtle termination condition is **hidden insolvency collapse**. The simulator enforces a minimum firm count to prevent total exit, because zero firms would crash the routing logic. But this minimum can hide collapse: in earlier versions, GDP would go to zero while two zombie firms remained alive, deeply insolvent, and the simulator would carry on for hundreds of additional weeks pretending nothing had gone wrong. The current build has a dedicated collapse detector that watches three signals together: a weekly-zero-GDP streak, the cash positions of the firms that remain, and the recent count of firm-exit-blocked events.

The specific thresholds are tuned together. Zero-GDP weeks must accumulate for at least twelve weeks - a full quarter - before the detector even considers firing. At that point, the detector consults the firm cash positions: if every remaining firm has negative cash and the mean cash position is below minus ten, it fires with reason `hidden_insolvency`. If the recent twelve-week window contains at least eight firm-exit-blocked events, indicating that the minimum-firm floor is preventing a cascade of exits, it fires with reason `blocked_exit_cascade`. If the firm population somehow drops to zero outright, it fires with reason `no_firms`. Each reason is recorded in the episode summary, alongside the week at which it fired and the final value of the zero-GDP streak. The macro agents are penalized in proportion to how much episode time remained, scaled by minus ten, which gives them a strong gradient to avoid pushing the world into a collapse state in the first place.

This kind of cause-tagged termination is important for analysis. It is not enough to know that an episode ended early. The researcher needs to know *why*, because the policy lessons are different depending on the failure mode.

## 16. The Typed Event Ledger

Every meaningful state transition during a run is recorded into an append-only typed event ledger. There are twenty distinct event types in the current build, organized into four families.

The **firm lifecycle** events are: normal firm entry, emergency firm entry (when the firm count drops to a minimum-floor-triggering level), firm exit, firm-exit-blocked (when a firm would have exited but the floor prevented it), firm insolvent (cash crossed the insolvency threshold), and firm-hire-refused (a work attempt was rejected by the solvency gate).

The **household-state** events are: household death, household hunger crisis (hunger crossed the crisis threshold), household displaced (typically from a flood or firm exit), household fatigue crisis, household infected, household severe illness onset, and household recovered.

The **healthcare bottleneck** events are: clinic overcrowded (demand past capacity in a given week) and clinic treatment denied (a specific household was untreated despite visiting).

The **disaster** events are: earthquake, fire, flood, supply shock, and epidemic wave.

Every event carries the simulated week it occurred, the actor identifier where applicable, and a payload of event-specific fields. Per-week event counts are written to the metrics CSV as twenty distinct columns. Episode totals are written to the summary JSON. This is the difference between a simulator that produces a number and a simulator that produces a *traceable history*. Any anomaly visible in the macro plots can be traced back to the specific weekly events that caused it. Why did GDP drop in week one hundred and twelve? The ledger will tell you that a fire affected three firms and a household death cascade started two weeks later. Nothing has to be inferred from net counts; everything is recorded as it happens.

## 17. The Struct-of-Arrays State Representation

A late but important refactor in the project replaced the original list-of-dicts state layout with a struct-of-arrays representation. The motivation is performance: numpy operations on parallel arrays are orders of magnitude faster than Python loops over dictionaries.

In the new representation, the household population lives in a `HouseholdState` object whose fields are fifteen numpy arrays of length N. Seven of them carry per-step physical state: wealth, health, hunger, fatigue, illness state, experience, and age. The remaining eight carry per-step bookkeeping consumed by observations, events, and Unity visualization: last action, routed firm id, action success flag, last reward, last wealth change, last health change, last crime success, and a displacement flag. The household identifier is the slot index in these arrays. Since no households are born or removed within an episode, the size never changes and the mapping from id to slot is trivial.

The firm population lives in a `FirmState` object whose fields are twenty-two numpy arrays of length max-firms. They cover the alive mask, the firm id, current price and wage offer, cash, current and target capacity, stock, productivity, weekly and monthly accumulators for sales, hires, unmet demand, revenue, wage bill, profit, previous price and wage for smoothness penalties, a negative-cash-weeks counter for exit detection, and a disaster downtime counter. A firm identifier is also a slot index. New firms occupy fresh slots; exited firms are marked dead via the `alive` boolean mask but never have their slots reused. This append-only discipline preserves bit-exact reproducibility against the original implementation: iteration order over slots zero through size-minus-one matches the insertion order of firms exactly, which means a single seeded run produces identical trajectories before and after the refactor. The pre-allocation also matters because the RLlib adapter exposes two hundred firm agent slots - far more than the maximum-alive-firms count of twelve - because firm IDs are monotonically increasing across the episode and slots are never reused.

Firms are not identical at initialization. Each firm's productivity is drawn from a log-normal distribution with median 3.0 (log-mean = ln(3) approximately 1.099, sigma = 0.2), so some firms are inherently more efficient than others. The heterogeneity matters: a firm with low productivity faces tighter solvency constraints than its competitors, even before any policy choices.

The struct-of-arrays representation is what makes vectorized dynamics possible. The hunger module operates on the entire population's hunger array in a single numpy call. The fatigue, illness, and epidemic modules do the same. The cost of advancing one weekly tick scales much more gracefully with population size than the old per-agent dict iteration could.

## 18. The RLlib Integration

The training itself uses Ray RLlib, a production multi-agent reinforcement learning library, configured for proximal policy optimization, or PPO. RLlib does not natively understand the simulator's notion of agents acting on different timescales, so the project includes an adapter layer that wraps the `World` object in a multi-agent gym-style environment.

The adapter does three important things. First, it gives each agent a flat string identifier - households become `h_0`, `h_1`, and so on, firms become `f_0` through `f_n`, the government is `gov`, the central bank is `cb`. Second, it accumulates rewards for slow agents across the weeks in which they do not act, delivering a lump-sum reward when they next take an action. Third, it filters the observation dictionary to only include agents that are scheduled to act on the current tick - households appear every week, firms only on firm-update weeks, government and central bank only on their quarterly weeks. This filtering is essential because otherwise RLlib would try to assign decisions to agents who, by design, are not yet making a new decision.

Each agent class has its own shared PPO policy module - `pi_household`, `pi_firm`, `pi_gov`, `pi_cb`. All households share `pi_household`, all firms share `pi_firm`. This parameter sharing is standard practice in multi-agent RL: it dramatically reduces sample complexity, and it reflects the assumption that all households are drawn from the same distribution of "what a household is."

The per-policy discount factor is chosen to match the agent's decision cadence. The base weekly discount is 0.995. Households use this base raised to the first power. Firms use it raised to the fourth power, which is about 0.980. The government and central bank use it raised to the twelfth power, which is about 0.942. Each policy's effective horizon is therefore matched to its actual update interval, which is one of the most important pieces of tuning in the system. A central bank with a household-frequency discount factor would be functionally myopic relative to the cadence at which it actually acts; matching its gamma to its decision interval restores the right time-preference scale.

The PPO hyperparameters are concrete. The base learning rate is three times ten to the minus four. The lambda parameter for generalized advantage estimation is 0.95. The clip parameter is 0.2. The training batch size per learner is 8192 transitions, and the minibatch size is 1024 - ten gradient steps per batch. The value-function loss coefficient is 0.5, the entropy coefficient is 0.01 by default, and gradient clipping is applied at a norm of 0.5. The policy networks are simple fully-connected MLPs with two hidden layers of 256 units each, applied uniformly to all four agent classes. RLlib is configured to use eight Ray actor processes for environment rollouts, each running two vectorized environments, with one GPU dedicated to the learner.

Per-policy fine tuning sits on top of these defaults. Each of the four policies has a fixed learning-rate multiplier relative to base: households train at the full base rate, firms at half, government at thirty percent, and central bank at the full base rate. These fixed multipliers stack with the curriculum stage LR scales, so a Stage-three firm policy with a curriculum scale of 0.30 actually trains at 0.5 times 0.30 - fifteen percent of base. Each policy also has its own entropy coefficient: 0.01 for households, 0.02 for firms, 0.05 for government, 0.03 for central bank. The government and central-bank policies have explicit entropy decay schedules: the government's entropy starts at 0.05, drops to 0.03 at thirty percent of training, to 0.01 at seventy percent, and to 0.005 at the end. The central bank starts at 0.03, drops to 0.01 at fifty percent, and ends at 0.005. These schedules let the macro agents explore aggressively early and exploit late, which is the standard prescription for low-dimensional discrete action spaces.

A subtle but important detail: every episode begins with a twelve-week warmup period during which all agents still act but their rewards are zeroed out. The warmup serves two purposes - it lets the simulator's macro metrics stabilize before being used for learning, and it prevents transient initial-condition artifacts from polluting the gradient signal. Twelve weeks is also exactly one quarter, so the first scheduled government and central-bank actions still see warmed-up macro inputs.

For development and debugging, a `compressed_quarterly` mode is available in stages one and two that drops the government and central-bank update period from twelve weeks down to six. The macro agents act twice as often, which lets a debugging run iterate over their decision cycle in half the wall-clock time. Production training does not use this mode.

## 19. The Batched-World Refactor

A significant recent effort in the project was a refactor that introduced a batched simulation core. The original `World` class ran one economy at a time. The batched version, called `BatchedWorld`, can run K economies in parallel inside a single process, with per-world random number generators that are spawned from a shared seed sequence so that each world has its own independent stochastic trajectory.

The motivation for this refactor was throughput. RL training is sample-hungry, and the bottleneck in earlier runs was the cost of stepping the simulator. By vectorizing the household and firm dynamics across K worlds simultaneously, the per-step computation can in principle be amortized.

The refactor was done with three explicit design goals. First, *maintenance cost* was eliminated by making the legacy single-world `World` a thin wrapper around `BatchedWorld(K=1)`, so there is exactly one implementation of the dynamics rather than two parallel ones. Second, *RNG determinism* was preserved by spawning per-world random generators, so a single world's trajectory is identical regardless of the batch size - a property that was verified by a bit-equivalence test at K=1 that compared thousands of records and found zero differences. Third, *debuggability* was preserved by adding cheap invariant assertions that run on every batched step, catching violations of wealth non-negativity, health and hunger and fatigue bounds, illness-state validity, NaN absence, firm cash finiteness, and firm-count bounds the moment they occur.

Three validation tests confirmed correctness end-to-end: a K=1 bit-equivalence test against the legacy implementation that compared thousands of records and found zero differences, a K=16 distributional test in which sixty-three Kolmogorov-Smirnov comparisons all passed at the one-percent significance level, and a K=8 smoke test that ran a large number of invariant assertions without a single failure.

A more honest finding from this work was that the batched environment is not yet faster than the single-environment path under RLlib's current overhead model. When K equals four, the batched environment exposes roughly three hundred and twenty households and twenty-four firms simultaneously to RLlib, versus eighty households and six firms in the single-world path. RLlib's per-agent policy mapping and buffer accounting overhead dominate the simulator-side savings. The current recommendation is therefore to train with K=1 in production, while preserving the batched code path for future optimization. The structural benefits of the refactor - single implementation, per-world RNG isolation, always-on invariant checks - hold regardless of throughput. The batched path is correct; it is simply waiting for a future RLlib configuration in which its speedup can be realized end-to-end.

## 20. Evaluation and the Unity Pipeline

After training, the project produces several layers of evaluation output.

The first is a step-by-step CSV of macro metrics for every week of every episode. It contains GDP, GDP per capita, GDP growth, inflation, Gini, poverty, unemployment, crime rate, average health, mean price, mean wage, firm count, household count, and aggregates over the human-condition stack - hunger, fatigue, and illness fractions. It also contains the twenty event-count columns from the typed ledger. Every interesting per-week phenomenon is in this file.

The second is a JSON summary capturing episode-final values, event totals, and collapse-detector status. If an episode terminated early, the summary records why, when, and the relevant state at termination.

The third is a set of standardized plots covering GDP, Gini, average health, poverty, crime, inflation, unemployment, firm count, price-wage trajectories, reward trajectories per agent class, and stylized-fact scatterplots like the Okun and Phillips relationships.

The fourth is a regime-comparison report that runs all four institutional configurations, computes pairwise cosine similarities between their macro outcomes, and produces weighted ranking tables.

A more ambitious evaluation path is the Unity playback pipeline. Each trained checkpoint can be loaded by an evaluation script that runs frozen inference - the policies no longer learn, they simply act - and produces a JSON file called `unity_playback.json` containing weekly snapshots of every household, every firm, the current policy state, and the current interest rate. This file is then consumed by a Unity 3D scene that renders the simulated economy visually: firms appear as buildings whose size and color encode capacity and cash, households move between weekly destinations, the government and central bank appear as distinct entities, and macro metrics are shown on an in-scene heads-up display. The Unity rendering is deliberately read-only - it is a playback of recorded simulator state, not a live training environment - which keeps the visualization concern cleanly separated from the learning concern.

There is also a smoke-test entry point that runs random-policy episodes across all four regimes and verifies a battery of runtime invariants: no NaN observations, no negative wealth, health bounded to the unit interval, finite firm cash, finite rewards, no rewards above a safety threshold, firm count between its hard bounds, and no routing to dead firms. The smoke test runs in seconds, which makes it cheap to invoke before any long training run. The project also includes a dedicated pytest suite at tests/test_invariants.py that covers unit-level invariants for hiring gates, epidemic mechanics, and household dynamics. The smoke test and the pytest suite together form the test infrastructure.

## 21. What This Simulator Is Useful For

The project is not a forecasting tool. It cannot tell anyone what policy a specific real country should adopt. It is too stylized for that. What it is useful for is reasoning about the *structural consequences* of different institutional configurations, in a setting where every causal mechanism is inspectable.

If a Social Democratic Welfare configuration produces a different equilibrium than a Liberal Market configuration - different inequality, different unemployment, different health, different infection burden - the simulator can be queried for the exact mechanism by which the difference arose. The event ledger reveals which households reached hunger crisis, which firms exited, which weeks the government's tax policy crossed which bound, which clinics overcrowded, which disasters struck. The plots reveal the time path of every macro variable. The snapshots reveal the spatial and individual heterogeneity that aggregate statistics conceal.

The project is also useful as a benchmark for multi-agent reinforcement learning itself. Most MARL benchmarks are abstract - predator-prey, cooperative navigation, gridworld coordination. An economy simulator with conflicting agent objectives, heterogeneous timescales, a layered human-condition stack, exogenous disaster shocks, and emergent macro dynamics is a much richer test of whether MARL methods can produce stable joint policies. The simulator architecture supports training all four agent classes simultaneously; validation of stable long-run equilibria across all four regimes is ongoing.

Finally, the project is useful pedagogically. The codebase is small enough to be read in a few hours, but it touches on labor economics, monetary economics, fiscal policy, criminology, public health, epidemiology, disaster economics, computational mechanism design, and modern deep reinforcement learning. As a teaching artifact, it shows what a self-contained economic simulator looks like when every part of it has to be implemented from first principles.

## 22. Key Engineering Choices in Summary

A few engineering choices deserve emphasis because they shape the entire project.

The orchestration is concentrated. There is one `step` method, in one class, and it is the unique source of state mutation. Every helper module is a pure function over state.

The agents are heterogeneous in timescale, observation, action, and reward, but they share one consistent world clock. Slow agents act less often, but their actions persist between updates, and their rewards accumulate.

Conservation is enforced. Money is not created or destroyed during routing, shopping, theft, clinic visits, or wage payment. This was not true in earlier versions and required several iterations of debugging to achieve.

The reward functions are explicit. Every coefficient is exposed as a named constant in a configuration file. Anyone reading the code can see exactly what each agent class is optimizing, in what units, with what weights.

The regimes are declarative. Changing from a Liberal Market Economy to a Social Democratic Welfare Economy requires no code change - only a regime name. The four regimes are encoded as data, with anchor values and bounds drawn from comparative-political-economy literature.

The curriculum is staged. The four-phase plan progressively introduces complexity and is essential for stable convergence in the multi-agent setting. No phase trains a brand-new agent class in isolation from agents that have not yet learned anything useful.

The human-condition stack is layered. Hunger, fatigue, and illness each operate independently but interact through shared health and shared susceptibility multipliers. The composition of these layers produces emergent realism that no single mechanic could provide alone.

The disaster generator is exogenous. No agent can predict or prevent the five disaster types. They function as stress tests on the trained policies, exposing whether the system has learned robust behavior or only brittle steady-state behavior.

The collapse detector is cause-tagged. Termination is never silent. Every early stop carries a reason - mass death, hyperinflation, hidden insolvency, blocked-exit cascade, or numerical invalidity - and that reason is recorded for post-hoc analysis.

The visualization is decoupled. The simulator does not know about plots, JSON snapshots, or Unity. Those are downstream consumers of state that the simulator exposes through a single snapshot method.

The state representation is struct-of-arrays. Every per-step field is a numpy array indexed by slot. Vectorized dynamics read and write these arrays directly without dict pack-and-unpack overhead, which is what makes the batched K-world path possible at all.

## 23. Closing: What the Project Demonstrates

This project demonstrates that an honest economic simulation can be built from reinforcement learning primitives, calibrated against real-world institutional taxonomies, extended with a non-trivial public-health and disaster layer, and made to produce traceable, replayable, falsifiable outcomes. The numbers it produces are stylized, but the *mechanisms* it produces are precise. Every weekly transition can be inspected. Every reward can be decomposed. Every collapse can be diagnosed. Every household death has a cause - old age, hunger, untreated severe illness, or an earthquake injury - and that cause is recorded.

This precision does not replace the empirical methods economists use to study real economies. It provides something those methods cannot easily provide: a setting in which the rules are fully known, the agents are fully observable, and the consequences of every change can be measured exactly. In a discipline where so much depends on what is unmeasurable, that kind of computational laboratory has clear value.

The project demonstrates that a working economy - with workers, firms, governments, central banks, infectious diseases, earthquakes, and the conflicts among them - can emerge from learning alone. Nothing has to be told to any agent about what it should do. They figure it out by living in the world together, under conditions that hurt them when they choose poorly.

---

*End of script. Total runtime when read at a moderate pace: approximately forty to forty-five minutes. Suitable for direct input into NotebookLM's audio overview pipeline, or for use as a written companion to a slide deck.*
