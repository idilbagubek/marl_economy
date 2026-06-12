# Unity Integration Plan

This document explains how to connect the `marl_economy` project to a Unity 3D visualization pipeline after training.

## Short Answer

Yes, this project can be visualized in Unity.

The right workflow is:

1. train policies in Python
2. save a checkpoint
3. run an evaluation episode using the trained checkpoint
4. export or stream world snapshots
5. render those snapshots in Unity as buildings, moving agents, and live graphs

## What Already Exists

The current codebase already has useful building blocks:

- `marl_economy/env/world.py`
  - exposes `get_snapshot()`
  - returns a dict with top-level keys: week, regime, households (each with position, action_label, wealth, health, etc.), firms (each with position, visual_type, etc.), institutions, macro, policy, interest_rate, and events

- `marl_economy/viz/export_json.py`
  - exports an episode as JSON snapshots

- `marl_economy/viz/export_unity.py`
  - converts an evaluation run folder into `unity_playback.json`
  - combines `snapshots.json`, `metrics_step.csv`, `summary.json`, and `config.json`

- `marl_economy/eval/evaluate_checkpoint.py`
  - loads a trained RLlib checkpoint
  - runs frozen inference
  - writes `metrics_step.csv`, `summary.json`, `snapshots.json`, plots, and now `unity_playback.json`

- `marl_economy/viz/viewer_pygame.py`
  - shows that live visualization is already possible

- `marl_economy/eval/plot_run.py`
  - generates static graphs from run logs

- `unity/Assets/Scripts/MarlEconomy/`
  - contains starter Unity C# scripts for JSON playback
  - spawns households, firms, government, central bank, and a market marker
  - animates households between weekly destinations
  - exposes a simple HUD script for macro metrics

## Current Bridge Status

The main Python-to-Unity bridge is now implemented.

After training finishes, run evaluation as usual:

```cmd
python -m marl_economy.eval.evaluate_checkpoint --checkpoint <path/to/checkpoint> --regime FreeMarket --output <path/to/output>
```

That evaluation command writes:

- `<output>\metrics_step.csv`
- `<output>\summary.json`
- `<output>\snapshots.json`
- `<output>\unity_playback.json`

If you already have an evaluation folder and only want to regenerate the Unity file:

```cmd
python -m marl_economy.viz.export_unity <path/to/eval/output>
```

Unity should consume `unity_playback.json`, not the training checkpoint directly.

## Can Unity Show New Firms as Buildings?

Yes.

When a new firm appears:

- Unity can spawn a new building prefab
- the building can be positioned in the 3D scene
- its size, color, or height can represent:
  - `capacity`
  - `cash`
  - `price`
  - `stock`

When a firm exits:

- the building can disappear
- or change into an abandoned/shutdown state

## Can Households Move Around the Scene?

Yes, but the snapshot schema should be extended.

The current snapshot contains:

- household `id`
- `wealth`
- `health`
- `last_action`
- `wealth_class`

For convincing Unity motion, it should also include:

- household world position
- current destination
- routed firm id
- action label such as `work`, `shop`, `clinic`, `rest`, `invest`, `steal`
- transaction success/failure

Then Unity can animate:

- agents walking to firms for work
- agents going shopping
- agents visiting clinics
- idle/resting behavior
- different materials or animations by class/health/action

## Important Modeling Detail

In the current simulation, firms are generic firms.

That means:

- there is not a separate economic class for "shop building"
- there is not a separate economic class for "clinic building"

Instead:

- `shop` and `clinic` are household actions
- they route to firms in the runtime

So in Unity you can either:

1. render all firms as generic buildings and animate visit types differently
2. extend the simulation to give firms explicit categories or service roles

## Best Integration Architecture

The safest version is offline playback first.

### Option 1: Offline Playback

Pipeline:

1. Python loads a trained checkpoint
2. Python runs one evaluation episode
3. Python writes snapshots to JSON
4. Unity loads the JSON file and replays it

Advantages:

- easiest to build
- deterministic
- stable for demos
- easiest to debug

### Option 2: Live Streaming

Pipeline:

1. Python runs the evaluation episode
2. each weekly snapshot is sent over WebSocket or TCP
3. Unity consumes snapshots live

Advantages:

- more interactive
- feels live during presentation

Disadvantages:

- more engineering complexity
- harder to debug
- harder to keep deterministic

Recommendation:

- build offline playback first
- add live streaming second

## Should the Simulation Keep Learning During Visualization?

Usually no.

There should be two separate modes:

### Training Mode

- RLlib updates policy weights
- agents continue learning
- used for offline training only

### Evaluation / Visualization Mode

- load a saved checkpoint
- freeze weights
- run inference only
- use this for Unity visualization

For demos, reports, and analysis, evaluation mode is the correct choice.

If the live simulation keeps learning:

- behavior changes during the demo
- results become harder to explain
- reproducibility gets worse
- debugging becomes much harder

So after training, the Unity-linked simulation should usually not keep learning.

## What Unity Could Show

A good Unity demo could include:

- firm creation as new buildings spawning
- firm exits as buildings closing or disappearing
- households moving between home, work, shopping, and clinic destinations
- building appearance changing with `capacity`, `cash`, or `stock`
- household color coding by `wealth_class`
- health indicators above agents
- macro dashboards showing:
  - GDP
  - Gini
  - inflation
  - crime
  - poverty
  - unemployment
- policy overlays for tax, welfare, policing, healthcare, education, minimum wage
- central bank interest-rate display

## Recommended Next Steps

Current implementation status:

1. [DONE] `evaluate_checkpoint.py` loads RLlib checkpoints and runs inference-only episodes
2. [DONE] `snapshots.json` and `unity_playback.json` are exported automatically
3. [DONE] `get_snapshot()` includes positions and event metadata
4. [DONE] Unity C# playback scripts consume `unity_playback.json`
5. add live streaming once offline playback is validated

## Suggested Snapshot Additions For Unity

The snapshot now includes the first version of these fields:

### Household additions

- `position`
- `destination`
- `home_position`
- `action_label`
- `target_firm_id`
- `action_success`
- `reward`
- `wealth_change`
- `health_change`
- `visual_state`

### Firm additions

- `position`
- `spawned_this_step`
- `visual_type`

### Global additions

- `events` -- list of event dicts; each has a `type` field (e.g. `firm_created`, `firm_closed`, `government_policy_changed`, `central_bank_rate_changed`, `household_death`)

## Unity Setup Steps

1. Create a Unity 3D project.
2. Install Newtonsoft JSON in Unity: Package Manager -> Add package by name -> `com.unity.nuget.newtonsoft-json`.
3. Copy or keep the folder `unity/Assets/Scripts/MarlEconomy/` inside the Unity project's `Assets/Scripts/` folder.
4. Copy an exported `unity_playback.json` into a Unity assets folder, for example `Assets/Data/unity_playback.json`.
5. Create an empty GameObject named `MarlEconomyPlayback`.
6. Add the `MarlEconomyPlayback` component.
7. Drag `unity_playback.json` into the component's `Playback Json` field.
8. Optionally assign custom prefabs for households, firms, government, central bank, market, clinic, bank, rest zone, and risk zone.
9. Press Play.

If no prefabs are assigned, the starter script creates default capsules for households and cubes for firms.

## Final Recommendation

Yes, Unity integration is realistic.

The clean architecture is:

- Python for training and inference
- JSON or sockets for state transfer
- Unity for 3D playback and presentation

The trained-checkpoint evaluation pipeline is now implemented in `marl_economy/eval/evaluate_checkpoint.py`. The next meaningful extension is live streaming (Option 2), or extending the snapshot schema with richer firm/household metadata.
