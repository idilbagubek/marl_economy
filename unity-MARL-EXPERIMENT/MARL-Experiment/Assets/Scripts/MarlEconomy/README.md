# MARL Economy — Unity Visualization

Replays a `unity_playback.json` export from the MARL-Economy simulation as an
animated low-poly city: rigged people (households) walking between homes, firms,
clinic and market; buildings (firms) that grow with capacity and recolour with
cash; and **unmistakable chaos events** — persistent environment changes, on-screen
alerts, a clickable timeline, and reacting agents.

Built for **Unity 6000.4.11f1 + URP**. No external packages required (uses the
built-in `JsonUtility`; no Newtonsoft).

## One-time setup

1. Open the project. Let it compile (Console should be clean).
2. Menu **`MARL Economy ▸ Rebuild Prefab Config`** — this scans the imported
   *Toon City* + *CityPeople* packs and writes
   `Assets/Resources/MarlEconomyPrefabConfig.asset`. (Skip this and you still get a
   primitive-shapes fallback.)
3. Open `Assets/Scenes/SampleScene.unity` and press **Play**.

Everything else (lights, camera, city, HUD, FX, timeline) is assembled at runtime
by `MarlEconomyAutoSetup` — nothing needs to be placed in the scene by hand.

## Swapping in your own models

All art is referenced from `MarlEconomyPrefabConfig`. Select the asset in
`Assets/Resources/` and drag your own prefabs into:
- `householdPrefabs` — your characters (need an `Animator`; walk/idle clips are
  matched by name: anything containing `walk`, `jog`/`running`, `idle`).
- `firmPrefabs`, `homePrefabs`, the landmark slots, and chaos vehicles.

No code changes or recompile needed — re-press Play.

## Controls

- **WASD / Arrows** move · **Right-drag** look · **Q/E** down/up · **Shift** fast
- **Space** pause · **, / .** slower / faster
- **Click the timeline** (bottom) to jump to a week; ticks mark firm openings
  (green), collapses (orange), deaths (white), and disasters (coloured).

## How the data is used (and de-bugged on the Unity side)

The exported JSON has two known quirks; the Unity layer compensates without any
Python changes:
- **Event names** (`firm_created`/`firm_closed`/`central_bank_rate_changed`/…) and
  **accumulating event lists** are normalized + de-duplicated by `MarlEventBus`,
  which keys each event off its own week and builds the timeline.
- **Disasters carry no position**, so `MarlEventBus` derives a site from the
  affected firms / displaced or sick households, and `MarlChaosDirector` plays the
  persistent overlay there (flood water, roof fire + fire-truck, quake rubble +
  camera shake, supply-shock global dim, epidemic fog + ambulance), reverting when
  the (inferred) window ends.

## Script map

| Script | Role |
|--------|------|
| `MarlEconomyAutoSetup` | runtime bootstrap; builds + wires the whole scene (SampleScene only) |
| `MarlEconomyPlayback` | loads JSON, spawns/animates firms & households, interpolates |
| `MarlEventBus` | normalizes/dedupes events, derives disaster windows + sites |
| `MarlChaosDirector` | persistent crisis visuals + global mood + crime heat |
| `MarlEconomyEventProcessor` | one-shot FX + alert banners |
| `MarlTimeline` | bottom scrubber with event ticks (click to jump) |
| `MarlHouseholdAgent` | per-person animation, facing, status orb |
| `MarlFirmVisual` | per-firm scale-by-capacity + cash beacon |
| `MarlEconomyHud` / `MarlEventBanner` | metrics panel / sliding alerts |
| `MarlWorldBuilder` | static city: ground, districts, roads, landmarks, homes, trees |
| `MarlEconomyPrefabConfig` | all art references (swap point) |
| `Editor/MarlPrefabConfigBuilder` | menu that fills the config from the packs |
