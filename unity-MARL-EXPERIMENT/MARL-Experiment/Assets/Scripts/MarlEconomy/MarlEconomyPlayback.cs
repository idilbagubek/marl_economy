using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Loads unity_playback.json and replays it: dynamic firms (buildings) and
    // households (rigged people) are instantiated from the prefab config and
    // driven each week; positions interpolate between weekly snapshots.
    //
    // Firms do NOT use the data positions (a ring through the middle of town
    // that collides with the static city). Instead each firm id is assigned a
    // lot in the business district (MarlCityLots) — the assignment is
    // precomputed from the whole run so it is deterministic under timeline
    // scrubbing, and lots are reused once their previous firm has closed.
    // Household walk targets that point at a firm are remapped to its lot.
    public class MarlEconomyPlayback : MonoBehaviour
    {
        // Fixed landmark anchors (match marl_economy/config/constants.py).
        public static readonly Vector3 GovernmentPos  = new Vector3(-19f, 0f, 22f);
        public static readonly Vector3 CentralBankPos = new Vector3(19f, 0f, 22f);
        public static readonly Vector3 MarketPos      = new Vector3(0f, 0f, 0f);
        public static readonly Vector3 ClinicPos      = new Vector3(-14f, 0f, 5f);
        public static readonly Vector3 BankPos        = new Vector3(14f, 0f, 5f);
        public static readonly Vector3 RestPos        = new Vector3(-14f, 0f, -8f);
        public static readonly Vector3 RiskZonePos    = new Vector3(14f, 0f, -8f);

        public MarlEconomyPrefabConfig config;
        public Transform agentsRoot;

        [Header("Playback")]
        public bool autoPlay = true;
        public float playbackSpeed = 1.0f;

        public UnityPlaybackRoot Playback { get; private set; }
        public UnitySnapshot CurrentSnapshot { get; private set; }
        public UnitySnapshot PreviousSnapshot { get; private set; }
        public MarlEventBus EventBus { get; private set; }
        public int FrameIndex => frameIndex;
        public int TotalFrames => Playback != null ? Playback.snapshots.Count : 0;

        private readonly Dictionary<int, GameObject> households = new Dictionary<int, GameObject>();
        private readonly Dictionary<int, MarlHouseholdAgent> agents = new Dictionary<int, MarlHouseholdAgent>();
        private readonly Dictionary<int, GameObject> firms = new Dictionary<int, GameObject>();
        private readonly Dictionary<int, MarlFirmVisual> firmVisuals = new Dictionary<int, MarlFirmVisual>();
        private readonly Dictionary<int, int> lotByFirm = new Dictionary<int, int>();
        private int maxConcurrentFirms;

        private int frameIndex;
        private float frameTimer;

        public void Load(string json)
        {
            Playback = JsonUtility.FromJson<UnityPlaybackRoot>(json);
            frameIndex = 0;
            frameTimer = 0f;
            ClearDynamic();

            if (Playback == null || Playback.snapshots == null || Playback.snapshots.Count == 0)
            {
                Debug.LogWarning("[MARL] Playback loaded no snapshots.");
                return;
            }

            BuildFirmLots();
            EventBus = MarlEventBus.Build(Playback);
            if (agentsRoot == null)
            {
                var root = new GameObject("MARL Agents");
                agentsRoot = root.transform;
            }

            SyncFrame(Playback.snapshots[0]);
            ApplyPositionsImmediate(Playback.snapshots[0]);
            CurrentSnapshot = Playback.snapshots[0];
            PreviousSnapshot = Playback.snapshots[0];

            Debug.Log($"[MARL] Loaded {Playback.snapshots.Count} weeks, " +
                      $"{Playback.snapshots[0].households.Count} households, " +
                      $"{Playback.snapshots[0].firms.Count} firms, " +
                      $"{lotByFirm.Count} firm ids over {maxConcurrentFirms} lots, " +
                      $"{EventBus.Log.Count} unique events, " +
                      $"{EventBus.DisasterWindows.Count} disaster windows.");
        }

        // Walk the whole run once and give every firm id a permanent lot in the
        // business district. A lot frees up when its firm leaves the snapshots,
        // so newly created firms reuse the plots of closed ones; the result is
        // a single id→lot map that holds for any week (scrub-safe).
        private void BuildFirmLots()
        {
            lotByFirm.Clear();
            maxConcurrentFirms = 0;
            var occupied = new Dictionary<int, int>(); // lot -> firm id
            var activeLot = new Dictionary<int, int>(); // firm id -> lot
            var prev = new HashSet<int>();
            var cur = new HashSet<int>();

            foreach (var snap in Playback.snapshots)
            {
                cur.Clear();
                if (snap.firms != null)
                    foreach (var f in snap.firms) cur.Add(f.id);

                foreach (var id in prev)
                {
                    if (cur.Contains(id) || !activeLot.TryGetValue(id, out int l)) continue;
                    occupied.Remove(l);
                    activeLot.Remove(id);
                }

                foreach (var id in cur)
                {
                    if (activeLot.ContainsKey(id)) continue;
                    if (!(lotByFirm.TryGetValue(id, out int lot) && !occupied.ContainsKey(lot)))
                    {
                        lot = 0;
                        while (occupied.ContainsKey(lot)) lot++;
                    }
                    lotByFirm[id] = lot;
                    activeLot[id] = lot;
                    occupied[lot] = id;
                }

                maxConcurrentFirms = Mathf.Max(maxConcurrentFirms, occupied.Count);
                prev.Clear();
                foreach (var id in cur) prev.Add(id);
            }
        }

        private int LotOf(int firmId)
        {
            if (lotByFirm.TryGetValue(firmId, out int lot)) return lot;
            lot = lotByFirm.Count; // unseen id (shouldn't happen) → next plot
            lotByFirm[firmId] = lot;
            return lot;
        }

        public Vector3 FirmAnchor(int firmId) => MarlCityLots.LotPosition(LotOf(firmId));

        // Best current position for a firm id: live object, else its lot.
        public bool TryGetFirmAnchor(int firmId, out Vector3 pos)
        {
            if (firms.TryGetValue(firmId, out var go) && go != null)
            {
                pos = go.transform.position;
                return true;
            }
            if (lotByFirm.TryGetValue(firmId, out int lot))
            {
                pos = MarlCityLots.LotPosition(lot);
                return true;
            }
            pos = MarketPos;
            return false;
        }

        private void Update()
        {
            if (Playback == null || Playback.snapshots.Count == 0) return;

            float secondsPerWeek = Mathf.Max(0.01f, Playback.replay_seconds_per_week);
            if (autoPlay)
            {
                frameTimer += Time.deltaTime * Mathf.Max(0f, playbackSpeed);
                while (frameTimer >= secondsPerWeek)
                {
                    frameTimer -= secondsPerWeek;
                    frameIndex = Mathf.Min(frameIndex + 1, Playback.snapshots.Count - 1);
                    PreviousSnapshot = Playback.snapshots[Mathf.Max(0, frameIndex - 1)];
                    SyncFrame(Playback.snapshots[frameIndex]);
                }
            }

            UnitySnapshot prev = Playback.snapshots[Mathf.Max(0, frameIndex - 1)];
            UnitySnapshot cur = Playback.snapshots[frameIndex];
            CurrentSnapshot = cur;
            PreviousSnapshot = prev;
            float t = autoPlay ? Mathf.Clamp01(frameTimer / secondsPerWeek) : 1f;
            Interpolate(prev, cur, t);
        }

        public void TogglePlayback() => autoPlay = !autoPlay;

        public void SetFrame(int index)
        {
            if (Playback == null || Playback.snapshots.Count == 0) return;
            frameIndex = Mathf.Clamp(index, 0, Playback.snapshots.Count - 1);
            frameTimer = 0f;
            PreviousSnapshot = Playback.snapshots[Mathf.Max(0, frameIndex - 1)];
            SyncFrame(Playback.snapshots[frameIndex]);
            ApplyPositionsImmediate(Playback.snapshots[frameIndex]);
            CurrentSnapshot = Playback.snapshots[frameIndex];
        }

        private void SyncFrame(UnitySnapshot snapshot)
        {
            // Firms: ensure/update on their lots, then despawn the closed ones.
            var alive = new HashSet<int>();
            foreach (var firm in snapshot.firms)
            {
                alive.Add(firm.id);
                int lot = LotOf(firm.id);
                MarlFirmVisual fv = EnsureFirm(firm.id);
                fv.transform.position = MarlCityLots.LotPosition(lot);
                fv.transform.rotation = Quaternion.Euler(0f, MarlCityLots.LotYaw(lot), 0f);
                fv.Apply(firm);
                fv.gameObject.name = $"Firm_{firm.id}_lot{lot}_cash{firm.cash:0.0}";
            }
            var stale = new List<int>();
            foreach (var id in firms.Keys) if (!alive.Contains(id)) stale.Add(id);
            foreach (var id in stale)
            {
                Destroy(firms[id]);
                firms.Remove(id);
                firmVisuals.Remove(id);
            }

            // Households: ensure/update visual state.
            foreach (var h in snapshot.households)
            {
                EnsureHousehold(h.id, out MarlHouseholdAgent agent);
                agent.ApplyWeek(h);
            }
        }

        private void ApplyPositionsImmediate(UnitySnapshot snapshot)
        {
            foreach (var h in snapshot.households)
            {
                if (households.TryGetValue(h.id, out var go))
                    go.transform.position = ResolveHouseholdPos(h);
            }
        }

        private void Interpolate(UnitySnapshot prev, UnitySnapshot cur, float t)
        {
            var prevById = new Dictionary<int, HouseholdSnapshot>();
            foreach (var h in prev.households) prevById[h.id] = h;

            float e = Mathf.SmoothStep(0f, 1f, t);
            foreach (var h in cur.households)
            {
                if (!households.TryGetValue(h.id, out var go)) continue;
                Vector3 a = prevById.TryGetValue(h.id, out var ph)
                    ? ResolveHouseholdPos(ph)
                    : ToWorld(h.home_position, FallbackHomePos(h.id));
                Vector3 b = ResolveHouseholdPos(h);
                go.transform.position = Vector3.Lerp(a, b, e);
            }
        }

        // Where this household should stand this week. Firm-routed actions
        // (work/shop) walk to the firm's LOT, not the stale data position; each
        // agent gets a deterministic spot by the building entrance so crowds
        // spread out instead of stacking. Everything else (home, clinic, bank,
        // rest, risk zone) uses the data position — those anchors still match.
        private Vector3 ResolveHouseholdPos(HouseholdSnapshot h)
        {
            if (h == null) return Vector3.zero;
            if ((h.action_label == "work" || h.action_label == "shop")
                && lotByFirm.TryGetValue(h.target_firm_id, out int lot))
                return MarlCityLots.LotPosition(lot) + DoorOffset(h.id, lot);
            return ToWorld(h.position, FallbackHomePos(h.id));
        }

        private static Vector3 DoorOffset(int householdId, int lot)
        {
            Vector3 front = MarlCityLots.LotFront(lot);
            Vector3 side = Vector3.Cross(Vector3.up, front);
            float u = Mathf.Repeat(householdId * 0.6180339f, 1f);
            float v = Mathf.Repeat(householdId * 0.7548777f, 1f);
            return front * (3.6f + u * 1.6f) + side * (v * 5.2f - 2.6f);
        }

        private MarlFirmVisual EnsureFirm(int id)
        {
            if (firmVisuals.TryGetValue(id, out var existing)) return existing;

            var holder = new GameObject($"Firm_{id}");
            holder.transform.SetParent(agentsRoot, false);
            var fv = holder.AddComponent<MarlFirmVisual>();

            GameObject prefab = config != null ? MarlEconomyPrefabConfig.Pick(config.firmPrefabs, id) : null;
            GameObject model = prefab != null ? Instantiate(prefab) : CreateFirmFallback();
            MarlWindowGlass.RestyleUnder(model);
            fv.Init(model, config != null ? config.firmScale : 1f);

            firms[id] = holder;
            firmVisuals[id] = fv;
            return fv;
        }

        private void EnsureHousehold(int id, out MarlHouseholdAgent agent)
        {
            if (agents.TryGetValue(id, out agent)) return;

            GameObject prefab = config != null ? MarlEconomyPrefabConfig.Pick(config.householdPrefabs, id) : null;
            GameObject go = prefab != null ? Instantiate(prefab) : CreateHouseholdFallback(id);
            go.name = $"Household_{id}";
            go.transform.SetParent(agentsRoot, false);
            agent = go.AddComponent<MarlHouseholdAgent>();
            agent.Init(config);

            households[id] = go;
            agents[id] = agent;
        }

        // ── Lookups for the chaos director / timeline ────────────────────────
        public GameObject GetFirmObject(int id) => firms.TryGetValue(id, out var g) ? g : null;
        public MarlFirmVisual GetFirmVisual(int id) => firmVisuals.TryGetValue(id, out var f) ? f : null;
        public GameObject GetHouseholdObject(int id) => households.TryGetValue(id, out var g) ? g : null;
        public IEnumerable<GameObject> AllFirmObjects() => firms.Values;
        public IEnumerable<GameObject> AllHouseholdObjects() => households.Values;

        public Vector3 ToWorld(Vec3Data v, Vector3 fallback)
        {
            if (v == null) return fallback;
            float s = Playback != null ? Playback.scene_scale : 1f;
            return new Vector3(v.x * s, v.y * s, v.z * s);
        }

        private static Vector3 FallbackHomePos(int id)
        {
            const int cols = 10;
            int row = id / cols, col = id % cols;
            return new Vector3((col - (cols - 1) * 0.5f) * 2.25f, 0f, -22f + row * 2.0f);
        }

        private GameObject CreateFirmFallback()
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.localScale = new Vector3(1.4f, 2.0f, 1.4f);
            cube.transform.localPosition = new Vector3(0f, 1.0f, 0f);
            return cube;
        }

        private GameObject CreateHouseholdFallback(int id)
        {
            var cap = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            cap.transform.localScale = new Vector3(0.6f, 0.9f, 0.6f);
            return cap;
        }

        private void ClearDynamic()
        {
            foreach (var g in households.Values) if (g) Destroy(g);
            foreach (var g in firms.Values) if (g) Destroy(g);
            households.Clear(); agents.Clear(); firms.Clear(); firmVisuals.Clear();
            lotByFirm.Clear();
        }
    }
}
