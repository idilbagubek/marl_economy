using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    public class MarlEconomyPlayback : MonoBehaviour
    {
        [Header("Data")]
        public TextAsset playbackJson;

        [Header("Prefabs")]
        public GameObject householdPrefab;
        public GameObject firmPrefab;
        public GameObject governmentPrefab;
        public GameObject centralBankPrefab;
        public GameObject marketPrefab;
        public GameObject clinicPrefab;
        public GameObject bankPrefab;
        public GameObject restPrefab;
        public GameObject riskZonePrefab;

        [Header("Prefab Tuning")]
        public Vector3 householdPrefabScale = Vector3.one;
        public Vector3 firmPrefabScale = Vector3.one;
        public bool scaleFirmPrefabWithCapacity = true;
        public float firmPrefabYOffset = 0.0f;

        [Header("Institution Tuning")]
        public Vector3 governmentPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 centralBankPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 marketPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 clinicPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 bankPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 restPrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 riskZonePrefabScale = new Vector3(1.0f, 1.0f, 1.0f);
        public Vector3 generatedInstitutionScale = new Vector3(2.5f, 2.5f, 2.5f);

        [Header("Institution Positions")]
        public Vector3 governmentPosition = new Vector3(-19.0f, 0.0f, 22.0f);
        public Vector3 centralBankPosition = new Vector3(19.0f, 0.0f, 22.0f);
        public Vector3 marketPosition = new Vector3(0.0f, 0.0f, 0.0f);
        public Vector3 clinicPosition = new Vector3(-14.0f, 0.0f, 5.0f);
        public Vector3 bankPosition = new Vector3(14.0f, 0.0f, 5.0f);
        public Vector3 restPosition = new Vector3(-14.0f, 0.0f, -8.0f);
        public Vector3 riskZonePosition = new Vector3(14.0f, 0.0f, -8.0f);

        [Header("Materials")]
        public Material poorMaterial;
        public Material middleMaterial;
        public Material richMaterial;
        public Material eliteMaterial;
        public Material failedActionMaterial;

        [Header("Playback")]
        public Transform worldRoot;
        public bool autoPlay = true;
        public float playbackSpeed = 1.0f;

        public UnityPlaybackRoot Playback { get; private set; }
        public UnitySnapshot CurrentSnapshot { get; private set; }

        private readonly Dictionary<int, GameObject> households = new Dictionary<int, GameObject>();
        private readonly Dictionary<int, GameObject> firms = new Dictionary<int, GameObject>();
        private readonly Dictionary<string, GameObject> institutions = new Dictionary<string, GameObject>();
        private readonly List<GameObject> environmentObjects = new List<GameObject>();
        private int frameIndex;
        private float frameTimer;

        private void Start()
        {
            if (playbackJson != null)
            {
                Load(playbackJson.text);
            }
        }

        private void Update()
        {
            if (!autoPlay || Playback == null || Playback.snapshots.Count == 0)
            {
                return;
            }

            float secondsPerWeek = Mathf.Max(0.01f, Playback.replay_seconds_per_week);
            frameTimer += Time.deltaTime * Mathf.Max(0.0f, playbackSpeed);

            while (frameTimer >= secondsPerWeek)
            {
                frameTimer -= secondsPerWeek;
                frameIndex = Mathf.Min(frameIndex + 1, Playback.snapshots.Count - 1);
                SyncFrameObjects(Playback.snapshots[frameIndex]);
            }

            UnitySnapshot previous = Playback.snapshots[Mathf.Max(0, frameIndex - 1)];
            UnitySnapshot current = Playback.snapshots[frameIndex];
            CurrentSnapshot = current;
            InterpolateHouseholds(previous, current, frameTimer / secondsPerWeek);
        }

        public void Load(string json)
        {
            Playback = JsonConvert.DeserializeObject<UnityPlaybackRoot>(json);
            frameIndex = 0;
            frameTimer = 0.0f;

            ClearScene();
            if (Playback == null || Playback.snapshots.Count == 0)
            {
                Debug.LogWarning("MarlEconomyPlayback loaded no snapshots.");
                return;
            }

            BuildEnvironment();
            SpawnInstitutions(Playback.snapshots[0]);
            SyncFrameObjects(Playback.snapshots[0]);
            ApplyHouseholdsImmediate(Playback.snapshots[0]);
            CurrentSnapshot = Playback.snapshots[0];
            Debug.Log($"MarlEconomyPlayback loaded {Playback.snapshots.Count} snapshots, {Playback.snapshots[0].households.Count} households, and {Playback.snapshots[0].firms.Count} firms.");
        }

        public void TogglePlayback()
        {
            autoPlay = !autoPlay;
        }

        public void SetFrame(int index)
        {
            if (Playback == null || Playback.snapshots.Count == 0)
            {
                return;
            }

            frameIndex = Mathf.Clamp(index, 0, Playback.snapshots.Count - 1);
            frameTimer = 0.0f;
            SyncFrameObjects(Playback.snapshots[frameIndex]);
            ApplyHouseholdsImmediate(Playback.snapshots[frameIndex]);
            CurrentSnapshot = Playback.snapshots[frameIndex];
        }

        private void SyncFrameObjects(UnitySnapshot snapshot)
        {
            HashSet<int> aliveFirmIds = new HashSet<int>();
            foreach (FirmSnapshot firm in snapshot.firms)
            {
                aliveFirmIds.Add(firm.id);
                GameObject obj = EnsureFirm(firm.id);
                Vector3 pos = GetFirmPosition(firm);
                float height = Mathf.Clamp(1.0f + firm.capacity * 0.04f, 1.0f, 8.0f);
                ApplyFirmTransform(obj, pos, height);
                obj.name = $"Firm_{firm.id}_cash_{firm.cash:0.00}";
                ApplyFirmMaterial(obj, firm);
            }

            List<int> staleFirmIds = new List<int>();
            foreach (int firmId in firms.Keys)
            {
                if (!aliveFirmIds.Contains(firmId))
                {
                    staleFirmIds.Add(firmId);
                }
            }

            foreach (int firmId in staleFirmIds)
            {
                Destroy(firms[firmId]);
                firms.Remove(firmId);
            }

            foreach (HouseholdSnapshot household in snapshot.households)
            {
                GameObject obj = EnsureHousehold(household.id);
                ApplyHouseholdMaterial(obj, household);
                float healthScale = Mathf.Lerp(0.65f, 1.25f, Mathf.Clamp01(household.health));
                obj.transform.localScale = householdPrefab != null
                    ? householdPrefabScale * healthScale
                    : Vector3.one * healthScale;
                obj.name = $"Household_{household.id}_{household.action_label}";
            }
        }

        private void ApplyHouseholdsImmediate(UnitySnapshot snapshot)
        {
            foreach (HouseholdSnapshot household in snapshot.households)
            {
                EnsureHousehold(household.id).transform.position = GetHouseholdPosition(household);
            }
        }

        private void InterpolateHouseholds(UnitySnapshot previous, UnitySnapshot current, float t)
        {
            Dictionary<int, HouseholdSnapshot> previousById = new Dictionary<int, HouseholdSnapshot>();
            foreach (HouseholdSnapshot household in previous.households)
            {
                previousById[household.id] = household;
            }

            foreach (HouseholdSnapshot household in current.households)
            {
                GameObject obj = EnsureHousehold(household.id);
                Vector3 start = previousById.TryGetValue(household.id, out HouseholdSnapshot previousHousehold)
                    ? GetHouseholdPosition(previousHousehold)
                    : GetHouseholdHomePosition(household);
                Vector3 end = GetHouseholdPosition(household);
                obj.transform.position = Vector3.Lerp(start, end, Mathf.SmoothStep(0.0f, 1.0f, t));
            }
        }

        private void SpawnInstitutions(UnitySnapshot snapshot)
        {
            if (snapshot.institutions != null)
            {
                SpawnMarker("Government", governmentPrefab, governmentPosition, new Color(0.25f, 0.40f, 0.95f), governmentPrefabScale);
                SpawnMarker("Central Bank", centralBankPrefab, centralBankPosition, new Color(0.95f, 0.75f, 0.25f), centralBankPrefabScale);
                SpawnMarker("Market", marketPrefab, marketPosition, new Color(0.35f, 0.80f, 0.55f), marketPrefabScale);
            }

            SpawnMarker("Clinic", clinicPrefab, clinicPosition, new Color(0.95f, 0.35f, 0.35f), clinicPrefabScale);
            SpawnMarker("Investment Bank", bankPrefab, bankPosition, new Color(0.95f, 0.85f, 0.35f), bankPrefabScale);
            SpawnMarker("Rest", restPrefab, restPosition, new Color(0.45f, 0.75f, 1.0f), restPrefabScale);
            SpawnMarker("Risk Zone", riskZonePrefab, riskZonePosition, new Color(0.85f, 0.30f, 0.75f), riskZonePrefabScale);
        }

        private void BuildEnvironment()
        {
            SetSceneLighting();
            TintExistingGround();

            AddPad("Households District", new Vector3(0.0f, 0.025f, -14.0f), new Vector3(24.0f, 0.04f, 11.0f), new Color(0.83f, 0.88f, 0.91f));
            AddPad("Market District", new Vector3(0.0f, 0.03f, 0.0f), new Vector3(12.0f, 0.04f, 9.0f), new Color(0.79f, 0.89f, 0.80f));
            AddPad("Civic District", new Vector3(0.0f, 0.035f, 16.0f), new Vector3(13.0f, 0.04f, 7.0f), new Color(0.78f, 0.83f, 0.94f));

            AddRoad("Main Road North South", new Vector3(0.0f, 0.06f, 0.0f), new Vector3(2.0f, 0.05f, 41.0f));
            AddRoad("Main Road East West", new Vector3(0.0f, 0.065f, 0.0f), new Vector3(33.0f, 0.05f, 2.0f));
            AddRoad("Household Road", new Vector3(0.0f, 0.07f, -14.0f), new Vector3(23.0f, 0.05f, 1.4f));
        }

        private void SetSceneLighting()
        {
            RenderSettings.ambientLight = new Color(0.62f, 0.68f, 0.75f);
            Camera camera = Camera.main;
            if (camera != null)
            {
                camera.backgroundColor = new Color(0.68f, 0.78f, 0.92f);
            }
        }

        private void TintExistingGround()
        {
            GameObject ground = GameObject.Find("Plane");
            if (ground == null)
            {
                return;
            }

            Renderer renderer = ground.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = new Color(0.68f, 0.72f, 0.68f);
            }
        }

        private void AddPad(string name, Vector3 position, Vector3 scale, Color color)
        {
            GameObject pad = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pad.name = name;
            pad.transform.SetParent(ParentTransform(), false);
            pad.transform.position = position;
            pad.transform.localScale = scale;
            Renderer renderer = pad.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
            }
            environmentObjects.Add(pad);
        }

        private void AddRoad(string name, Vector3 position, Vector3 scale)
        {
            GameObject road = GameObject.CreatePrimitive(PrimitiveType.Cube);
            road.name = name;
            road.transform.SetParent(ParentTransform(), false);
            road.transform.position = position;
            road.transform.localScale = scale;
            Renderer renderer = road.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = new Color(0.24f, 0.25f, 0.27f);
            }
            environmentObjects.Add(road);
        }
        private void SpawnLandmark(string key, Vector3 position, Color color)
        {
            SpawnMarker(key, null, position, color, new Vector3(1.6f, 1.0f, 1.6f));
        }

        private void SpawnMarker(string key, GameObject prefab, Vector3 position, Color color, Vector3 scale)
        {
            Vector3 appliedScale = prefab != null ? scale : generatedInstitutionScale;
            GameObject obj = prefab != null
                ? Instantiate(prefab, ParentTransform())
                : CreateLandmarkVisual(key, color);
            obj.name = key;
            obj.transform.SetParent(ParentTransform(), false);
            obj.transform.position = position;
            obj.transform.localScale = appliedScale;

            AddLabel(obj, key);
            institutions[key] = obj;
        }

        private GameObject CreateLandmarkVisual(string key, Color color)
        {
            GameObject root = new GameObject($"{key}_Visual");

            GameObject baseObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            baseObject.name = "Base";
            baseObject.transform.SetParent(root.transform, false);
            baseObject.transform.localPosition = new Vector3(0.0f, 0.45f, 0.0f);
            baseObject.transform.localScale = new Vector3(1.0f, 0.9f, 1.0f);
            SetColor(baseObject, color);

            GameObject cap = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cap.name = "Cap";
            cap.transform.SetParent(root.transform, false);
            cap.transform.localPosition = new Vector3(0.0f, 1.12f, 0.0f);
            cap.transform.localScale = new Vector3(1.15f, 0.18f, 1.15f);
            SetColor(cap, Color.Lerp(color, Color.white, 0.25f));

            return root;
        }

        private void AddLabel(GameObject parent, string text)
        {
            GameObject labelObject = new GameObject($"{text}_Label");
            labelObject.transform.SetParent(parent.transform, false);
            labelObject.transform.localPosition = new Vector3(0.0f, 1.45f, 0.0f);

            TextMesh label = labelObject.AddComponent<TextMesh>();
            label.text = text;
            label.anchor = TextAnchor.MiddleCenter;
            label.alignment = TextAlignment.Center;
            label.characterSize = 0.16f;
            label.fontSize = 32;
            label.color = new Color(0.08f, 0.09f, 0.11f);
            labelObject.AddComponent<MarlBillboardLabel>();
        }

        private GameObject EnsureHousehold(int id)
        {
            if (households.TryGetValue(id, out GameObject existing))
            {
                return existing;
            }

            GameObject obj = householdPrefab != null
                ? Instantiate(householdPrefab, ParentTransform())
                : CreateHouseholdVisual(id);
            obj.name = $"Household_{id}";
            obj.transform.SetParent(ParentTransform(), false);
            households[id] = obj;
            return obj;
        }

        private GameObject EnsureFirm(int id)
        {
            if (firms.TryGetValue(id, out GameObject existing))
            {
                return existing;
            }

            GameObject obj = firmPrefab != null
                ? Instantiate(firmPrefab, ParentTransform())
                : CreateFirmVisual(id);
            obj.name = $"Firm_{id}";
            obj.transform.SetParent(ParentTransform(), false);
            firms[id] = obj;
            return obj;
        }

        private GameObject CreateHouseholdVisual(int id)
        {
            GameObject root = new GameObject($"Household_{id}_Visual");

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = "Body";
            body.transform.SetParent(root.transform, false);
            body.transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);
            body.transform.localScale = new Vector3(0.75f, 0.9f, 0.75f);

            GameObject shadow = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            shadow.name = "Shadow";
            shadow.transform.SetParent(root.transform, false);
            shadow.transform.localPosition = new Vector3(0.0f, -0.95f, 0.0f);
            shadow.transform.localScale = new Vector3(0.65f, 0.025f, 0.65f);
            SetColor(shadow, new Color(0.05f, 0.06f, 0.07f, 0.35f));

            return root;
        }

        private GameObject CreateFirmVisual(int id)
        {
            GameObject root = new GameObject($"Firm_{id}_Visual");

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cube);
            body.name = "Body";
            body.transform.SetParent(root.transform, false);

            GameObject roof = GameObject.CreatePrimitive(PrimitiveType.Cube);
            roof.name = "Roof";
            roof.transform.SetParent(root.transform, false);
            SetColor(roof, new Color(0.22f, 0.24f, 0.30f));

            GameObject windowBand = GameObject.CreatePrimitive(PrimitiveType.Cube);
            windowBand.name = "Window Band";
            windowBand.transform.SetParent(root.transform, false);
            SetColor(windowBand, new Color(0.82f, 0.92f, 1.0f));

            return root;
        }

        private void ApplyFirmTransform(GameObject obj, Vector3 pos, float height)
        {
            Transform body = obj.transform.Find("Body");
            Transform roof = obj.transform.Find("Roof");
            Transform windowBand = obj.transform.Find("Window Band");
            if (body == null)
            {
                obj.transform.position = firmPrefab != null
                    ? new Vector3(pos.x, firmPrefabYOffset, pos.z)
                    : new Vector3(pos.x, height * 0.5f, pos.z);
                if (firmPrefab != null)
                {
                    float capacityScale = scaleFirmPrefabWithCapacity
                        ? Mathf.Lerp(0.85f, 1.8f, Mathf.InverseLerp(1.0f, 8.0f, height))
                        : 1.0f;
                    obj.transform.localScale = new Vector3(
                        firmPrefabScale.x,
                        firmPrefabScale.y * capacityScale,
                        firmPrefabScale.z
                    );
                }
                else
                {
                    obj.transform.localScale = new Vector3(1.5f, height, 1.5f);
                }
                return;
            }

            obj.transform.position = new Vector3(pos.x, 0.0f, pos.z);
            obj.transform.localScale = Vector3.one;
            body.localPosition = new Vector3(0.0f, height * 0.5f, 0.0f);
            body.localScale = new Vector3(1.35f, height, 1.35f);

            if (roof != null)
            {
                roof.localPosition = new Vector3(0.0f, height + 0.18f, 0.0f);
                roof.localScale = new Vector3(1.65f, 0.35f, 1.65f);
            }

            if (windowBand != null)
            {
                windowBand.localPosition = new Vector3(0.0f, Mathf.Max(0.65f, height * 0.68f), -0.69f);
                windowBand.localScale = new Vector3(1.05f, 0.18f, 0.04f);
            }
        }

        private void ApplyHouseholdMaterial(GameObject obj, HouseholdSnapshot household)
        {
            Renderer renderer = obj.GetComponentInChildren<Renderer>();
            if (renderer == null)
            {
                return;
            }

            if (!household.action_success && failedActionMaterial != null)
            {
                renderer.material = failedActionMaterial;
                return;
            }

            Color baseColor;
            if (household.wealth_class == "poor" && poorMaterial != null)
            {
                renderer.material = poorMaterial;
                baseColor = renderer.material.color;
            }
            else if (household.wealth_class == "middle" && middleMaterial != null)
            {
                renderer.material = middleMaterial;
                baseColor = renderer.material.color;
            }
            else if (household.wealth_class == "rich" && richMaterial != null)
            {
                renderer.material = richMaterial;
                baseColor = renderer.material.color;
            }
            else if (household.wealth_class == "elite" && eliteMaterial != null)
            {
                renderer.material = eliteMaterial;
                baseColor = renderer.material.color;
            }
            else
            {
                baseColor = HouseholdColor(household.wealth_class);
                renderer.material.color = baseColor;
            }

            // Hunger tints toward a sallow yellow as hunger rises past 0.5;
            // fully tinted at the crisis threshold.
            float hungerVisible = Mathf.InverseLerp(0.5f, 0.95f, household.hunger);
            if (hungerVisible > 0.0f)
            {
                Color hungryTint = new Color(0.85f, 0.78f, 0.35f);
                renderer.material.color = Color.Lerp(baseColor, hungryTint, hungerVisible);
            }

            // Fatigue fades toward gray as the household becomes exhausted.
            float fatigueVisible = Mathf.InverseLerp(0.55f, 0.95f, household.fatigue);
            if (fatigueVisible > 0.0f)
            {
                Color fatigueTint = new Color(0.45f, 0.45f, 0.55f);
                renderer.material.color = Color.Lerp(renderer.material.color, fatigueTint, fatigueVisible * 0.5f);
            }

            // Illness overlay: severe = bright red, mild = pinkish, recovering = pale blue.
            if (household.illness == 3)
            {
                renderer.material.color = Color.Lerp(renderer.material.color,
                    new Color(0.95f, 0.20f, 0.20f), 0.7f);
            }
            else if (household.illness == 2)
            {
                renderer.material.color = Color.Lerp(renderer.material.color,
                    new Color(0.90f, 0.50f, 0.50f), 0.4f);
            }
            else if (household.illness == 4)
            {
                renderer.material.color = Color.Lerp(renderer.material.color,
                    new Color(0.55f, 0.85f, 0.95f), 0.3f);
            }
        }

        private void ApplyFirmMaterial(GameObject obj, FirmSnapshot firm)
        {
            Renderer renderer = obj.GetComponentInChildren<Renderer>();
            if (renderer == null)
            {
                return;
            }

            renderer.material.color = firm.spawned_this_step
                ? new Color(0.33f, 0.72f, 0.45f)
                : new Color(0.48f, 0.52f, 0.62f);
        }

        private void SetColor(GameObject obj, Color color)
        {
            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
            }
        }

        private Color HouseholdColor(string wealthClass)
        {
            if (wealthClass == "poor")
            {
                return new Color(0.25f, 0.55f, 1.0f);
            }
            if (wealthClass == "middle")
            {
                return new Color(0.25f, 0.85f, 0.55f);
            }
            if (wealthClass == "rich")
            {
                return new Color(1.0f, 0.78f, 0.25f);
            }
            if (wealthClass == "elite")
            {
                return new Color(0.95f, 0.35f, 0.85f);
            }

            return new Color(0.9f, 0.9f, 0.9f);
        }

        private Vector3 GetHouseholdPosition(HouseholdSnapshot household)
        {
            Vector3 position = household.position != null
                ? ToUnityVector(household.position)
                : FallbackHouseholdPosition(household.id);
            position.y = Mathf.Max(position.y, 1.0f);
            return position;
        }

        private Vector3 GetHouseholdHomePosition(HouseholdSnapshot household)
        {
            Vector3 position = household.home_position != null
                ? ToUnityVector(household.home_position)
                : FallbackHouseholdPosition(household.id);
            position.y = Mathf.Max(position.y, 1.0f);
            return position;
        }

        private Vector3 GetFirmPosition(FirmSnapshot firm)
        {
            return firm.position != null ? ToUnityVector(firm.position) : FallbackFirmPosition(firm.id);
        }

        private Vector3 FallbackHouseholdPosition(int id)
        {
            const int columns = 10;
            const float spacing = 2.0f;
            int row = id / columns;
            int column = id % columns;
            float x = (column - (columns - 1) * 0.5f) * spacing;
            float z = -8.0f + row * spacing;
            return new Vector3(x, 1.0f, z);
        }

        private Vector3 FallbackFirmPosition(int id)
        {
            float angle = id * 2.399963f;
            float radius = 7.0f + (id % 3) * 1.8f;
            return new Vector3(Mathf.Cos(angle) * radius, 0.0f, Mathf.Sin(angle) * radius);
        }

        private Vector3 ToUnityVector(Vec3Data value)
        {
            if (value == null)
            {
                return Vector3.zero;
            }

            float scale = Playback != null ? Playback.scene_scale : 1.0f;
            return new Vector3(value.x * scale, value.y * scale, value.z * scale);
        }

        private Transform ParentTransform()
        {
            return worldRoot != null ? worldRoot : transform;
        }

        private void ClearScene()
        {
            foreach (GameObject obj in households.Values)
            {
                Destroy(obj);
            }
            foreach (GameObject obj in firms.Values)
            {
                Destroy(obj);
            }
            foreach (GameObject obj in institutions.Values)
            {
                Destroy(obj);
            }
            households.Clear();
            firms.Clear();
            institutions.Clear();
            foreach (GameObject obj in environmentObjects)
            {
                Destroy(obj);
            }
            environmentObjects.Clear();
        }
    }

    public class MarlBillboardLabel : MonoBehaviour
    {
        private void LateUpdate()
        {
            Camera camera = Camera.main;
            if (camera == null)
            {
                return;
            }

            transform.rotation = Quaternion.LookRotation(transform.position - camera.transform.position);
        }
    }
}
