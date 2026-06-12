using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Owns the *persistent* look of a crisis: overlays that appear when a disaster
    // window opens and revert when it closes, plus global mood (supply-shock dim,
    // epidemic fog) and a decaying crime-heat patch on the risk zone.
    public class MarlChaosDirector : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public MarlEconomyFX fx;
        public MarlEconomyPrefabConfig config;

        private readonly List<DisasterWindow> _active = new List<DisasterWindow>();
        private readonly Dictionary<DisasterWindow, Overlay> _overlays = new Dictionary<DisasterWindow, Overlay>();
        private readonly List<DisasterWindow> _toRemove = new List<DisasterWindow>();

        private Light _keyLight;
        private float _baseLightIntensity = 1.15f;
        private Color _baseAmbient;
        private bool _baseCaptured;

        private GameObject _crimeDisc;
        private Material _crimeMat;
        private float _crimeHeat;

        private Transform _cam;
        private Vector3 _lastShake;

        private class Overlay
        {
            public GameObject root;
            public ParticleSystem loop;
            public Transform water;
        }

        private void Start()
        {
            foreach (var l in FindObjectsByType<Light>(FindObjectsSortMode.None))
                if (l.type == LightType.Directional) { _keyLight = l; break; }
            if (_keyLight != null) _baseLightIntensity = _keyLight.intensity;
            _baseAmbient = RenderSettings.ambientLight;
            _baseCaptured = true;

            BuildCrimeDisc();
        }

        private void Update()
        {
            if (playback == null || playback.EventBus == null || playback.CurrentSnapshot == null) return;
            int week = playback.CurrentSnapshot.week;

            playback.EventBus.ActiveWindows(week, _active);

            // Spawn overlays for newly-active windows.
            foreach (var w in _active)
                if (!_overlays.ContainsKey(w)) _overlays[w] = CreateOverlay(w);

            // Tear down overlays whose window ended.
            _toRemove.Clear();
            foreach (var kv in _overlays)
                if (!kv.Key.IsActive(week)) _toRemove.Add(kv.Key);
            foreach (var w in _toRemove) { DestroyOverlay(_overlays[w]); _overlays.Remove(w); }

            UpdateOverlays(week);
            UpdateGlobalMood();
            UpdateCrimeHeat();
        }

        private void LateUpdate()
        {
            if (playback == null || playback.CurrentSnapshot == null) return;
            if (_cam == null) _cam = Camera.main != null ? Camera.main.transform : null;
            if (_cam == null) return;

            // Earthquake shake: revert last offset, apply a fresh one if active.
            _cam.position -= _lastShake;
            float amp = 0f;
            foreach (var w in _active)
                if (w.kind == DisasterKind.Earthquake)
                    amp = Mathf.Max(amp, 0.35f * (1f - w.Progress(playback.CurrentSnapshot.week)));
            _lastShake = amp > 0f ? Random.insideUnitSphere * amp : Vector3.zero;
            _cam.position += _lastShake;
        }

        private Overlay CreateOverlay(DisasterWindow w)
        {
            var ov = new Overlay();
            ov.root = new GameObject("Chaos_" + w.kind);
            Vector3 site = ResolveSite(w);
            ov.root.transform.position = site;

            MarlVisualPalette.Disaster(w.kind, out Color a, out Color b);

            switch (w.kind)
            {
                case DisasterKind.Fire:
                    ov.loop = fx != null ? fx.PlayLooping(site + Vector3.up * 1.5f, "Fire", a, b,
                        60, 1.4f, 3.5f, 0.7f, 1.2f, 3.5f) : null;
                    if (ov.loop != null) ov.loop.transform.SetParent(ov.root.transform, true);
                    SpawnVehicle(config != null ? config.fireTruckPrefab : null, site + new Vector3(3f, 0f, 0f), ov.root.transform);
                    break;

                case DisasterKind.Flood:
                    ov.water = BuildWater(ov.root.transform, site);
                    break;

                case DisasterKind.Earthquake:
                    SpawnRubble(site, ov.root.transform);
                    break;

                case DisasterKind.EpidemicWave:
                    ov.loop = fx != null ? fx.PlayLooping(MarlEconomyPlayback.ClinicPos + Vector3.up * 1.2f, "Epidemic", a, b,
                        40, 2.0f, 1.5f, 0.5f, 3.5f, 1.2f) : null;
                    if (ov.loop != null) ov.loop.transform.SetParent(ov.root.transform, true);
                    SpawnVehicle(config != null ? config.ambulancePrefab : null, MarlEconomyPlayback.ClinicPos + new Vector3(3f, 0f, 0f), ov.root.transform);
                    break;

                case DisasterKind.SupplyShock:
                    // Global mood only (handled in UpdateGlobalMood).
                    break;
            }
            return ov;
        }

        private void UpdateOverlays(int week)
        {
            foreach (var kv in _overlays)
            {
                if (kv.Key.kind == DisasterKind.Flood && kv.Value.water != null)
                {
                    // Water rises as the flood progresses.
                    float p = kv.Key.Progress(week);
                    float y = Mathf.Lerp(-0.6f, 0.5f, Mathf.Sin(p * Mathf.PI)); // rise then recede
                    Vector3 pos = kv.Value.water.position;
                    kv.Value.water.position = new Vector3(pos.x, y, pos.z);
                }
            }
        }

        private void UpdateGlobalMood()
        {
            if (!_baseCaptured) return;
            bool supplyShock = false, epidemic = false;
            foreach (var w in _active)
            {
                if (w.kind == DisasterKind.SupplyShock) supplyShock = true;
                if (w.kind == DisasterKind.EpidemicWave) epidemic = true;
            }

            // Supply shock dims and desaturates the world.
            float targetLight = supplyShock ? _baseLightIntensity * 0.55f : _baseLightIntensity;
            if (_keyLight != null)
                _keyLight.intensity = Mathf.Lerp(_keyLight.intensity, targetLight, Time.deltaTime * 3f);
            Color targetAmbient = supplyShock ? _baseAmbient * 0.6f : _baseAmbient;
            RenderSettings.ambientLight = Color.Lerp(RenderSettings.ambientLight, targetAmbient, Time.deltaTime * 3f);

            // Epidemic casts a sickly green fog.
            if (epidemic)
            {
                RenderSettings.fog = true;
                RenderSettings.fogColor = Color.Lerp(RenderSettings.fogColor, new Color(0.45f, 0.62f, 0.40f), Time.deltaTime * 3f);
                RenderSettings.fogMode = FogMode.Linear;
                RenderSettings.fogStartDistance = 10f;
                RenderSettings.fogEndDistance = 120f;
            }
            else if (RenderSettings.fog)
            {
                RenderSettings.fogColor = Color.Lerp(RenderSettings.fogColor, new Color(0.7f, 0.78f, 0.85f), Time.deltaTime * 3f);
                if (RenderSettings.fogColor.g > 0.7f) RenderSettings.fog = false;
            }
        }

        private void UpdateCrimeHeat()
        {
            var m = playback.CurrentSnapshot.macro;
            float target = m != null ? Mathf.Clamp01(m.crime_rate * 8f) : 0f;
            _crimeHeat = Mathf.Lerp(_crimeHeat, target, Time.deltaTime * 1.5f);
            if (_crimeMat != null)
            {
                Color c = new Color(0.85f, 0.12f, 0.10f, Mathf.Clamp01(_crimeHeat) * 0.6f);
                if (_crimeMat.HasProperty("_BaseColor")) _crimeMat.SetColor("_BaseColor", c);
                _crimeMat.color = c;
            }
        }

        // Firms live on business-district lots, not at the data positions the
        // bus derived, so prefer the live firm object / its lot.
        private Vector3 ResolveSite(DisasterWindow w)
        {
            if (w.firmId >= 0 && playback != null && playback.TryGetFirmAnchor(w.firmId, out Vector3 p))
                return p;
            return w.hasSite ? w.site : MarlEconomyPlayback.MarketPos;
        }

        // ── construction helpers ─────────────────────────────────────────────

        private Transform BuildWater(Transform parent, Vector3 site)
        {
            var plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.name = "FloodWater";
            var col = plane.GetComponent<Collider>(); if (col != null) Destroy(col);
            plane.transform.SetParent(parent, true);
            plane.transform.position = new Vector3(site.x, -0.6f, site.z);
            plane.transform.localScale = new Vector3(2.2f, 1f, 2.2f); // ~22m
            var r = plane.GetComponent<Renderer>();
            r.sharedMaterial = config != null && config.waterMaterial != null
                ? config.waterMaterial
                : NewTransparent(new Color(0.18f, 0.42f, 0.85f, 0.55f));
            return plane.transform;
        }

        private void SpawnRubble(Vector3 site, Transform parent)
        {
            for (int i = 0; i < 5; i++)
            {
                Vector3 p = site + new Vector3(Random.Range(-3f, 3f), 0f, Random.Range(-3f, 3f));
                GameObject rock = MarlEconomyPrefabConfig.Pick(config != null ? config.propPrefabs : null, i);
                GameObject g = rock != null ? Instantiate(rock) : GameObject.CreatePrimitive(PrimitiveType.Cube);
                g.transform.SetParent(parent, true);
                g.transform.position = p;
                g.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
                g.transform.localScale = Vector3.one * Random.Range(0.5f, 1.1f);
            }
        }

        private void SpawnVehicle(GameObject prefab, Vector3 pos, Transform parent)
        {
            if (prefab == null) return;
            var v = Instantiate(prefab);
            MarlWindowGlass.RestyleUnder(v);
            v.transform.SetParent(parent, true);
            v.transform.position = pos;
        }

        private void BuildCrimeDisc()
        {
            _crimeDisc = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _crimeDisc.name = "CrimeHeat";
            var col = _crimeDisc.GetComponent<Collider>(); if (col != null) Destroy(col);
            _crimeDisc.transform.position = MarlEconomyPlayback.RiskZonePos + new Vector3(0f, 0.08f, 0f);
            _crimeDisc.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            _crimeDisc.transform.localScale = Vector3.one * 10f;
            _crimeMat = NewTransparent(new Color(0.85f, 0.12f, 0.10f, 0f));
            _crimeDisc.GetComponent<Renderer>().sharedMaterial = _crimeMat;
        }

        private static Material NewTransparent(Color color)
        {
            Shader s = Shader.Find("Universal Render Pipeline/Lit");
            if (s == null) s = Shader.Find("Standard");
            var m = new Material(s);
            m.SetFloat("_Surface", 1f);            // transparent
            m.SetFloat("_Blend", 0f);
            if (m.HasProperty("_SrcBlend")) m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (m.HasProperty("_DstBlend")) m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (m.HasProperty("_ZWrite")) m.SetInt("_ZWrite", 0);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            m.color = color;
            return m;
        }

        private void DestroyOverlay(Overlay ov)
        {
            if (ov == null) return;
            if (ov.loop != null) { ov.loop.Stop(); Destroy(ov.loop.gameObject, 2f); }
            if (ov.root != null) Destroy(ov.root);
        }
    }
}
