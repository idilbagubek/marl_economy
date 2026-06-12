using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    public static class MarlEconomyAutoSetup
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (Object.FindObjectOfType<MarlEconomyPlayback>() != null)
                return;

            CreateGround();
            CreateLights();
            var playback = CreatePlayback();
            if (playback == null) return;
            CreateCamera();
            var canvas = CreateHudCanvas();
            CreateFxAndEvents(playback, canvas);
            CreateControlsHint(canvas);
        }

        // ── Environment ──────────────────────────────────────────────────────

        private static void CreateGround()
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Plane);
            go.name = "Plane";
            go.transform.localScale = new Vector3(5f, 1f, 5f);
            var rend = go.GetComponent<Renderer>();
            if (rend != null)
                rend.material.color = new Color(0.66f, 0.70f, 0.66f);
        }

        private static void CreateLights()
        {
            // Key light — warm sun
            var key = new GameObject("Key Light");
            var kl = key.AddComponent<Light>();
            kl.type = LightType.Directional;
            kl.intensity = 1.15f;
            kl.color = new Color(1.00f, 0.96f, 0.86f);
            kl.shadows = LightShadows.Soft;
            key.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            // Fill light — cool sky
            var fill = new GameObject("Fill Light");
            var fl = fill.AddComponent<Light>();
            fl.type = LightType.Directional;
            fl.intensity = 0.38f;
            fl.color = new Color(0.70f, 0.80f, 1.00f);
            fl.shadows = LightShadows.None;
            fill.transform.rotation = Quaternion.Euler(28f, 148f, 0f);

            // Ambient
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.52f, 0.58f, 0.66f);
        }

        // ── Playback ─────────────────────────────────────────────────────────

        private static MarlEconomyPlayback CreatePlayback()
        {
            var go = new GameObject("MarlEconomyPlayback");
            var playback = go.AddComponent<MarlEconomyPlayback>();
            playback.autoPlay = true;
            playback.playbackSpeed = 1f;

            // Apply prefabs from config if available
            var config = Resources.Load<MarlEconomyPrefabConfig>("MarlEconomyPrefabConfig");
            if (config != null)
            {
                playback.householdPrefab    = config.householdPrefab;
                playback.firmPrefab         = config.firmPrefab;
                playback.governmentPrefab   = config.governmentPrefab;
                playback.centralBankPrefab  = config.centralBankPrefab;
                playback.marketPrefab       = config.marketPrefab;
                playback.clinicPrefab       = config.clinicPrefab;
                playback.bankPrefab         = config.bankPrefab;
                playback.restPrefab         = config.restPrefab;
                playback.riskZonePrefab     = config.riskZonePrefab;
                playback.poorMaterial       = config.poorMaterial;
                playback.middleMaterial     = config.middleMaterial;
                playback.richMaterial       = config.richMaterial;
                playback.eliteMaterial      = config.eliteMaterial;
                playback.failedActionMaterial = config.failedActionMaterial;
            }

            TextAsset json = Resources.Load<TextAsset>("unity_playback");
            if (json == null)
            {
                Debug.LogError("[MarlEconomy] Assets/Resources/unity_playback.json not found.");
                return null;
            }
            playback.Load(json.text);
            return playback;
        }

        // ── Camera ───────────────────────────────────────────────────────────

        private static void CreateCamera()
        {
            var go = new GameObject("Main Camera");
            go.tag = "MainCamera";
            var cam = go.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.66f, 0.77f, 0.91f);
            cam.farClipPlane = 300f;
            var fc = go.AddComponent<MarlFreeCamera>();
            fc.moveSpeed = 14f;
            go.transform.position = new Vector3(0f, 28f, -38f);
            go.transform.rotation = Quaternion.Euler(36f, 0f, 0f);
        }

        // ── HUD ──────────────────────────────────────────────────────────────

        private static Canvas CreateHudCanvas()
        {
            var canvasGo = new GameObject("HUD Canvas");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();
            return canvas;
        }

        private static void CreateFxAndEvents(MarlEconomyPlayback playback, Canvas canvas)
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            // Metrics panel — top left
            var metricsPanel = MakePanel(canvas.transform, "Metrics Panel",
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f),
                new Vector2(256f, 356f), new Color(0.04f, 0.05f, 0.08f, 0.72f));
            metricsPanel.anchoredPosition = new Vector2(10f, -10f);
            var metricsText = MakeText(metricsPanel.transform, "Metrics Text",
                font, 12, new Color(0.90f, 0.95f, 1.00f), TextAnchor.UpperLeft, "Loading...");

            // Week / speed panel — top right
            var weekPanel = MakePanel(canvas.transform, "Week Panel",
                new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f),
                new Vector2(244f, 44f), new Color(0.04f, 0.05f, 0.08f, 0.72f));
            weekPanel.anchoredPosition = new Vector2(-10f, -10f);
            var weekText = MakeText(weekPanel.transform, "Week Text",
                font, 13, new Color(1f, 1f, 0.72f), TextAnchor.MiddleCenter, "Week 0");

            // HUD driver
            var hudGo = new GameObject("HUD Driver");
            hudGo.transform.SetParent(canvas.transform, false);
            var hud = hudGo.AddComponent<MarlEconomyHud>();
            hud.playback = playback;
            hud.label = metricsText;

            // Playback controls driver
            var ctrlGo = new GameObject("Playback Controls");
            ctrlGo.transform.SetParent(canvas.transform, false);
            var ctrl = ctrlGo.AddComponent<MarlPlaybackControls>();
            ctrl.playback = playback;
            ctrl.weekLabel = weekText;

            // Particle FX
            var fxGo = new GameObject("MarlEconomyFX");
            var fx = fxGo.AddComponent<MarlEconomyFX>();

            // Event banner
            var bannerGo = new GameObject("MarlEventBanner");
            bannerGo.transform.SetParent(canvas.transform, false);
            var bannerComp = bannerGo.AddComponent<MarlEventBanner>();
            bannerComp.Init(canvas);

            // Event processor — watches snapshots, triggers FX and banners
            var procGo = new GameObject("MarlEconomyEventProcessor");
            var proc = procGo.AddComponent<MarlEconomyEventProcessor>();
            proc.playback = playback;
            proc.fx = fx;
            proc.banner = bannerComp;
        }

        private static void CreateControlsHint(Canvas canvas)
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            var bar = MakePanel(canvas.transform, "Controls Bar",
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 0f),
                new Vector2(0f, 26f), new Color(0.02f, 0.02f, 0.04f, 0.55f));
            bar.anchoredPosition = new Vector2(0f, 4f);
            MakeText(bar.transform, "Controls Text", font, 11,
                new Color(0.68f, 0.68f, 0.78f, 0.90f), TextAnchor.MiddleCenter,
                "WASD / Arrows: move   Right-click: look   Q/E: down/up   Shift: fast   Space: pause   , / .: slower/faster");
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        private static RectTransform MakePanel(Transform parent, string name,
            Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot, Vector2 size, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = anchorMin;
            rect.anchorMax = anchorMax;
            rect.pivot = pivot;
            rect.sizeDelta = size;
            return rect;
        }

        private static Text MakeText(Transform parent, string name, Font font,
            int fontSize, Color color, TextAnchor anchor, string initialText)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = font;
            t.fontSize = fontSize;
            t.color = color;
            t.text = initialText;
            t.alignment = anchor;
            t.supportRichText = false;
            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = new Vector2(7f, 7f);
            rect.offsetMax = new Vector2(-7f, -7f);
            return t;
        }
    }
}
