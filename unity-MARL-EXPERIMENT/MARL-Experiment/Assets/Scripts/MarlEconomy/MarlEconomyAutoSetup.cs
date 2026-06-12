using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // One-touch bootstrap: on entering Play in SampleScene it loads the config +
    // playback data, builds the city, spawns agents, and wires HUD / FX / banners /
    // chaos director / timeline. Nothing needs to be placed in the scene by hand.
    public static class MarlEconomyAutoSetup
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (SceneManager.GetActiveScene().name != "SampleScene") return;
            if (Object.FindFirstObjectByType<MarlEconomyPlayback>() != null) return;

            var config = Resources.Load<MarlEconomyPrefabConfig>("MarlEconomyPrefabConfig");
            if (config == null)
                Debug.LogWarning("[MARL] MarlEconomyPrefabConfig not found in Resources — using primitive fallbacks. " +
                                 "Run menu 'MARL Economy/Rebuild Prefab Config' to use the art packs.");

            EnsureLights();
            var cam = EnsureCamera();

            var playback = CreatePlayback(config);
            if (playback == null) return;

            MarlWorldBuilder.Build(config);

            var canvas = CreateHudCanvas();
            WireUi(playback, canvas, config);
            FrameCamera(cam);
        }

        private static MarlEconomyPlayback CreatePlayback(MarlEconomyPrefabConfig config)
        {
            TextAsset json = Resources.Load<TextAsset>("unity_playback");
            if (json == null)
            {
                Debug.LogError("[MARL] Resources/unity_playback.json not found.");
                return null;
            }
            var go = new GameObject("MarlEconomyPlayback");
            var pb = go.AddComponent<MarlEconomyPlayback>();
            pb.config = config;
            pb.autoPlay = true;
            pb.playbackSpeed = 1f;
            pb.Load(json.text);
            return pb;
        }

        private static void WireUi(MarlEconomyPlayback playback, Canvas canvas, MarlEconomyPrefabConfig config)
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            var metricsPanel = Panel(canvas.transform, "Metrics Panel",
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f),
                new Vector2(258f, 320f), new Color(0.04f, 0.05f, 0.08f, 0.74f));
            metricsPanel.anchoredPosition = new Vector2(10f, -10f);
            var metricsText = Label(metricsPanel.transform, font, 12,
                new Color(0.90f, 0.95f, 1f), TextAnchor.UpperLeft, "Loading...");

            var weekPanel = Panel(canvas.transform, "Week Panel",
                new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f),
                new Vector2(260f, 44f), new Color(0.04f, 0.05f, 0.08f, 0.74f));
            weekPanel.anchoredPosition = new Vector2(-10f, -10f);
            var weekText = Label(weekPanel.transform, font, 13,
                new Color(1f, 1f, 0.72f), TextAnchor.MiddleCenter, "Week 0");

            var hud = new GameObject("HUD Driver").AddComponent<MarlEconomyHud>();
            hud.transform.SetParent(canvas.transform, false);
            hud.playback = playback; hud.label = metricsText;

            var ctrl = new GameObject("Playback Controls").AddComponent<MarlPlaybackControls>();
            ctrl.transform.SetParent(canvas.transform, false);
            ctrl.playback = playback; ctrl.weekLabel = weekText;

            var fx = new GameObject("MarlEconomyFX").AddComponent<MarlEconomyFX>();
            if (config != null && config.particleMaterial != null) fx.particleMaterial = config.particleMaterial;

            var banner = new GameObject("MarlEventBanner").AddComponent<MarlEventBanner>();
            banner.transform.SetParent(canvas.transform, false);
            banner.Init(canvas);

            var proc = new GameObject("MarlEconomyEventProcessor").AddComponent<MarlEconomyEventProcessor>();
            proc.playback = playback; proc.fx = fx; proc.banner = banner;

            var chaos = new GameObject("MarlChaosDirector").AddComponent<MarlChaosDirector>();
            chaos.playback = playback; chaos.fx = fx; chaos.config = config;

            var timeline = new GameObject("MarlTimeline").AddComponent<MarlTimeline>();
            timeline.transform.SetParent(canvas.transform, false);
            timeline.Init(canvas, playback);

            var bar = Panel(canvas.transform, "Controls Bar",
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 0f),
                new Vector2(0f, 22f), new Color(0.02f, 0.02f, 0.04f, 0.55f));
            bar.anchoredPosition = new Vector2(0f, 60f);
            Label(bar.transform, font, 11, new Color(0.7f, 0.7f, 0.8f, 0.9f), TextAnchor.MiddleCenter,
                "WASD/Arrows move · Right-drag look · Q/E down/up · Shift fast · Space pause · ,/. speed · click timeline to jump");
        }

        // ── scene scaffolding ────────────────────────────────────────────────

        private static void EnsureLights()
        {
            bool hasDirectional = false;
            foreach (var l in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
                if (l.type == LightType.Directional) hasDirectional = true;

            if (!hasDirectional)
            {
                var key = new GameObject("MARL Key Light").AddComponent<Light>();
                key.type = LightType.Directional;
                key.intensity = 1.15f;
                key.color = new Color(1f, 0.96f, 0.86f);
                key.shadows = LightShadows.Soft;
                key.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            }
        }

        private static Camera EnsureCamera()
        {
            var cam = Camera.main;
            if (cam == null)
            {
                var go = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = go.AddComponent<Camera>();
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.66f, 0.77f, 0.91f);
            }
            cam.farClipPlane = 400f;
            if (cam.GetComponent<MarlFreeCamera>() == null) cam.gameObject.AddComponent<MarlFreeCamera>();
            return cam;
        }

        private static void FrameCamera(Camera cam)
        {
            if (cam == null) return;
            // Slightly east + higher so the business district is in frame too.
            cam.transform.position = new Vector3(8f, 40f, -52f);
            cam.transform.rotation = Quaternion.Euler(36f, 4f, 0f);
        }

        private static Canvas CreateHudCanvas()
        {
            var go = new GameObject("HUD Canvas");
            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            go.AddComponent<CanvasScaler>();
            go.AddComponent<GraphicRaycaster>();
            return canvas;
        }

        private static RectTransform Panel(Transform parent, string name,
            Vector2 aMin, Vector2 aMax, Vector2 pivot, Vector2 size, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = aMin; rt.anchorMax = aMax; rt.pivot = pivot; rt.sizeDelta = size;
            return rt;
        }

        private static Text Label(Transform parent, Font font, int size, Color color, TextAnchor anchor, string text)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = font; t.fontSize = size; t.color = color; t.text = text;
            t.alignment = anchor; t.supportRichText = false; t.horizontalOverflow = HorizontalWrapMode.Wrap;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(8f, 6f); rt.offsetMax = new Vector2(-8f, -6f);
            return t;
        }
    }
}
