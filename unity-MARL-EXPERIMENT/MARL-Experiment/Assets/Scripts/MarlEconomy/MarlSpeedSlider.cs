using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // On-screen playback-speed slider. Built and driven WITHOUT an EventSystem —
    // it mirrors MarlTimeline's manual Input.mousePosition + RectTransformUtility
    // approach (the scene has no EventSystem/input module). Drag anywhere on the
    // panel to set playback.playbackSpeed; the handle also follows the keyboard
    // speed keys (comma/period) because it re-reads playbackSpeed every frame.
    public class MarlSpeedSlider : MonoBehaviour
    {
        public const float MinSpeed = 0f; // 0 = frozen
        public const float MaxSpeed = 8f;

        private MarlEconomyPlayback playback;
        private RectTransform panel;   // hit area for dragging
        private RectTransform track;   // maps x -> speed
        private RectTransform fill;
        private RectTransform handle;
        private Text label;
        private bool dragging;

        public void Init(Canvas canvas, MarlEconomyPlayback pb, Font font)
        {
            playback = pb;

            panel = MakeImage(canvas.transform, "Speed Panel",
                new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f),
                new Color(0.04f, 0.05f, 0.08f, 0.74f));
            panel.sizeDelta = new Vector2(260f, 42f);
            panel.anchoredPosition = new Vector2(-10f, -60f); // just below the week panel

            label = MakeText(panel, font, 12, new Color(0.85f, 0.92f, 1f), TextAnchor.UpperLeft, "Speed");
            label.rectTransform.offsetMin = new Vector2(10f, 18f);
            label.rectTransform.offsetMax = new Vector2(-10f, -4f);

            track = MakeImage(panel, "Track",
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 0f),
                new Color(0.16f, 0.18f, 0.24f, 1f));
            track.offsetMin = new Vector2(12f, 8f);
            track.offsetMax = new Vector2(-12f, 16f); // 8 px tall, 8 px off the bottom

            fill = MakeImage(track, "Fill",
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(0f, 0.5f),
                new Color(0.32f, 0.72f, 1f, 1f));

            handle = MakeImage(track, "Handle",
                new Vector2(0f, 0.5f), new Vector2(0f, 0.5f), new Vector2(0.5f, 0.5f),
                new Color(0.92f, 0.96f, 1f, 1f));
            handle.sizeDelta = new Vector2(10f, 16f);
        }

        private void Update()
        {
            if (playback == null || track == null) return;

            // Begin dragging on a press anywhere over the panel; map the pointer's
            // X within the track to a speed for as long as the button is held.
            if (Input.GetMouseButtonDown(0) &&
                RectTransformUtility.RectangleContainsScreenPoint(panel, Input.mousePosition, null))
                dragging = true;
            if (!Input.GetMouseButton(0)) dragging = false;

            if (dragging &&
                RectTransformUtility.ScreenPointToLocalPointInRectangle(track, Input.mousePosition, null, out Vector2 local))
            {
                float w = track.rect.width;
                float f = Mathf.Clamp01((local.x + w * 0.5f) / w);
                playback.playbackSpeed = Mathf.Lerp(MinSpeed, MaxSpeed, f);
            }

            // Reflect the current speed (also catches keyboard changes).
            float frac = Mathf.Clamp01((playback.playbackSpeed - MinSpeed) / (MaxSpeed - MinSpeed));
            handle.anchorMin = new Vector2(frac, 0.5f);
            handle.anchorMax = new Vector2(frac, 0.5f);
            handle.anchoredPosition = Vector2.zero;
            fill.anchorMax = new Vector2(frac, 1f);

            if (label != null)
                label.text = playback.playbackSpeed < 0.01f
                    ? "Speed   —   PAUSED"
                    : $"Speed   {playback.playbackSpeed:0.0}x";
        }

        private static RectTransform MakeImage(Transform parent, string name,
            Vector2 aMin, Vector2 aMax, Vector2 pivot, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = aMin; rt.anchorMax = aMax; rt.pivot = pivot;
            rt.sizeDelta = Vector2.zero; rt.anchoredPosition = Vector2.zero;
            return rt;
        }

        private static Text MakeText(Transform parent, Font font, int size, Color color, TextAnchor anchor, string text)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = font; t.fontSize = size; t.color = color; t.text = text;
            t.alignment = anchor; t.supportRichText = false; t.horizontalOverflow = HorizontalWrapMode.Overflow;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
            return t;
        }
    }
}
