using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // Bottom scrubber: a bar with coloured ticks for each major event, a moving
    // playhead, and click/drag-to-jump. Built from the de-duplicated event log.
    public class MarlTimeline : MonoBehaviour
    {
        private MarlEconomyPlayback playback;
        private RectTransform bar;
        private RectTransform playhead;
        private bool built;

        public void Init(Canvas canvas, MarlEconomyPlayback pb)
        {
            playback = pb;

            bar = MakeImage(canvas.transform, "Timeline Bar",
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 0f),
                new Color(0.05f, 0.06f, 0.09f, 0.85f));
            bar.offsetMin = new Vector2(40f, 34f);
            bar.offsetMax = new Vector2(-40f, 34f + 20f);

            playhead = MakeImage(bar, "Playhead",
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(0.5f, 0.5f),
                new Color(1f, 1f, 1f, 0.95f));
            playhead.sizeDelta = new Vector2(3f, 0f);
        }

        private void Update()
        {
            if (playback == null || playback.EventBus == null || playback.Playback == null) return;
            int total = playback.TotalFrames - 1;
            if (total <= 0) return;

            if (!built) { BuildTicks(total); built = true; }

            // Move the playhead.
            float frac = Mathf.Clamp01(playback.CurrentSnapshot.week / (float)total);
            playhead.anchorMin = new Vector2(frac, 0f);
            playhead.anchorMax = new Vector2(frac, 1f);
            playhead.anchoredPosition = Vector2.zero;

            // Click / drag to scrub.
            if (Input.GetMouseButton(0))
            {
                Vector2 mouse = Input.mousePosition;
                if (RectTransformUtility.RectangleContainsScreenPoint(bar, mouse, null) &&
                    RectTransformUtility.ScreenPointToLocalPointInRectangle(bar, mouse, null, out Vector2 local))
                {
                    float w = bar.rect.width;
                    float f = Mathf.Clamp01((local.x + w * 0.5f) / w);
                    playback.SetFrame(Mathf.RoundToInt(f * total));
                    playback.autoPlay = false;
                }
            }
        }

        private void BuildTicks(int total)
        {
            foreach (var e in playback.EventBus.Log)
            {
                Color c; float height; float width = 2f;
                switch (e.kind)
                {
                    case MarlEventKind.Disaster:
                        MarlVisualPalette.Disaster(e.disaster, out c, out _);
                        height = 1f; width = 3f; break;
                    case MarlEventKind.FirmClose: c = new Color(0.95f, 0.5f, 0.18f); height = 0.7f; break;
                    case MarlEventKind.FirmOpen:  c = new Color(0.35f, 0.9f, 0.5f); height = 0.55f; break;
                    case MarlEventKind.Death:     c = new Color(0.85f, 0.85f, 0.9f); height = 0.45f; break;
                    default: continue; // skip per-week policy/fatigue noise
                }

                float frac = Mathf.Clamp01(e.week / (float)total);
                var tick = MakeImage(bar, "tick", new Vector2(frac, 0f), new Vector2(frac, height),
                    new Vector2(0.5f, 0f), c);
                tick.sizeDelta = new Vector2(width, 0f);
                tick.anchoredPosition = Vector2.zero;
            }
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
    }
}
