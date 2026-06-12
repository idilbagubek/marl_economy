using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // Queued, sliding on-screen alert banners.
    public class MarlEventBanner : MonoBehaviour
    {
        private Canvas canvas;
        private Font font;
        private readonly Queue<(string message, Color color)> queue = new Queue<(string, Color)>();
        private bool running;

        public void Init(Canvas targetCanvas)
        {
            canvas = targetCanvas;
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }

        public void Post(string message, Color color)
        {
            if (canvas == null) return;
            queue.Enqueue((message, color));
            if (!running) StartCoroutine(Drain());
        }

        private IEnumerator Drain()
        {
            running = true;
            while (queue.Count > 0)
            {
                var entry = queue.Dequeue();
                yield return StartCoroutine(ShowBanner(entry.message, entry.color));
            }
            running = false;
        }

        private IEnumerator ShowBanner(string message, Color col)
        {
            var go = new GameObject("EventBanner");
            go.transform.SetParent(canvas.transform, false);

            var bg = go.AddComponent<Image>();
            bg.color = new Color(col.r * 0.10f, col.g * 0.10f, col.b * 0.10f, 0.93f);

            var rect = go.GetComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.5f, 0.5f);
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.pivot = new Vector2(0.5f, 0.5f);
            rect.sizeDelta = new Vector2(560f, 56f);

            var accentGo = new GameObject("Accent");
            accentGo.transform.SetParent(go.transform, false);
            accentGo.AddComponent<Image>().color = col;
            var ar = accentGo.GetComponent<RectTransform>();
            ar.anchorMin = new Vector2(0f, 0f);
            ar.anchorMax = new Vector2(0f, 1f);
            ar.pivot = new Vector2(0f, 0.5f);
            ar.sizeDelta = new Vector2(5f, 0f);
            ar.anchoredPosition = Vector2.zero;

            var textGo = new GameObject("Label");
            textGo.transform.SetParent(go.transform, false);
            var txt = textGo.AddComponent<Text>();
            txt.font = font;
            txt.fontSize = 16;
            txt.fontStyle = FontStyle.Bold;
            txt.color = Color.white;
            txt.text = message;
            txt.alignment = TextAnchor.MiddleCenter;
            txt.supportRichText = false;
            var tr = textGo.GetComponent<RectTransform>();
            tr.anchorMin = Vector2.zero;
            tr.anchorMax = Vector2.one;
            tr.offsetMin = new Vector2(14f, 0f);
            tr.offsetMax = new Vector2(-14f, 0f);

            float t = 0f;
            while (t < 1f)
            {
                t = Mathf.MoveTowards(t, 1f, Time.deltaTime * 7f);
                rect.anchoredPosition = new Vector2(0f, Mathf.Lerp(150f, 70f, Mathf.SmoothStep(0f, 1f, t)));
                yield return null;
            }

            yield return new WaitForSeconds(2.6f);

            var imgs = go.GetComponentsInChildren<Image>();
            var txts = go.GetComponentsInChildren<Text>();
            Color[] ic = System.Array.ConvertAll(imgs, x => x.color);
            Color[] tc = System.Array.ConvertAll(txts, x => x.color);
            t = 1f;
            while (t > 0f)
            {
                t = Mathf.MoveTowards(t, 0f, Time.deltaTime * 3.5f);
                for (int i = 0; i < imgs.Length; i++)
                    imgs[i].color = new Color(ic[i].r, ic[i].g, ic[i].b, ic[i].a * t);
                for (int i = 0; i < txts.Length; i++)
                    txts[i].color = new Color(tc[i].r, tc[i].g, tc[i].b, tc[i].a * t);
                yield return null;
            }
            Destroy(go);
        }
    }
}
