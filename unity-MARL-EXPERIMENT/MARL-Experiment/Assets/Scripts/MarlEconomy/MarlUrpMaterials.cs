using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // SimplePoly City (like most imported low-poly packs) ships every material on
    // the Built-in "Standard" shader. Under URP those render flat MAGENTA — the
    // "unrendered" look. The editor menu "MARL Economy/Fix Magenta Materials"
    // converts the .mat ASSETS on disk, but this RUNTIME pass makes Play "just
    // work" even if that menu was never run: every instantiated prefab is walked
    // and any non-URP material is swapped for a URP/Lit equivalent with the same
    // albedo texture + colour.
    //
    // Conversions are cached per source material, so all instances of a building
    // share one converted material (draw-call batching stays intact) and the
    // original .mat asset on disk is never modified.
    public static class MarlUrpMaterials
    {
        private static readonly Dictionary<Material, Material> Cache =
            new Dictionary<Material, Material>();

        private static Shader _lit;
        private static bool _searched;

        private static Shader Lit
        {
            get
            {
                if (!_searched)
                {
                    _lit = Shader.Find("Universal Render Pipeline/Lit");
                    _searched = true;
                }
                return _lit;
            }
        }

        // Replace every non-URP material under `go` so nothing renders magenta.
        // No-op for prefabs already authored on a URP shader (e.g. the CityPeople
        // characters), and for non-URP projects (Shader.Find returns null).
        public static void Fix(GameObject go)
        {
            if (go == null) return;
            Shader lit = Lit;
            if (lit == null) return;

            var renderers = go.GetComponentsInChildren<Renderer>(true);
            foreach (var r in renderers)
            {
                // Only touch mesh-based renderers — never particles/trails/lines.
                if (!(r is MeshRenderer) && !(r is SkinnedMeshRenderer)) continue;

                var src = r.sharedMaterials;
                if (src == null || src.Length == 0) continue;

                var dst = new Material[src.Length];
                bool changed = false;
                for (int i = 0; i < src.Length; i++)
                {
                    Material m = src[i];
                    if (m == null) { dst[i] = null; continue; }
                    if (m.shader != null &&
                        m.shader.name.StartsWith("Universal Render Pipeline"))
                    {
                        dst[i] = m; // already URP — keep it (and keep batching)
                        continue;
                    }
                    dst[i] = Convert(m, lit);
                    changed = true;
                }
                if (changed) r.sharedMaterials = dst;
            }
        }

        private static Material Convert(Material src, Shader lit)
        {
            if (Cache.TryGetValue(src, out var cached) && cached != null) return cached;

            // Read albedo + colour BEFORE swapping (Standard uses _MainTex/_Color).
            Texture tex = null;
            Vector2 scale = Vector2.one, offset = Vector2.zero;
            if (src.HasProperty("_MainTex") && src.GetTexture("_MainTex") != null)
            {
                tex = src.GetTexture("_MainTex");
                scale = src.GetTextureScale("_MainTex");
                offset = src.GetTextureOffset("_MainTex");
            }
            else if (src.HasProperty("_BaseMap") && src.GetTexture("_BaseMap") != null)
            {
                tex = src.GetTexture("_BaseMap");
                scale = src.GetTextureScale("_BaseMap");
                offset = src.GetTextureOffset("_BaseMap");
            }

            Color color = src.HasProperty("_Color") ? src.GetColor("_Color")
                : (src.HasProperty("_BaseColor") ? src.GetColor("_BaseColor") : Color.white);

            var conv = new Material(lit) { name = src.name + " (URP)" };
            if (tex != null)
            {
                conv.SetTexture("_BaseMap", tex);
                conv.SetTextureScale("_BaseMap", scale);
                conv.SetTextureOffset("_BaseMap", offset);
                if (conv.HasProperty("_MainTex")) conv.SetTexture("_MainTex", tex);
            }
            if (conv.HasProperty("_BaseColor")) conv.SetColor("_BaseColor", color);

            Cache[src] = conv;
            return conv;
        }
    }
}
