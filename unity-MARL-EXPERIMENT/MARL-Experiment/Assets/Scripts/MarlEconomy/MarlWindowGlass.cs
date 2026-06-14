using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Window/glass submeshes on the toon buildings must read as solid, lit glass
    // panes — never see-through holes, never flickering. Restyle() gives them a
    // bright emissive glass-blue and draws them in the TRANSPARENT queue (after
    // every opaque wall) with depth-write ON: being strictly later than the walls
    // means a coplanar pane can never z-fight, while alpha 1 + ZWrite keep the
    // pane looking solid. Used by the editor menu "MARL Economy/Style Window
    // Materials (glass)" for a persistent fix and as a runtime safety net on every
    // building the visualization instantiates.
    public static class MarlWindowGlass
    {
        // Alpha 1 → a solid pane, not a hole. The transparent queue + ZWrite are
        // only there to dodge coplanar z-fighting, not to make the glass see-through.
        public static readonly Color GlassColor = new Color(0.52f, 0.78f, 0.98f, 1f);
        public static readonly Color GlassGlow  = new Color(0.16f, 0.34f, 0.55f, 1f);

        private static readonly HashSet<Material> _done = new HashSet<Material>();

        public static bool LooksLikeWindow(string materialName)
        {
            if (string.IsNullOrEmpty(materialName)) return false;
            string n = materialName.ToLowerInvariant();
            if (n.Contains("sill")) return false;
            return n.Contains("window") || n.Contains("glass");
        }

        // Fix every window/glass shared material under root (idempotent).
        public static void RestyleUnder(GameObject root)
        {
            if (root == null) return;
            foreach (var r in root.GetComponentsInChildren<Renderer>(true))
            {
                var mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++)
                {
                    Material m = mats[i];
                    if (m == null || !LooksLikeWindow(m.name)) continue;
                    if (!_done.Add(m)) continue;
                    Restyle(m);
                }
            }
        }

        public static void Restyle(Material m)
        {
            if (m == null) return;

            if (m.shader == null || !m.shader.name.Contains("Universal Render Pipeline/Lit"))
            {
                Shader lit = Shader.Find("Universal Render Pipeline/Lit");
                if (lit != null) m.shader = lit;
            }

            // Alpha-blend surface, but drawn in the TRANSPARENT queue (after every
            // opaque wall) with ZWrite ON and an opaque colour — so a coplanar pane
            // can never z-fight, yet it still reads as a solid, lit glass pane and
            // not a see-through hole.
            m.SetFloat("_Surface", 1f);
            m.SetFloat("_Blend", 0f); // alpha
            m.SetOverrideTag("RenderType", "Transparent");
            if (m.HasProperty("_SrcBlend")) m.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (m.HasProperty("_DstBlend")) m.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (m.HasProperty("_SrcBlendAlpha")) m.SetFloat("_SrcBlendAlpha", (float)UnityEngine.Rendering.BlendMode.One);
            if (m.HasProperty("_DstBlendAlpha")) m.SetFloat("_DstBlendAlpha", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (m.HasProperty("_ZWrite")) m.SetFloat("_ZWrite", 1f); // solid: write depth, occlude
            if (m.HasProperty("_AlphaClip")) m.SetFloat("_AlphaClip", 0f);
            m.DisableKeyword("_ALPHATEST_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            // ZWrite is on, so let the panes rejoin the depth + shadow passes.
            m.SetShaderPassEnabled("DepthOnly", true);
            m.SetShaderPassEnabled("ShadowCaster", true);

            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f); // double-sided → always visible
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", GlassColor);
            if (m.HasProperty("_Color")) m.SetColor("_Color", GlassColor);
            if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", 0.9f);
            if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", 0f);
            if (m.HasProperty("_EmissionColor"))
            {
                m.SetColor("_EmissionColor", GlassGlow);
                m.EnableKeyword("_EMISSION");
                m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
            }
        }
    }
}
