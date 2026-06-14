using UnityEditor;
using UnityEngine;

namespace MarlEconomy.UnityViz.EditorTools
{
    // Some imported materials end up on a shader URP can't render, so they show
    // up flat MAGENTA ("unrendered"): the URP *2D* Mesh-Lit shader (wrong for 3D
    // meshes), a Built-in Standard/Legacy/Mobile shader, or a *missing* shader.
    // This reassigns every such material to the proper 3D "Universal Render
    // Pipeline/Lit" shader and remaps _MainTex/_Color -> _BaseMap/_BaseColor so
    // nothing is lost. Materials already on a URP shader are left untouched.
    public static class MarlMaterialFixer
    {
        [MenuItem("MARL Economy/Fix Magenta Materials (→ URP Lit)")]
        public static void FixMaterials()
        {
            Shader lit = Shader.Find("Universal Render Pipeline/Lit");
            if (lit == null)
            {
                Debug.LogError("[MARL] 'Universal Render Pipeline/Lit' shader not found — is URP installed?");
                return;
            }

            int changed = 0, skipped = 0;
            foreach (string guid in AssetDatabase.FindAssets("t:Material"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (mat == null) continue;

                // Leave materials that already use any URP shader; convert the rest
                // (URP 2D, Built-in Standard/Legacy/Mobile, or a null/missing shader).
                bool isUrp = mat.shader != null &&
                             mat.shader.name.StartsWith("Universal Render Pipeline");
                if (isUrp) { skipped++; continue; }

                // Capture albedo texture (+ tiling/offset), base colour, emission
                // BEFORE swapping the shader.
                Texture tex = null; Vector2 scale = Vector2.one, offset = Vector2.zero;
                if (mat.HasProperty("_MainTex") && mat.GetTexture("_MainTex") != null)
                {
                    tex = mat.GetTexture("_MainTex");
                    scale = mat.GetTextureScale("_MainTex");
                    offset = mat.GetTextureOffset("_MainTex");
                }
                else if (mat.HasProperty("_BaseMap") && mat.GetTexture("_BaseMap") != null)
                {
                    tex = mat.GetTexture("_BaseMap");
                    scale = mat.GetTextureScale("_BaseMap");
                    offset = mat.GetTextureOffset("_BaseMap");
                }

                Color baseColor = mat.HasProperty("_Color") ? mat.GetColor("_Color")
                    : (mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") : Color.white);
                Color emission = mat.HasProperty("_EmissionColor") ? mat.GetColor("_EmissionColor") : Color.black;

                mat.shader = lit;

                if (tex != null)
                {
                    mat.SetTexture("_BaseMap", tex);
                    mat.SetTextureScale("_BaseMap", scale);
                    mat.SetTextureOffset("_BaseMap", offset);
                    if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", tex);
                }
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", baseColor);
                if (mat.HasProperty("_EmissionColor"))
                {
                    mat.SetColor("_EmissionColor", emission);
                    if (emission.maxColorComponent > 0.001f) mat.EnableKeyword("_EMISSION");
                }

                EditorUtility.SetDirty(mat);
                changed++;
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[MARL] Reassigned {changed} non-URP material(s) to URP/Lit ({skipped} already URP).");
        }

        // The toon window panes are modeled COPLANAR with the walls and were
        // authored as transparent glass; the URP upgrade left them opaque, so
        // they z-fight the wall behind them and read as broken/unrendered.
        // This turns every window/glass material into real transparent glass
        // (the same restyle MarlWindowGlass applies at runtime) and saves it.
        [MenuItem("MARL Economy/Style Window Materials (glass)")]
        public static void StyleWindows()
        {
            int styled = 0;
            foreach (string guid in AssetDatabase.FindAssets("t:Material"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                string lower = path.ToLowerInvariant();
                if (lower.Contains("sill")) continue;
                if (!lower.Contains("window") && !lower.Contains("glass")) continue;

                var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (mat == null) continue;
                MarlWindowGlass.Restyle(mat);
                EditorUtility.SetDirty(mat);
                styled++;
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[MARL] Styled {styled} window material(s) as transparent glass (no more z-fighting).");
        }
    }
}
