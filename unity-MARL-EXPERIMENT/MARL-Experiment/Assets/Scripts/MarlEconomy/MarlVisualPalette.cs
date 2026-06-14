using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Single source of truth for the colour language of the visualisation, so
    // pips, beacons, banners, FX and chaos overlays all read consistently.
    public static class MarlVisualPalette
    {
        public static Color Wealth(string wealthClass)
        {
            switch (wealthClass)
            {
                case "poor":   return new Color(0.25f, 0.55f, 1.00f);
                case "middle": return new Color(0.25f, 0.85f, 0.55f);
                case "rich":   return new Color(1.00f, 0.78f, 0.25f);
                case "elite":  return new Color(0.95f, 0.35f, 0.85f);
                default:       return new Color(0.85f, 0.85f, 0.88f);
            }
        }

        // Firm cash health: red (insolvent) -> amber -> green (healthy).
        public static Color FirmCash(float cash)
        {
            float t = Mathf.Clamp01(cash / 20f);
            Color low  = new Color(0.85f, 0.18f, 0.16f);
            Color mid  = new Color(0.96f, 0.74f, 0.18f);
            Color high = new Color(0.30f, 0.85f, 0.42f);
            return t < 0.5f ? Color.Lerp(low, mid, t * 2f) : Color.Lerp(mid, high, (t - 0.5f) * 2f);
        }

        public static readonly Color Hunger   = new Color(0.85f, 0.78f, 0.35f); // sallow
        public static readonly Color Fatigue  = new Color(0.45f, 0.45f, 0.55f); // gray
        public static readonly Color Steal    = new Color(0.90f, 0.12f, 0.12f);

        public static Color Illness(int illness)
        {
            switch (illness)
            {
                case 2:  return new Color(0.90f, 0.50f, 0.50f); // mild
                case 3:  return new Color(0.95f, 0.18f, 0.18f); // severe
                case 4:  return new Color(0.55f, 0.85f, 0.95f); // recovering
                default: return Color.clear;
            }
        }

        // Action colour language — shared by the per-agent action tag and the
        // destination buildings, so "this agent is working" and "this is the work
        // building" read in the same colour.
        public static Color Action(string action)
        {
            switch (action)
            {
                case "work":   return new Color(0.30f, 0.62f, 1.00f); // blue
                case "shop":   return new Color(0.30f, 0.85f, 0.55f); // green
                case "rest":   return new Color(0.45f, 0.85f, 0.95f); // cyan
                case "steal":  return new Color(0.92f, 0.18f, 0.18f); // red
                case "clinic": return new Color(0.95f, 0.45f, 0.55f);
                case "invest": return new Color(0.96f, 0.80f, 0.25f);
                default:       return new Color(0.85f, 0.85f, 0.88f);
            }
        }

        public static string ActionLabel(string action)
        {
            switch (action)
            {
                case "work":   return "WORK";
                case "shop":   return "SHOP";
                case "rest":   return "REST";
                case "steal":  return "STEAL";
                case "clinic": return "CLINIC";
                case "invest": return "INVEST";
                default:       return string.IsNullOrEmpty(action) ? "" : action.ToUpperInvariant();
            }
        }

        public static void Disaster(DisasterKind kind, out Color a, out Color b)
        {
            switch (kind)
            {
                case DisasterKind.Fire:         a = new Color(1.00f, 0.42f, 0.08f); b = new Color(0.90f, 0.15f, 0.05f); break;
                case DisasterKind.Flood:        a = new Color(0.22f, 0.52f, 1.00f); b = new Color(0.08f, 0.20f, 0.65f); break;
                case DisasterKind.Earthquake:   a = new Color(0.78f, 0.62f, 0.34f); b = new Color(0.45f, 0.34f, 0.16f); break;
                case DisasterKind.SupplyShock:  a = new Color(1.00f, 0.85f, 0.18f); b = new Color(0.80f, 0.50f, 0.08f); break;
                case DisasterKind.EpidemicWave: a = new Color(0.55f, 0.95f, 0.28f); b = new Color(0.25f, 0.60f, 0.10f); break;
                default:                        a = new Color(1.00f, 0.40f, 0.40f); b = new Color(0.65f, 0.10f, 0.10f); break;
            }
        }

        public static string DisasterLabel(DisasterKind kind)
        {
            switch (kind)
            {
                case DisasterKind.Fire: return "FIRE";
                case DisasterKind.Flood: return "FLOOD";
                case DisasterKind.Earthquake: return "EARTHQUAKE";
                case DisasterKind.SupplyShock: return "SUPPLY SHOCK";
                case DisasterKind.EpidemicWave: return "EPIDEMIC WAVE";
                default: return "DISASTER";
            }
        }

        // Creates an unlit, fully self-coloured material that survives URP without
        // a configured shader graph (used for pips, beacons, water, FX fallback).
        public static Material NewUnlit(Color color)
        {
            Shader s = Shader.Find("Universal Render Pipeline/Unlit");
            if (s == null) s = Shader.Find("Unlit/Color");
            if (s == null) s = Shader.Find("Sprites/Default");
            var m = new Material(s);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            if (m.HasProperty("_Color")) m.SetColor("_Color", color);
            m.color = color;
            return m;
        }

        // Tints a renderer, but FIRST guarantees a URP-compatible material.
        // GameObject.CreatePrimitive hands back Unity's Built-in "Default-Material"
        // (Standard shader), which renders flat MAGENTA under URP — tinting it does
        // nothing. So if the renderer isn't already on a URP shader we swap in a
        // fresh URP/Lit instance, then colour it. (Instance, never the asset.)
        public static void SetColor(Renderer r, Color color)
        {
            if (r == null) return;
            var m = r.material; // instance
            if (m == null || m.shader == null ||
                !m.shader.name.StartsWith("Universal Render Pipeline"))
            {
                Shader lit = Shader.Find("Universal Render Pipeline/Lit");
                if (lit == null) lit = Shader.Find("Universal Render Pipeline/Unlit");
                if (lit != null)
                {
                    m = new Material(lit);
                    r.material = m;
                }
            }
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            if (m.HasProperty("_Color")) m.SetColor("_Color", color);
            m.color = color;
        }
    }
}
