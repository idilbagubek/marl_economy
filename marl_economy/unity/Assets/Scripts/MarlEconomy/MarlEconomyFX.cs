using UnityEngine;

namespace MarlEconomy.UnityViz
{
    public class MarlEconomyFX : MonoBehaviour
    {
        public void PlayFirmSpawn(Vector3 position)
        {
            Emit(position, new FXConfig
            {
                label = "FirmSpawn",
                count = 45,
                lifetime = 2.0f,
                speed = 5f,
                size = 0.28f,
                radius = 0.9f,
                colorStart = new Color(0.35f, 1.0f, 0.50f),
                colorEnd = new Color(0.80f, 1.0f, 0.82f),
                velocityY = 1.8f
            });
        }

        public void PlayFirmExit(Vector3 position)
        {
            Emit(position, new FXConfig
            {
                label = "FirmExit",
                count = 28,
                lifetime = 2.8f,
                speed = 2.5f,
                size = 0.45f,
                radius = 1.1f,
                colorStart = new Color(0.60f, 0.58f, 0.62f),
                colorEnd = new Color(0.22f, 0.22f, 0.25f),
                velocityY = 0.6f
            });
        }

        public void PlayDisaster(Vector3 position, string disasterType)
        {
            Color a, b;
            switch (disasterType)
            {
                case "fire":          a = new Color(1.00f, 0.42f, 0.08f); b = new Color(0.90f, 0.15f, 0.05f); break;
                case "flood":         a = new Color(0.22f, 0.52f, 1.00f); b = new Color(0.08f, 0.20f, 0.65f); break;
                case "earthquake":    a = new Color(0.88f, 0.72f, 0.28f); b = new Color(0.55f, 0.40f, 0.10f); break;
                case "supply_shock":  a = new Color(1.00f, 0.88f, 0.18f); b = new Color(0.80f, 0.50f, 0.08f); break;
                case "epidemic_wave": a = new Color(0.55f, 0.95f, 0.28f); b = new Color(0.25f, 0.60f, 0.10f); break;
                default:              a = new Color(1.00f, 0.40f, 0.40f); b = new Color(0.65f, 0.10f, 0.10f); break;
            }

            Emit(position, new FXConfig
            {
                label = "Disaster_" + disasterType,
                count = 90,
                lifetime = 4.0f,
                speed = 7f,
                size = 0.55f,
                radius = 9f,
                colorStart = a,
                colorEnd = b,
                velocityY = 2.5f
            });
        }

        public void PlayCrime(Vector3 position)
        {
            Emit(position, new FXConfig
            {
                label = "Crime",
                count = 22,
                lifetime = 1.8f,
                speed = 3.5f,
                size = 0.22f,
                radius = 0.5f,
                colorStart = new Color(0.90f, 0.12f, 0.12f),
                colorEnd = new Color(0.30f, 0.04f, 0.04f),
                velocityY = 1.5f
            });
        }

        public void PlayHouseholdDeath(Vector3 position)
        {
            Emit(position, new FXConfig
            {
                label = "Death",
                count = 14,
                lifetime = 1.6f,
                speed = 2.2f,
                size = 0.16f,
                radius = 0.35f,
                colorStart = new Color(0.92f, 0.92f, 0.96f),
                colorEnd = new Color(0.45f, 0.45f, 0.50f),
                velocityY = 2.4f
            });
        }

        private struct FXConfig
        {
            public string label;
            public int count;
            public float lifetime;
            public float speed;
            public float size;
            public float radius;
            public Color colorStart;
            public Color colorEnd;
            public float velocityY;
        }

        private void Emit(Vector3 position, FXConfig cfg)
        {
            var go = new GameObject("FX_" + cfg.label);
            go.transform.position = position;

            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            var main = ps.main;
            main.duration = 0.4f;
            main.loop = false;
            main.startLifetime = cfg.lifetime;
            main.startSpeed = cfg.speed;
            main.startSize = cfg.size;
            main.startColor = cfg.colorStart;
            main.maxParticles = cfg.count;
            var emission = ps.emission;
            emission.rateOverTime = 0;
            emission.SetBursts(new[] { new ParticleSystem.Burst(0f, cfg.count) });

            var shape = ps.shape;
            shape.enabled = true;
            shape.radius = cfg.radius;

            var col = ps.colorOverLifetime;
            col.enabled = true;
            var grad = new Gradient();
            grad.SetKeys(
                new[] { new GradientColorKey(cfg.colorStart, 0f), new GradientColorKey(cfg.colorEnd, 1f) },
                new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0f, 1f) }
            );
            col.color = grad;

            if (cfg.velocityY != 0f)
            {
                var vel = ps.velocityOverLifetime;
                vel.enabled = true;
                vel.y = cfg.velocityY;
            }

            ps.Play();
            Destroy(go, cfg.lifetime + 1f);
        }
    }
}
