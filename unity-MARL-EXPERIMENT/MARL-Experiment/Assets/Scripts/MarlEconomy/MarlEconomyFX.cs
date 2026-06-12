using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Particle bursts for events. Uses a URP-compatible particle material so the
    // bursts aren't rendered as magenta error shaders.
    public class MarlEconomyFX : MonoBehaviour
    {
        public Material particleMaterial;

        private Material ParticleMat()
        {
            if (particleMaterial != null) return particleMaterial;
            Shader s = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (s == null) s = Shader.Find("Universal Render Pipeline/Unlit");
            if (s == null) s = Shader.Find("Sprites/Default");
            particleMaterial = new Material(s);
            return particleMaterial;
        }

        public void PlayFirmSpawn(Vector3 p) => Emit(p, "FirmSpawn", 45, 2.0f, 5f, 0.28f, 0.9f,
            new Color(0.35f, 1f, 0.50f), new Color(0.80f, 1f, 0.82f), 1.8f);

        public void PlayFirmExit(Vector3 p) => Emit(p, "FirmExit", 28, 2.8f, 2.5f, 0.45f, 1.1f,
            new Color(0.60f, 0.58f, 0.62f), new Color(0.22f, 0.22f, 0.25f), 0.6f);

        public void PlayCrime(Vector3 p) => Emit(p, "Crime", 18, 1.4f, 3.5f, 0.20f, 0.45f,
            new Color(0.90f, 0.12f, 0.12f), new Color(0.30f, 0.04f, 0.04f), 1.5f);

        public void PlayHouseholdDeath(Vector3 p) => Emit(p, "Death", 16, 1.8f, 2.2f, 0.18f, 0.4f,
            new Color(0.92f, 0.92f, 0.96f), new Color(0.45f, 0.45f, 0.50f), 2.6f);

        public void PlayDisaster(Vector3 p, DisasterKind kind)
        {
            MarlVisualPalette.Disaster(kind, out Color a, out Color b);
            Emit(p, "Disaster_" + kind, 90, 4.0f, 7f, 0.55f, 9f, a, b, 2.5f);
        }

        public ParticleSystem PlayLooping(Vector3 p, string label, Color a, Color b,
            int rate, float lifetime, float speed, float size, float radius, float velY)
        {
            var go = new GameObject("FXLoop_" + label);
            go.transform.position = p;
            var ps = Build(go, rate, lifetime, speed, size, radius, a, b, velY, true);
            ps.Play();
            return ps;
        }

        private void Emit(Vector3 p, string label, int count, float lifetime, float speed,
            float size, float radius, Color a, Color b, float velY)
        {
            var go = new GameObject("FX_" + label);
            go.transform.position = p;
            var ps = Build(go, count, lifetime, speed, size, radius, a, b, velY, false);
            ps.Play();
            Destroy(go, lifetime + 1f);
        }

        private ParticleSystem Build(GameObject go, float countOrRate, float lifetime, float speed,
            float size, float radius, Color a, Color b, float velY, bool looping)
        {
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            var main = ps.main;
            main.loop = looping;
            main.duration = looping ? 1.0f : 0.4f;
            main.startLifetime = lifetime;
            main.startSpeed = speed;
            main.startSize = size;
            main.startColor = a;
            main.maxParticles = looping ? 600 : (int)countOrRate;

            var emission = ps.emission;
            if (looping) emission.rateOverTime = countOrRate;
            else { emission.rateOverTime = 0; emission.SetBursts(new[] { new ParticleSystem.Burst(0f, (short)countOrRate) }); }

            var shape = ps.shape;
            shape.enabled = true;
            shape.radius = radius;

            var col = ps.colorOverLifetime;
            col.enabled = true;
            var grad = new Gradient();
            grad.SetKeys(
                new[] { new GradientColorKey(a, 0f), new GradientColorKey(b, 1f) },
                new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0f, 1f) });
            col.color = grad;

            if (velY != 0f)
            {
                var vel = ps.velocityOverLifetime;
                vel.enabled = true;
                vel.y = velY;
            }

            var rend = ps.GetComponent<ParticleSystemRenderer>();
            if (rend != null) rend.material = ParticleMat();
            return ps;
        }
    }
}
