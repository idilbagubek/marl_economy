using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // One household: drives the rigged character's Animator by clip name (there is
    // no Speed blend parameter in the CityPeople controllers), faces its direction
    // of travel, and shows a status orb encoding wealth / hunger / fatigue / illness.
    public class MarlHouseholdAgent : MonoBehaviour
    {
        private Animator _animator;
        private string _walkState, _idleState, _runState;
        private string _currentState;

        private Renderer _orb;
        private Material _orbMat;
        private TextMesh _actionTag;

        private Vector3 _lastPos;
        private float _baseScale = 1f;
        private bool _initialised;

        // Latest pushed week data.
        private string _action = "rest";
        private bool _isDead;
        private float _pulse;

        public void Init(MarlEconomyPrefabConfig config)
        {
            _baseScale = config != null ? config.householdScale : 1f;

            // Stop the asset's random animation shuffler (decoupled by type name so
            // the project still compiles if the CityPeople pack is removed).
            foreach (var mb in GetComponentsInChildren<MonoBehaviour>())
                if (mb != null && mb.GetType().Name == "CityPeople") Destroy(mb);

            _animator = GetComponentInChildren<Animator>();
            ResolveClips();

            // Status orb above the head.
            var orb = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            orb.name = "StatusOrb";
            var col = orb.GetComponent<Collider>();
            if (col != null) Destroy(col);
            orb.transform.SetParent(transform, false);
            orb.transform.localPosition = new Vector3(0f, 2.15f, 0f);
            orb.transform.localScale = Vector3.one * 0.34f;
            _orb = orb.GetComponent<Renderer>();
            _orbMat = MarlVisualPalette.NewUnlit(Color.white);
            _orb.sharedMaterial = _orbMat;

            // Floating action tag so each agent's chosen action is readable.
            var tag = new GameObject("ActionTag");
            tag.transform.SetParent(transform, false);
            tag.transform.localPosition = new Vector3(0f, 2.62f, 0f);
            _actionTag = tag.AddComponent<TextMesh>();
            _actionTag.anchor = TextAnchor.MiddleCenter;
            _actionTag.alignment = TextAlignment.Center;
            _actionTag.characterSize = 0.1f;
            _actionTag.fontSize = 48;
            _actionTag.color = Color.white;
            tag.AddComponent<MarlBillboard>();

            _lastPos = transform.position;
            _initialised = true;
        }

        private void ResolveClips()
        {
            if (_animator == null || _animator.runtimeAnimatorController == null) return;
            foreach (var clip in _animator.runtimeAnimatorController.animationClips)
            {
                if (clip == null) continue;
                string n = clip.name.ToLowerInvariant();
                if (_walkState == null && (n.Contains("basicwalk") || n.Contains("slowwalk") || n.Contains("walk")))
                    _walkState = clip.name;
                if (_runState == null && (n.Contains("running") || n.Contains("jog")))
                    _runState = clip.name;
                if (_idleState == null && n.Contains("idle"))
                    _idleState = clip.name;
            }
            if (_runState == null) _runState = _walkState;
            if (_walkState == null) _walkState = _idleState;
        }

        public void ApplyWeek(HouseholdSnapshot h)
        {
            if (!_initialised) return;
            _action = h.action_label ?? "rest";
            _isDead = h.health < 0.02f;

            // Body scale by health.
            float healthScale = Mathf.Lerp(0.72f, 1.12f, Mathf.Clamp01(h.health));
            if (_isDead) healthScale = 0.5f;
            transform.localScale = Vector3.one * (_baseScale * healthScale);

            // Status orb colour: wealth base, shifted by hunger/fatigue/illness.
            Color c = MarlVisualPalette.Wealth(h.wealth_class);
            float hungerT = Mathf.InverseLerp(0.5f, 0.95f, h.hunger);
            if (hungerT > 0f) c = Color.Lerp(c, MarlVisualPalette.Hunger, hungerT);
            float fatigueT = Mathf.InverseLerp(0.55f, 0.95f, h.fatigue);
            if (fatigueT > 0f) c = Color.Lerp(c, MarlVisualPalette.Fatigue, fatigueT * 0.6f);
            Color ill = MarlVisualPalette.Illness(h.illness);
            if (ill.a > 0f) c = Color.Lerp(c, ill, 0.7f);
            if (_action == "steal") c = MarlVisualPalette.Steal;
            if (_isDead) c = new Color(0.35f, 0.35f, 0.38f);
            SetOrb(c);

            if (_actionTag != null)
            {
                _actionTag.text  = _isDead ? "DEAD" : MarlVisualPalette.ActionLabel(_action);
                _actionTag.color = _isDead ? new Color(0.55f, 0.55f, 0.58f) : MarlVisualPalette.Action(_action);
            }
        }

        private void SetOrb(Color c)
        {
            if (_orbMat == null) return;
            if (_orbMat.HasProperty("_BaseColor")) _orbMat.SetColor("_BaseColor", c);
            _orbMat.color = c;
        }

        private void Update()
        {
            if (!_initialised) return;

            float dt = Mathf.Max(Time.deltaTime, 1e-4f);
            Vector3 delta = transform.position - _lastPos;
            delta.y = 0f;
            float speed = delta.magnitude / dt;
            _lastPos = transform.position;

            // Face travel direction.
            if (speed > 0.05f)
            {
                Quaternion target = Quaternion.LookRotation(delta.normalized, Vector3.up);
                transform.rotation = Quaternion.Slerp(transform.rotation, target, 12f * dt);
            }

            // Pick animation state from movement.
            string desired = _isDead ? _idleState
                : speed > 2.5f ? _runState
                : speed > 0.15f ? _walkState
                : _idleState;
            PlayState(desired);

            // Pulse the orb for thieves and the dying so they catch the eye.
            if ((_action == "steal" || _isDead) && _orb != null)
            {
                _pulse += dt * 6f;
                float s = 0.34f * (1f + 0.35f * Mathf.Sin(_pulse));
                _orb.transform.localScale = Vector3.one * s;
            }
        }

        private void PlayState(string state)
        {
            if (_animator == null || string.IsNullOrEmpty(state) || state == _currentState) return;
            _currentState = state;
            _animator.CrossFadeInFixedTime(state, 0.2f);
        }
    }
}
