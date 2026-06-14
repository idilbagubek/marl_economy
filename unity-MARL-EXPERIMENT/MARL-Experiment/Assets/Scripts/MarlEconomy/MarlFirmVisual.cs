using System.Collections;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Drives one firm building: the model is first normalized so its ground
    // footprint fits a business-district lot (Toon City prefabs range from a
    // tiny coffee shop to a huge shopping center), then capacity modulates the
    // size within the lot. A colour beacon (red->amber->green) tracks cash
    // without recolouring the building art.
    public class MarlFirmVisual : MonoBehaviour
    {
        private Transform _model;
        private Renderer _beacon;
        private Material _beaconMat;
        private float _baseScale = 1f;
        private float _normScale = 1f;   // scale that makes the footprint == lot target
        private float _rawHeight = 4f;   // model height at scale 1

        // Fire/wet overlays toggled by the chaos director.
        public Transform model => _model;

        public void Init(GameObject modelInstance, float baseScale)
        {
            _baseScale = baseScale;
            if (modelInstance != null)
            {
                Bounds b = MarlCityLots.MeasureBounds(modelInstance);
                float footprint = Mathf.Max(b.size.x, b.size.z);
                _normScale = footprint > 0.001f ? MarlCityLots.FirmFootprint / footprint : 1f;
                _rawHeight = Mathf.Max(0.5f, b.size.y);

                _model = modelInstance.transform;
                _model.SetParent(transform, false);
                _model.localPosition = Vector3.zero;
            }

            // Beacon orb floating above the building.
            var orb = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            orb.name = "CashBeacon";
            var col = orb.GetComponent<Collider>();
            if (col != null) Destroy(col);
            orb.transform.SetParent(transform, false);
            orb.transform.localPosition = new Vector3(0f, 5.0f, 0f);
            orb.transform.localScale = Vector3.one * 0.7f;
            _beacon = orb.GetComponent<Renderer>();
            _beaconMat = MarlVisualPalette.NewUnlit(Color.gray);
            _beacon.sharedMaterial = _beaconMat;
            orb.AddComponent<MarlBillboard>();
        }

        public void Apply(FirmSnapshot firm)
        {
            // Capacity -> modest size variation INSIDE the lot (the normalized
            // footprint already guarantees the building can't spill over).
            float cap = Mathf.InverseLerp(1f, 12f, firm.capacity);
            float s = _normScale * Mathf.Lerp(0.82f, 1.15f, cap) * _baseScale;
            float sy = s * 1.12f; // slightly taller than wide for legibility
            if (_model != null)
                _model.localScale = new Vector3(s, sy, s);

            float beaconY = _rawHeight * sy + 1.2f;
            if (_beacon != null)
            {
                _beacon.transform.localPosition = new Vector3(0f, beaconY, 0f);
                Color c = firm.spawned_this_step ? new Color(0.33f, 0.95f, 0.5f) : MarlVisualPalette.FirmCash(firm.cash);
                if (_beaconMat.HasProperty("_BaseColor")) _beaconMat.SetColor("_BaseColor", c);
                _beaconMat.color = c;
            }
        }

        // A firm that left the snapshots has closed. Make the despawn legible:
        // raise a BANKRUPT sign, then sink + flatten the building into the lot
        // before destroying it (instead of blinking out of existence).
        public void Collapse()
        {
            if (_beacon != null) _beacon.gameObject.SetActive(false);
            StartCoroutine(CollapseRoutine());
        }

        private IEnumerator CollapseRoutine()
        {
            var sign = new GameObject("BANKRUPT");
            sign.transform.SetParent(transform, false);
            sign.transform.localPosition = new Vector3(0f, Mathf.Max(2.5f, _rawHeight * 0.6f), 0f);
            var tm = sign.AddComponent<TextMesh>();
            tm.text = "BANKRUPT";
            tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center;
            tm.characterSize = 0.2f;
            tm.fontSize = 56;
            tm.color = new Color(0.96f, 0.32f, 0.15f);
            sign.AddComponent<MarlBillboard>();

            Vector3 s0 = _model != null ? _model.localScale : Vector3.one;
            Vector3 basePos = transform.position;
            const float dur = 1.3f;
            float t = 0f;
            while (t < dur)
            {
                t += Time.deltaTime;
                float k = Mathf.Clamp01(t / dur);
                if (_model != null)
                    _model.localScale = new Vector3(
                        Mathf.Lerp(s0.x, s0.x * 0.85f, k),
                        Mathf.Lerp(s0.y, s0.y * 0.04f, k * k), // flatten downward
                        Mathf.Lerp(s0.z, s0.z * 0.85f, k));
                transform.position = basePos + Vector3.down * (k * 1.6f); // sink into the lot
                sign.transform.localPosition += Vector3.up * (Time.deltaTime * 0.7f);
                tm.color = new Color(0.96f, 0.32f, 0.15f, 1f - k); // rise + fade the sign
                yield return null;
            }
            Destroy(gameObject);
        }
    }
}
