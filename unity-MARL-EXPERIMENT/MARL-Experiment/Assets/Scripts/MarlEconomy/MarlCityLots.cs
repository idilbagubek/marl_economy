using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // The firm "business district": a fixed grid of lots east of the town where
    // dynamic firms spawn/despawn without ever touching the static city. Lot i
    // fills column-by-column (6 rows per column); columns face each other across
    // a central avenue at x = 31. Also hosts the footprint-fitting helpers that
    // keep the wildly differently sized Toon City prefabs inside their plots.
    public static class MarlCityLots
    {
        public const int RowsPerColumn = 6;
        public const float ColumnPitch = 9f;
        public const float RowPitch = 8.4f;
        public const float OriginX = 26.5f;   // first column
        public const float OriginZ = -23f;    // first row
        public const float FirmFootprint = 6f; // max building extent inside a lot

        public static int ColumnOf(int lot) => lot / RowsPerColumn;
        public static int RowOf(int lot) => lot % RowsPerColumn;

        public static Vector3 LotPosition(int lot) => new Vector3(
            OriginX + ColumnOf(lot) * ColumnPitch, 0f, OriginZ + RowOf(lot) * RowPitch);

        // Buildings face the central avenue: even columns look +x, odd columns -x.
        public static float LotYaw(int lot) => ColumnOf(lot) % 2 == 0 ? 90f : -90f;
        public static Vector3 LotFront(int lot) => ColumnOf(lot) % 2 == 0 ? Vector3.right : Vector3.left;

        // ── prefab fitting ───────────────────────────────────────────────────

        public static Bounds MeasureBounds(GameObject go)
        {
            var rs = go.GetComponentsInChildren<Renderer>(true);
            if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.one);
            Bounds b = rs[0].bounds;
            for (int i = 1; i < rs.Length; i++) b.Encapsulate(rs[i].bounds);
            return b;
        }

        public static float MeasureHeight(GameObject go) => MeasureBounds(go).size.y;

        // Uniformly rescales the instance so its ground footprint (max of x/z
        // extents) equals targetFootprint. Returns the applied factor.
        public static float NormalizeFootprint(GameObject go, float targetFootprint)
        {
            Bounds b = MeasureBounds(go);
            float footprint = Mathf.Max(b.size.x, b.size.z);
            if (footprint < 0.001f) return 1f;
            float k = targetFootprint / footprint;
            go.transform.localScale *= k;
            return k;
        }
    }
}
