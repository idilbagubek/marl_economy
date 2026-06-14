using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Builds the static city: ground, district pads, roads, landmark buildings at
    // the fixed simulation anchors, residential dressing, greenery, and the
    // business-district plots where dynamic firms live. Every placed building is
    // footprint-normalized so nothing overlaps. Uses the prefab config when
    // present, falling back to primitives so it always renders.
    public static class MarlWorldBuilder
    {
        public static Transform Build(MarlEconomyPrefabConfig config)
        {
            var root = new GameObject("MARL City").transform;

            BuildGround(root);
            Pad(root, "Residential District", new Vector3(0f, 0.02f, -18f), new Vector3(48f, 0.04f, 20f), new Color(0.80f, 0.84f, 0.80f));
            Pad(root, "Market District",      new Vector3(0f, 0.03f, 2f),  new Vector3(34f, 0.04f, 26f), new Color(0.80f, 0.86f, 0.82f));
            Pad(root, "Civic District",       new Vector3(0f, 0.035f, 22f), new Vector3(48f, 0.04f, 14f), new Color(0.79f, 0.83f, 0.90f));

            Road(root, new Vector3(0f, 0.06f, 2f), new Vector3(2.4f, 0.05f, 60f));
            Road(root, new Vector3(4f, 0.065f, 2f), new Vector3(72f, 0.05f, 2.4f));   // main street, reaches the business district
            Road(root, new Vector3(0f, 0.07f, -14f), new Vector3(44f, 0.05f, 1.8f));

            BuildBusinessDistrict(root);

            // Landmarks at the simulation anchors (footprint keeps them apart).
            Landmark(root, config != null ? config.governmentPrefab : null, "GOVERNMENT",
                MarlEconomyPlayback.GovernmentPos, new Color(0.25f, 0.40f, 0.95f), config, 8.5f);
            Landmark(root, config != null ? config.centralBankPrefab : null, "CENTRAL BANK",
                MarlEconomyPlayback.CentralBankPos, new Color(0.95f, 0.75f, 0.25f), config, 8.5f);
            Landmark(root, config != null ? config.marketPrefab : null, "MARKET",
                MarlEconomyPlayback.MarketPos, new Color(0.35f, 0.80f, 0.55f), config, 9f);
            Landmark(root, config != null ? config.clinicPrefab : null, "CLINIC",
                MarlEconomyPlayback.ClinicPos, new Color(0.95f, 0.35f, 0.35f), config, 6.5f);
            Landmark(root, config != null ? config.bankPrefab : null, "INVESTMENT BANK",
                MarlEconomyPlayback.BankPos, new Color(0.95f, 0.85f, 0.35f), config, 6.5f);
            Landmark(root, PickRestApartment(config), "REST AREA",
                MarlEconomyPlayback.RestPos, MarlVisualPalette.Action("rest"), config, 9f);
            Landmark(root, config != null ? config.riskZonePrefab : null, "RISK ZONE (STEAL)",
                MarlEconomyPlayback.RiskZonePos, MarlVisualPalette.Action("steal"), config, 6f);

            BuildHomes(root, config);
            BuildGreenery(root, config);

            // Window panes ship opaque after the URP upgrade and z-fight the
            // walls; restyle everything just placed into transparent glass.
            MarlWindowGlass.RestyleUnder(root.gameObject);
            return root;
        }

        // The reserved firm zone east of town: a pad, an avenue down the middle,
        // and one visible plot slab per lot so vacant lots read as "for sale"
        // ground instead of accidental gaps.
        private static void BuildBusinessDistrict(Transform root)
        {
            Pad(root, "Business District", new Vector3(31f, 0.025f, -2f), new Vector3(18f, 0.04f, 50f), new Color(0.82f, 0.81f, 0.76f));
            Road(root, new Vector3(31f, 0.06f, -2f), new Vector3(2.4f, 0.05f, 50f)); // central avenue

            for (int lot = 0; lot < 12; lot++)
            {
                Vector3 p = MarlCityLots.LotPosition(lot);
                var slab = GameObject.CreatePrimitive(PrimitiveType.Cube);
                slab.name = $"Lot_{lot}";
                slab.transform.SetParent(root, false);
                slab.transform.position = new Vector3(p.x, 0.05f, p.z);
                slab.transform.localScale = new Vector3(7.4f, 0.03f, 7.4f);
                var col = slab.GetComponent<Collider>(); if (col != null) Object.Destroy(col);
                MarlVisualPalette.SetColor(slab.GetComponent<Renderer>(), new Color(0.70f, 0.70f, 0.66f));
            }

            var labelHolder = new GameObject("Business District Label");
            labelHolder.transform.SetParent(root, false);
            labelHolder.transform.position = new Vector3(31f, 0f, 25f);
            AddLabel(labelHolder, "BUSINESS DISTRICT  ·  WORK & SHOP", new Color(0.30f, 0.45f, 0.75f), 6.5f);
        }

        private static void BuildGround(Transform root)
        {
            var plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
            plane.name = "Ground";
            plane.transform.SetParent(root, false);
            plane.transform.localScale = new Vector3(8f, 1f, 8f); // 80m
            var r = plane.GetComponent<Renderer>();
            if (r != null) MarlVisualPalette.SetColor(r, new Color(0.62f, 0.66f, 0.60f));

            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.55f, 0.60f, 0.66f);
        }

        private static void Pad(Transform root, string name, Vector3 pos, Vector3 scale, Color color)
        {
            var pad = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pad.name = name;
            pad.transform.SetParent(root, false);
            pad.transform.position = pos;
            pad.transform.localScale = scale;
            var col = pad.GetComponent<Collider>(); if (col != null) Object.Destroy(col);
            MarlVisualPalette.SetColor(pad.GetComponent<Renderer>(), color);
        }

        private static void Road(Transform root, Vector3 pos, Vector3 scale)
        {
            var road = GameObject.CreatePrimitive(PrimitiveType.Cube);
            road.name = "Road";
            road.transform.SetParent(root, false);
            road.transform.position = pos;
            road.transform.localScale = scale;
            var col = road.GetComponent<Collider>(); if (col != null) Object.Destroy(col);
            MarlVisualPalette.SetColor(road.GetComponent<Renderer>(), new Color(0.22f, 0.23f, 0.25f));
        }

        private static void Landmark(Transform root, GameObject prefab, string label, Vector3 pos, Color color,
            MarlEconomyPrefabConfig config, float footprint)
        {
            GameObject obj = prefab != null ? Object.Instantiate(prefab) : Fallback(color);
            obj.name = label;
            obj.transform.SetParent(root, false);
            obj.transform.position = pos;
            if (prefab != null)
            {
                if (config != null) obj.transform.localScale = Vector3.one * config.landmarkScale;
                MarlCityLots.NormalizeFootprint(obj, footprint);
            }
            float labelY = Mathf.Max(4.5f, MarlCityLots.MeasureHeight(obj) + 1.6f);
            AddLabel(obj, label, color, labelY);
        }

        // Households rest at a single landmark, so it should read as a prominent
        // APARTMENT, not a little house/school. Pick the tallest apartment among the
        // configured home prefabs (cb-apartment-*). Chosen in code on purpose: the
        // restPrefab field in the .asset doesn't reliably stick when edited outside
        // Unity, whereas homePrefabs already references the apartments we want.
        private static GameObject PickRestApartment(MarlEconomyPrefabConfig config)
        {
            if (config == null || config.homePrefabs == null || config.homePrefabs.Count == 0)
                return config != null ? config.restPrefab : null;

            GameObject best = null;
            float bestScore = float.NegativeInfinity;
            foreach (var p in config.homePrefabs)
            {
                if (p == null) continue;
                bool isApartment = p.name.ToLowerInvariant().Contains("apartment");
                // Apartments win outright; ties (and the no-apartment case) break by height.
                float score = (isApartment ? 1000f : 0f) + MarlCityLots.MeasureHeight(p);
                if (score > bestScore) { bestScore = score; best = p; }
            }
            return best != null ? best : config.restPrefab;
        }

        private static GameObject Fallback(Color color)
        {
            var root = new GameObject("Landmark");
            var body = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            body.transform.SetParent(root.transform, false);
            body.transform.localPosition = new Vector3(0f, 1.4f, 0f);
            body.transform.localScale = new Vector3(2.4f, 1.4f, 2.4f);
            MarlVisualPalette.SetColor(body.GetComponent<Renderer>(), color);
            return root;
        }

        private static void AddLabel(GameObject parent, string text, Color color, float height)
        {
            var go = new GameObject(text + "_Label");
            go.transform.SetParent(parent.transform, false);
            go.transform.localPosition = new Vector3(0f, height, 0f);
            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center;
            tm.characterSize = 0.25f;
            tm.fontSize = 48;
            tm.color = Color.Lerp(color, Color.black, 0.35f);
            go.AddComponent<MarlBillboard>();
        }

        private static void BuildHomes(Transform root, MarlEconomyPrefabConfig config)
        {
            if (config == null || config.homePrefabs == null || config.homePrefabs.Count == 0) return;
            int i = 0;
            for (float z = -28f; z <= -24f; z += 4f)
            {
                for (float x = -22f; x <= 22f; x += 5.5f)
                {
                    var prefab = MarlEconomyPrefabConfig.Pick(config.homePrefabs, i++);
                    if (prefab == null) continue;
                    var h = Object.Instantiate(prefab);
                    h.name = "Home";
                    h.transform.SetParent(root, false);
                    h.transform.position = new Vector3(x, 0f, z);
                    h.transform.rotation = Quaternion.Euler(0f, z < -26f ? 180f : 0f, 0f);
                    h.transform.localScale = Vector3.one * config.homeScale;
                    MarlCityLots.NormalizeFootprint(h, 4.6f); // row pitch is 5.5
                }
            }
        }

        private static void BuildGreenery(Transform root, MarlEconomyPrefabConfig config)
        {
            if (config == null || config.treePrefabs == null || config.treePrefabs.Count == 0) return;
            int i = 0;
            float[] xs = { -26f, -20f, 19f }; // east columns removed — that's the business district now
            for (float z = -30f; z <= 30f; z += 6f)
                foreach (float x in xs)
                {
                    var prefab = MarlEconomyPrefabConfig.Pick(config.treePrefabs, i++);
                    if (prefab == null) continue;
                    var t = Object.Instantiate(prefab);
                    t.transform.SetParent(root, false);
                    t.transform.position = new Vector3(x + Random.Range(-1f, 1f), 0f, z + Random.Range(-1f, 1f));
                    t.transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
                }
        }
    }
}
