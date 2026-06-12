using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace MarlEconomy.UnityViz.EditorTools
{
    // Populates Assets/Resources/MarlEconomyPrefabConfig.asset from the imported
    // Toon City + CityPeople packs. Run once via the menu; re-run after swapping art.
    public static class MarlPrefabConfigBuilder
    {
        private const string Toon   = "Assets/Loading Games/Toon City Pack/Prefabs";
        private const string People = "Assets/DenysAlmaral/CityPeople/Prefabs";
        private const string AssetPath = "Assets/Resources/MarlEconomyPrefabConfig.asset";

        [MenuItem("MARL Economy/Rebuild Prefab Config")]
        public static void Rebuild()
        {
            var cfg = AssetDatabase.LoadAssetAtPath<MarlEconomyPrefabConfig>(AssetPath);
            bool isNew = cfg == null;
            if (isNew) cfg = ScriptableObject.CreateInstance<MarlEconomyPrefabConfig>();

            cfg.householdPrefabs = Load(
                $"{People}/city/casual_Female_G.prefab",
                $"{People}/city/casual_Male_G.prefab",
                $"{People}/downtown/casual_Female_K.prefab",
                $"{People}/downtown/casual_Male_K.prefab",
                $"{People}/elder/elder_Female_A.prefab",
                $"{People}/little_kids/little_boy_B.prefab",
                $"{People}/professions/Doctor_Male_B.prefab",
                $"{People}/professions/police_Female_A.prefab",
                $"{People}/disabilities/prostheticLeg_girl.prefab");

            cfg.firmPrefabs = Load(
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-coffee-shop.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-pizza-shop.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-fastfood-shop.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-ice-cream-shop.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-office.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-hotel.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-cinema.prefab",
                $"{Toon}/Buildings/Commercial Buildings/commercial-building-concert-hall.prefab",
                $"{Toon}/Buildings/Industrial Buildings/industrial-buildings-building-A.prefab",
                $"{Toon}/Buildings/Industrial Buildings/industrial-buildings-building-B.prefab",
                $"{Toon}/Buildings/Industrial Buildings/industrial-buildings-building-C.prefab");

            cfg.homePrefabs = Load(
                $"{Toon}/Buildings/Civil Buildings/suburban-house-A.prefab",
                $"{Toon}/Buildings/Civil Buildings/suburban-house-B.prefab",
                $"{Toon}/Buildings/Civil Buildings/suburban-house-C.prefab",
                $"{Toon}/Buildings/Civil Buildings/suburban-house-D.prefab",
                $"{Toon}/Buildings/Civil Buildings/suburban-house-E.prefab",
                $"{Toon}/Buildings/Civil Buildings/cb-apartment-A.prefab",
                $"{Toon}/Buildings/Civil Buildings/cb-apartment-B.prefab",
                $"{Toon}/Buildings/Civil Buildings/cb-apartment-C.prefab");

            cfg.governmentPrefab = One($"{Toon}/Buildings/Government Buildings/government-building-city-hall.prefab");
            cfg.centralBankPrefab = One($"{Toon}/Buildings/Government Buildings/government-building-bank.prefab");
            cfg.marketPrefab = One($"{Toon}/Buildings/Commercial Buildings/commercial-building-shopping-center.prefab");
            cfg.clinicPrefab = One($"{Toon}/Buildings/Government Buildings/government-building-museum.prefab");
            cfg.bankPrefab = One($"{Toon}/Buildings/Commercial Buildings/commercial-building-office.prefab");
            cfg.restPrefab = One($"{Toon}/Buildings/Government Buildings/government-building-school.prefab");
            cfg.riskZonePrefab = One($"{Toon}/Buildings/Government Buildings/government-building-police-station.prefab");

            cfg.fireTruckPrefab = One($"{Toon}/Vehicles/Government Vehicles/fire-truck.prefab");
            cfg.ambulancePrefab = One($"{Toon}/Vehicles/Government Vehicles/ambulance.prefab");
            cfg.policeCarPrefab = One($"{Toon}/Vehicles/Government Vehicles/police-car.prefab");

            cfg.treePrefabs = Load(
                $"{Toon}/Vegetation/tree-type-A.prefab",
                $"{Toon}/Vegetation/tree-type-B.prefab",
                $"{Toon}/Vegetation/tree-type-C.prefab",
                $"{Toon}/Vegetation/tree-type-D.prefab");

            cfg.propPrefabs = Load(
                $"{Toon}/Vegetation/rock-type-A.prefab",
                $"{Toon}/Vegetation/rock-type-B.prefab");

            if (!AssetDatabase.IsValidFolder("Assets/Resources"))
                AssetDatabase.CreateFolder("Assets", "Resources");

            if (isNew) AssetDatabase.CreateAsset(cfg, AssetPath);
            EditorUtility.SetDirty(cfg);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            Debug.Log($"[MARL] Prefab config {(isNew ? "created" : "updated")} at {AssetPath}: " +
                      $"{cfg.householdPrefabs.Count} people, {cfg.firmPrefabs.Count} firm buildings, " +
                      $"{cfg.homePrefabs.Count} homes.");
            Selection.activeObject = cfg;
        }

        private static List<GameObject> Load(params string[] paths)
        {
            var list = new List<GameObject>();
            foreach (var p in paths)
            {
                var go = AssetDatabase.LoadAssetAtPath<GameObject>(p);
                if (go != null) list.Add(go);
                else Debug.LogWarning($"[MARL] Prefab not found (skipped): {p}");
            }
            return list;
        }

        private static GameObject One(string path)
        {
            var go = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (go == null) Debug.LogWarning($"[MARL] Landmark prefab not found (skipped): {path}");
            return go;
        }
    }
}
