using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace MarlEconomy.UnityViz.EditorTools
{
    // Populates Assets/Resources/MarlEconomyPrefabConfig.asset from the imported
    // SimplePoly City buildings/vehicles + the DenysAlmaral CityPeople characters.
    // Run once via the menu; re-run after swapping art. Writing the asset from
    // inside the editor (not externally) is why this is reliable — Unity won't
    // revert it the way it does an out-of-editor .asset edit.
    //
    // SimplePoly buildings are single-mesh with the windows baked into each
    // building's texture (no separate glass submesh), so the old "unrendered
    // window" problem can't occur. Their materials ship on Built-in/Standard,
    // which MarlUrpMaterials converts to URP at runtime (and the "Fix Magenta
    // Materials" menu converts on disk).
    public static class MarlPrefabConfigBuilder
    {
        private const string Simple = "Assets/SimplePoly City - Low Poly Assets/Prefab";
        private const string People = "Assets/DenysAlmaral/CityPeople/Prefabs";
        private const string Static = Simple + "/Vehicles/Vehicle with Static Wheels";
        private const string Cfxr   = "Assets/JMO Assets/Cartoon FX Remaster/CFXR Prefabs";
        private const string AssetPath = "Assets/Resources/MarlEconomyPrefabConfig.asset";

        [MenuItem("MARL Economy/Rebuild Prefab Config")]
        public static void Rebuild()
        {
            var cfg = AssetDatabase.LoadAssetAtPath<MarlEconomyPrefabConfig>(AssetPath);
            bool isNew = cfg == null;
            if (isNew) cfg = ScriptableObject.CreateInstance<MarlEconomyPrefabConfig>();

            // Households stay CityPeople — SimplePoly ships no characters.
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

            // Firms — a varied parade of SimplePoly shops (picked by firm id).
            cfg.firmPrefabs = Load(
                $"{Simple}/Buildings/Building_Coffee Shop.prefab",
                $"{Simple}/Buildings/Building_Pizza.prefab",
                $"{Simple}/Buildings/Building_Fast Food.prefab",
                $"{Simple}/Buildings/Building_Restaurant.prefab",
                $"{Simple}/Buildings/Building_Bakery.prefab",
                $"{Simple}/Buildings/Building_Books Shop.prefab",
                $"{Simple}/Buildings/Building_Clothing.prefab",
                $"{Simple}/Buildings/Building_Shoes Shop.prefab",
                $"{Simple}/Buildings/Building_Gift Shop.prefab",
                $"{Simple}/Buildings/Building_Music Store.prefab",
                $"{Simple}/Buildings/Building_Chicken Shop.prefab",
                $"{Simple}/Buildings/Building_Auto Service.prefab");

            // Homes — detached houses + mid-rise residential (apartment) blocks.
            cfg.homePrefabs = Load(
                $"{Simple}/Buildings/Building_House_01_color01.prefab",
                $"{Simple}/Buildings/Building_House_02_color01.prefab",
                $"{Simple}/Buildings/Building_House_03_color01.prefab",
                $"{Simple}/Buildings/Building_House_04_color01.prefab",
                $"{Simple}/Buildings/Building_House_01_color02.prefab",
                $"{Simple}/Buildings/Building_House_02_color02.prefab",
                $"{Simple}/Buildings/Building_Residential_color01.prefab",
                $"{Simple}/Buildings/Building_Residential_color02.prefab",
                $"{Simple}/Buildings/Building_Residential_color03.prefab");

            cfg.governmentPrefab  = One($"{Simple}/Buildings/Building Sky_big_color01.prefab");   // imposing tower
            cfg.centralBankPrefab = One($"{Simple}/Buildings/Building Sky_big_color02.prefab");
            cfg.marketPrefab      = One($"{Simple}/Buildings/Building_Super Market.prefab");      // literal market
            cfg.clinicPrefab      = One($"{Simple}/Buildings/Building_Drug Store.prefab");        // literal pharmacy
            cfg.bankPrefab        = One($"{Simple}/Buildings/Building Sky_small_color01.prefab"); // investment bank
            cfg.restPrefab        = One($"{Simple}/Buildings/Building_Residential_color01.prefab"); // big apartment
            cfg.riskZonePrefab    = One($"{Simple}/Buildings/Building_Factory.prefab");           // gritty crime zone

            // SimplePoly has no fire truck — a heavy truck stands in for one.
            cfg.fireTruckPrefab = One($"{Static}/Vehicle_Truck_color01.prefab");
            cfg.ambulancePrefab = One($"{Static}/Vehicle_Ambulance.prefab");
            cfg.policeCarPrefab = One($"{Static}/Vehicle_Police Car.prefab");

            cfg.treePrefabs = Load(
                $"{Simple}/Natures/Natures_Big Tree.prefab",
                $"{Simple}/Natures/Natures_Cube Tree.prefab",
                $"{Simple}/Natures/Natures_Fir Tree.prefab",
                $"{Simple}/Natures/Natures_Bush_02.prefab");

            cfg.propPrefabs = Load(  // earthquake rubble
                $"{Simple}/Natures/Natures_Rock_Big.prefab",
                $"{Simple}/Natures/Natures_Rock_small.prefab");

            // Event VFX — Cartoon FX Remaster prefabs matched to each event's mood.
            cfg.fxFirmOpen    = One($"{Cfxr}/Misc/CFXR Magic Poof.prefab");                       // celebratory poof
            cfg.fxFirmClose   = One($"{Cfxr}/Explosions/CFXR Explosion Smoke 2 Solo (HDR).prefab"); // collapse smoke
            cfg.fxCrime       = One($"{Cfxr}/Impacts/CFXR Hit A (Red).prefab");                   // theft strike
            cfg.fxDeath       = One($"{Cfxr}/Eerie/CFXR2 Souls Escape.prefab");                   // soul leaves
            cfg.fxFire        = One($"{Cfxr}/Fire/CFXR Fire.prefab");                             // persistent blaze (loops)
            cfg.fxFireBurst   = One($"{Cfxr}/Explosions/CFXR3 Fire Explosion B.prefab");          // fire onset
            cfg.fxFlood       = One($"{Cfxr}/Liquids/CFXR Water Splash (Smaller).prefab");        // flood splash
            cfg.fxEarthquake  = One($"{Cfxr}/Impacts/CFXR2 Ground Hit.prefab");                   // quake dust hit
            cfg.fxEpidemic    = One($"{Cfxr}/Misc/CFXR2 Poison Cloud.prefab");                    // sickly green cloud
            cfg.fxSupplyShock = One($"{Cfxr}/Misc/CFXR Flash.prefab");                            // shock flash

            if (!AssetDatabase.IsValidFolder("Assets/Resources"))
                AssetDatabase.CreateFolder("Assets", "Resources");

            if (isNew) AssetDatabase.CreateAsset(cfg, AssetPath);
            EditorUtility.SetDirty(cfg);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            Debug.Log($"[MARL] Prefab config {(isNew ? "created" : "updated")} at {AssetPath}: " +
                      $"{cfg.householdPrefabs.Count} people, {cfg.firmPrefabs.Count} firm buildings, " +
                      $"{cfg.homePrefabs.Count} homes (SimplePoly City).");
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
