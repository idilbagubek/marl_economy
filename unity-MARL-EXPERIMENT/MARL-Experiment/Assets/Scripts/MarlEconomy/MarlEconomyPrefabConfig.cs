using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // All art references live here so the user can swap in their own models with
    // zero code changes. Populate it via the editor menu
    // "MARL Economy/Rebuild Prefab Config" (which fills it from the imported packs)
    // or by hand in the Inspector.
    [CreateAssetMenu(menuName = "MarlEconomy/Prefab Config", fileName = "MarlEconomyPrefabConfig")]
    public class MarlEconomyPrefabConfig : ScriptableObject
    {
        [Header("Households (rigged CityPeople)")]
        public List<GameObject> householdPrefabs = new List<GameObject>();
        public float householdScale = 1.0f;

        [Header("Firms (commercial / industrial buildings)")]
        public List<GameObject> firmPrefabs = new List<GameObject>();
        public float firmScale = 1.0f;

        [Header("Homes (civil buildings, static dressing)")]
        public List<GameObject> homePrefabs = new List<GameObject>();
        public float homeScale = 1.0f;

        [Header("Landmarks")]
        public GameObject governmentPrefab;
        public GameObject centralBankPrefab;
        public GameObject marketPrefab;
        public GameObject clinicPrefab;
        public GameObject bankPrefab;        // investment / savings
        public GameObject restPrefab;
        public GameObject riskZonePrefab;    // crime hotspot
        public float landmarkScale = 1.0f;

        [Header("Dressing (props / vegetation)")]
        public List<GameObject> treePrefabs = new List<GameObject>();
        public List<GameObject> propPrefabs = new List<GameObject>();

        [Header("Chaos vehicles")]
        public GameObject fireTruckPrefab;
        public GameObject ambulancePrefab;
        public GameObject policeCarPrefab;

        [Header("Materials")]
        public Material particleMaterial;    // URP/Particles/Unlit recommended
        public Material waterMaterial;       // translucent blue for floods

        public static T Pick<T>(IList<T> list, int index) where T : Object
        {
            if (list == null || list.Count == 0) return null;
            int i = ((index % list.Count) + list.Count) % list.Count;
            return list[i];
        }
    }
}
