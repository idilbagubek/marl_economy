using UnityEngine;

namespace MarlEconomy.UnityViz
{
    [CreateAssetMenu(menuName = "MarlEconomy/Prefab Config", fileName = "MarlEconomyPrefabConfig")]
    public class MarlEconomyPrefabConfig : ScriptableObject
    {
        [Header("Agents")]
        public GameObject householdPrefab;

        [Header("Firms")]
        public GameObject firmPrefab;

        [Header("Institutions")]
        public GameObject governmentPrefab;
        public GameObject centralBankPrefab;
        public GameObject marketPrefab;
        public GameObject clinicPrefab;
        public GameObject bankPrefab;
        public GameObject restPrefab;
        public GameObject riskZonePrefab;

        [Header("Materials")]
        public Material poorMaterial;
        public Material middleMaterial;
        public Material richMaterial;
        public Material eliteMaterial;
        public Material failedActionMaterial;
    }
}
