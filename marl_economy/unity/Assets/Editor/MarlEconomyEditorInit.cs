using UnityEditor;
using UnityEngine;
using MarlEconomy.UnityViz;

[InitializeOnLoad]
public static class MarlEconomyEditorInit
{
    static MarlEconomyEditorInit()
    {
        EditorApplication.delayCall += EnsurePrefabConfig;
    }

    private static void EnsurePrefabConfig()
    {
        if (Resources.Load<MarlEconomyPrefabConfig>("MarlEconomyPrefabConfig") != null)
            return;

        if (!AssetDatabase.IsValidFolder("Assets/Resources"))
            AssetDatabase.CreateFolder("Assets", "Resources");

        var config = ScriptableObject.CreateInstance<MarlEconomyPrefabConfig>();
        AssetDatabase.CreateAsset(config, "Assets/Resources/MarlEconomyPrefabConfig.asset");
        AssetDatabase.SaveAssets();
        Debug.Log("[MarlEconomy] Created Assets/Resources/MarlEconomyPrefabConfig.asset — assign your prefabs there.");
    }
}
