using System;
using System.Collections.Generic;

namespace MarlEconomy.UnityViz
{
    // Mirrors the schema produced by marl_economy/viz/export_unity.py, shaped for
    // Unity's built-in JsonUtility (no external JSON dependency):
    //   * the `policy` map is a typed PolicyData (JsonUtility can't do dictionaries)
    //   * nullable ids are plain ints (JSON null -> 0; firm/household events also
    //     carry a position, which we prefer for placement anyway)
    //   * `metrics` is omitted on purpose — the HUD reads everything it needs from
    //     each snapshot's macro block, and skipping it avoids CSV-typing pitfalls.

    [Serializable]
    public class UnityPlaybackRoot
    {
        public int schema_version;
        public string generated_at;
        public string source_run_dir;
        public float replay_seconds_per_week = 0.35f;
        public float scene_scale = 1.0f;
        public string mode;
        public string checkpoint_path;
        public string regime;
        public int stage;
        public int seed;
        public List<UnitySnapshot> snapshots = new List<UnitySnapshot>();
    }

    [Serializable]
    public class UnitySnapshot
    {
        public int week;
        public string regime;
        public List<HouseholdSnapshot> households = new List<HouseholdSnapshot>();
        public List<FirmSnapshot> firms = new List<FirmSnapshot>();
        public InstitutionSet institutions;
        public MacroSnapshot macro;
        public PolicyData policy;
        public float interest_rate;
        public List<UnityEventRecord> events = new List<UnityEventRecord>();
    }

    [Serializable]
    public class HouseholdSnapshot
    {
        public int id;
        public float wealth;
        public float health;
        public float hunger;
        public float fatigue;
        public int illness;           // 0=healthy, 1=exposed, 2=mild, 3=severe, 4=recovering
        public int last_action;
        public string action_label;
        public bool action_success;
        public int target_firm_id;
        public float reward;
        public float wealth_change;
        public float health_change;
        public bool crime_success;
        public string wealth_class;
        public Vec3Data home_position;
        public Vec3Data position;
        public Vec3Data destination;
        public string visual_state;
    }

    [Serializable]
    public class FirmSnapshot
    {
        public int id;
        public float price;
        public float wage_offer;
        public float stock;
        public float cash;
        public float capacity;
        public Vec3Data position;
        public bool spawned_this_step;
        public string visual_type;
    }

    [Serializable]
    public class InstitutionSet
    {
        public InstitutionSnapshot government;
        public InstitutionSnapshot central_bank;
        public InstitutionSnapshot market;
    }

    [Serializable]
    public class InstitutionSnapshot
    {
        public Vec3Data position;
    }

    [Serializable]
    public class MacroSnapshot
    {
        public float gdp;
        public float gini;
        public float inflation;
        public float avg_health;
        public float poverty;
        public float crime_rate;
        public float unemployment;
        public int num_firms;
        public float mean_hunger;
        public float frac_hungry;
        public float mean_firm_cash;
        public float mean_fatigue;
        public float frac_fatigued;
        public float frac_severe;
        public float frac_mild;
        public int clinic_capacity;
        public int clinic_used;
        public int disaster_signal;
        public int firms_in_downtime;
    }

    [Serializable]
    public class PolicyData
    {
        public float tax_rate;
        public float welfare;
        public float policing;
        public float healthcare_access;
        public float education_quality;
        public float min_wage_floor;
        public float price_level;
        public float wage_base;
    }

    [Serializable]
    public class UnityEventRecord
    {
        public string type;
        public int week;
        public int firm_id;
        public Vec3Data position;
        public float previous_rate;
        public float interest_rate;
        public PolicyData policy;
        public int household_id;
        public float hunger;
        public float wealth;
    }

    [Serializable]
    public class Vec3Data
    {
        public float x;
        public float y;
        public float z;
    }
}
