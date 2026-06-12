using System;
using System.Collections.Generic;

namespace MarlEconomy.UnityViz
{
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
        public int? stage;
        public int seed;
        public List<UnityMetricRow> metrics = new List<UnityMetricRow>();
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
        public Dictionary<string, float> policy = new Dictionary<string, float>();
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
        public int? target_firm_id;
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
        // Hunger and firm-cash diagnostics. Default 0 if missing
        // (older snapshots without these fields parse cleanly).
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
    public class UnityMetricRow
    {
        public int t_week;
        public string regime;
        public int num_households;
        public int num_firms;
        public float gdp_week;
        public float mean_price;
        public float mean_wage;
        public float inflation_proxy;
        public float gini;
        public float poverty_rate;
        public float avg_health;
        public float crime_rate;
        public float unemp_proxy;
        public int firm_entries;
        public int firm_exits;
        // Diagnostic columns. Newtonsoft fills missing primitives with their
        // default, so older CSV files without these fields still deserialize.
        public float mean_firm_cash;
        public float min_firm_cash;
        public float max_firm_cash;
        public float frac_insolvent_firms;
        public float total_inventory;
        public float mean_hunger;
        public float max_hunger;
        public float frac_hungry;
        public float frac_crisis;
        public int shop_attempts;
        public int shop_matched;
        public int clinic_attempts;
        public int clinic_matched;
        public int steal_success;
        public int work_refused_solvency;
        public int weekly_unmet_demand;
        public int weekly_total_demand;
        public int weekly_deaths;
        public int weekly_displacements;
        public int weekly_hunger_crisis;
        public float reward_households_mean;
        public float reward_firms_mean;
        public float reward_gov;
        public float reward_cb;
        public float reward_total;
        // Per-week typed event counters (events_*). Field names must match
        // the CSV columns exactly so Newtonsoft binds them by name.
        public int events_firm_entry_normal;
        public int events_firm_entry_emergency;
        public int events_firm_exit;
        public int events_firm_exit_blocked;
        public int events_firm_insolvent;
        public int events_firm_hire_refused;
        public int events_household_death;
        public int events_household_hunger_crisis;
        public int events_household_displaced;
        public int events_household_fatigue_crisis;
        public int events_household_infected;
        public int events_household_severe_illness;
        public int events_household_recovered;
        public int events_clinic_overcrowded;
        public int events_clinic_treatment_denied;
        public int events_disaster_earthquake;
        public int events_disaster_fire;
        public int events_disaster_flood;
        public int events_disaster_supply_shock;
        public int events_disaster_epidemic_wave;
        public float mean_fatigue;
        public float max_fatigue;
        public float frac_fatigued;
        public float frac_exhausted;
        public float frac_healthy;
        public float frac_exposed;
        public float frac_mild;
        public float frac_severe;
        public float frac_recovering;
        public int clinic_capacity;
        public int clinic_used;
        public int disaster_signal;
        public int supply_shock_active;
        public int epidemic_wave_active;
        public int firms_in_downtime;
    }

    [Serializable]
    public class UnityEventRecord
    {
        public string type;
        public int week;
        public int? firm_id;
        public Vec3Data position;
        public float previous_rate;
        public float interest_rate;
        public Dictionary<string, float> policy = new Dictionary<string, float>();
        // Household-side event payload (death, hunger crisis, fatigue crisis, etc.).
        public int? household_id;
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
