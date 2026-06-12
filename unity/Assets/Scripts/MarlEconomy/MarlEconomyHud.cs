using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    public class MarlEconomyHud : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public Text label;

        private void Update()
        {
            if (playback == null || label == null || playback.CurrentSnapshot == null)
            {
                return;
            }

            UnitySnapshot snapshot = playback.CurrentSnapshot;
            MacroSnapshot macro = snapshot.macro;
            if (macro == null)
            {
                label.text = $"Week {snapshot.week}";
                return;
            }

            int stealsThisWeek = 0;
            foreach (var h in snapshot.households)
                if (h.action_label == "steal") stealsThisWeek++;

            label.text =
                $"Regime: {snapshot.regime}\n" +
                $"Week: {snapshot.week}\n" +
                $"GDP: {macro.gdp:0.000}\n" +
                $"Gini: {macro.gini:0.000}\n" +
                $"Inflation: {macro.inflation:0.000}\n" +
                $"Unemployment: {macro.unemployment:0.000}\n" +
                $"Crime: {macro.crime_rate:0.000}  ({stealsThisWeek} this wk)\n" +
                $"Avg Health: {macro.avg_health:0.000}\n" +
                $"Mean Hunger: {macro.mean_hunger:0.000}\n" +
                $"Hungry %: {macro.frac_hungry:0.0%}\n" +
                $"Mean Fatigue: {macro.mean_fatigue:0.000}\n" +
                $"Fatigued %: {macro.frac_fatigued:0.0%}\n" +
                $"Sick (mild+severe): {(macro.frac_mild + macro.frac_severe):0.0%}\n" +
                $"Severe %: {macro.frac_severe:0.0%}\n" +
                $"Clinic: {macro.clinic_used}/{macro.clinic_capacity}\n" +
                $"Firms: {macro.num_firms}" +
                (macro.firms_in_downtime > 0 ? $" ({macro.firms_in_downtime} down)" : "") + "\n" +
                $"Mean Firm Cash: {macro.mean_firm_cash:0.000}\n" +
                $"Active Disasters: {macro.disaster_signal}\n" +
                $"Interest Rate: {snapshot.interest_rate:0.000}";
        }
    }
}
