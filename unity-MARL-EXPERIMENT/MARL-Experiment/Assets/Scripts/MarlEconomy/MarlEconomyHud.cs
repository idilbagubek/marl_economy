using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // Top-left metrics panel, including the active-disaster readout.
    public class MarlEconomyHud : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public Text label;

        private readonly List<DisasterWindow> _active = new List<DisasterWindow>();
        private readonly StringBuilder _sb = new StringBuilder(512);

        private void Update()
        {
            if (playback == null || label == null || playback.CurrentSnapshot == null) return;

            UnitySnapshot snap = playback.CurrentSnapshot;
            MacroSnapshot m = snap.macro;
            if (m == null) { label.text = $"Week {snap.week}"; return; }

            int steals = 0;
            foreach (var h in snap.households) if (h.action_label == "steal") steals++;

            string disasterLine = "Calm";
            if (playback.EventBus != null)
            {
                playback.EventBus.ActiveWindows(snap.week, _active);
                if (_active.Count > 0)
                {
                    var names = new List<string>();
                    foreach (var w in _active) names.Add(MarlVisualPalette.DisasterLabel(w.kind));
                    disasterLine = string.Join(", ", names);
                }
            }

            _sb.Clear();
            _sb.Append("Regime: ").Append(snap.regime).Append('\n');
            _sb.Append("Week: ").Append(snap.week).Append(" / ").Append(playback.TotalFrames - 1).Append('\n');
            _sb.Append("GDP: ").Append(m.gdp.ToString("0.000")).Append('\n');
            _sb.Append("Gini: ").Append(m.gini.ToString("0.000")).Append('\n');
            _sb.Append("Inflation: ").Append(m.inflation.ToString("0.000")).Append('\n');
            _sb.Append("Unemployment: ").Append(m.unemployment.ToString("0.0%")).Append('\n');
            _sb.Append("Crime: ").Append(m.crime_rate.ToString("0.000")).Append("  (").Append(steals).Append(" thefts)\n");
            _sb.Append("Avg Health: ").Append(m.avg_health.ToString("0.000")).Append('\n');
            _sb.Append("Hungry %: ").Append(m.frac_hungry.ToString("0.0%")).Append('\n');
            _sb.Append("Fatigued %: ").Append(m.frac_fatigued.ToString("0.0%")).Append('\n');
            _sb.Append("Sick %: ").Append((m.frac_mild + m.frac_severe).ToString("0.0%")).Append('\n');
            _sb.Append("Clinic: ").Append(m.clinic_used).Append('/').Append(m.clinic_capacity).Append('\n');
            _sb.Append("Firms: ").Append(m.num_firms);
            if (m.firms_in_downtime > 0) _sb.Append(" (").Append(m.firms_in_downtime).Append(" down)");
            _sb.Append('\n');
            _sb.Append("Interest Rate: ").Append(snap.interest_rate.ToString("0.000")).Append('\n');
            _sb.Append("Event: ").Append(disasterLine);

            label.text = _sb.ToString();
        }
    }
}
