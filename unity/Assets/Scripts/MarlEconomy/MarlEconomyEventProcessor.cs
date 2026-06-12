using UnityEngine;

namespace MarlEconomy.UnityViz
{
    public class MarlEconomyEventProcessor : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public MarlEconomyFX fx;
        public MarlEventBanner banner;

        private int lastWeek = -1;

        private void Update()
        {
            if (playback?.CurrentSnapshot == null) return;
            var snap = playback.CurrentSnapshot;
            if (snap.week == lastWeek) return;
            lastWeek = snap.week;

            foreach (var evt in snap.events)
                HandleEvent(evt, snap.week);

            int stealCount = 0;
            foreach (var household in snap.households)
            {
                if (household.action_label != "steal") continue;
                stealCount++;
                Vector3 pos = household.position != null
                    ? new Vector3(household.position.x, household.position.y, household.position.z)
                    : Vector3.zero;
                fx?.PlayCrime(pos);
            }
        }

        private void HandleEvent(UnityEventRecord evt, int week)
        {
            Vector3 pos = evt.position != null
                ? new Vector3(evt.position.x, evt.position.y, evt.position.z)
                : Vector3.zero;

            switch (evt.type)
            {
                case "firm_entry_normal":
                    fx?.PlayFirmSpawn(pos);
                    banner?.Post($"Week {week}  —  NEW FIRM OPENED", new Color(0.35f, 0.95f, 0.50f));
                    break;

                case "firm_entry_emergency":
                    fx?.PlayFirmSpawn(pos);
                    banner?.Post($"Week {week}  —  EMERGENCY FIRM ENTRY", new Color(0.55f, 1.00f, 0.35f));
                    break;

                case "firm_exit":
                case "firm_insolvent":
                    fx?.PlayFirmExit(pos);
                    banner?.Post($"Week {week}  —  FIRM COLLAPSED", new Color(0.95f, 0.50f, 0.18f));
                    break;

                case "disaster_fire":
                    fx?.PlayDisaster(Vector3.zero, "fire");
                    banner?.Post($"Week {week}  —  FIRE DISASTER", new Color(1.00f, 0.32f, 0.08f));
                    break;

                case "disaster_flood":
                    fx?.PlayDisaster(Vector3.zero, "flood");
                    banner?.Post($"Week {week}  —  FLOOD DISASTER", new Color(0.28f, 0.58f, 1.00f));
                    break;

                case "disaster_earthquake":
                    fx?.PlayDisaster(Vector3.zero, "earthquake");
                    banner?.Post($"Week {week}  —  EARTHQUAKE", new Color(0.90f, 0.72f, 0.28f));
                    break;

                case "disaster_supply_shock":
                    fx?.PlayDisaster(Vector3.zero, "supply_shock");
                    banner?.Post($"Week {week}  —  SUPPLY SHOCK", new Color(1.00f, 0.85f, 0.18f));
                    break;

                case "disaster_epidemic_wave":
                    fx?.PlayDisaster(Vector3.zero, "epidemic_wave");
                    banner?.Post($"Week {week}  —  EPIDEMIC WAVE", new Color(0.50f, 0.95f, 0.28f));
                    break;

                case "household_death":
                    fx?.PlayHouseholdDeath(pos);
                    break;

                case "interest_rate_change":
                    banner?.Post($"Week {week}  —  INTEREST RATE  {evt.previous_rate:0.000} → {evt.interest_rate:0.000}",
                        new Color(0.95f, 0.85f, 0.28f));
                    break;
            }
        }
    }
}
