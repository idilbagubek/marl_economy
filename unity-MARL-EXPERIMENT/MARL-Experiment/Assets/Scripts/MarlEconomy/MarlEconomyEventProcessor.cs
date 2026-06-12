using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Watches the playback week and fires one-shot FX + alert banners through the
    // normalized MarlEventBus (so the correct events actually trigger and land on
    // the right buildings). Persistent disaster state lives in MarlChaosDirector.
    public class MarlEconomyEventProcessor : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public MarlEconomyFX fx;
        public MarlEventBanner banner;

        private int lastWeek = -1;

        private void Update()
        {
            var snap = playback != null ? playback.CurrentSnapshot : null;
            if (snap == null || playback.EventBus == null) return;
            if (snap.week == lastWeek) return;

            int from = lastWeek < 0 ? snap.week : lastWeek + 1;
            lastWeek = snap.week;

            for (int w = from; w <= snap.week; w++)
                FireEventsForWeek(w);

            FireRateAndPolicy(snap);
            FireCrime(snap);
        }

        private void FireEventsForWeek(int week)
        {
            var events = playback.EventBus.EventsAtWeek(week);
            if (events == null) return;

            foreach (var e in events)
            {
                switch (e.kind)
                {
                    case MarlEventKind.FirmOpen:
                        fx?.PlayFirmSpawn(FirmPos(e));
                        banner?.Post($"Week {week}  —  NEW FIRM OPENED", new Color(0.35f, 0.95f, 0.50f));
                        break;

                    case MarlEventKind.FirmClose:
                        fx?.PlayFirmExit(FirmPos(e));
                        banner?.Post($"Week {week}  —  FIRM COLLAPSED", new Color(0.95f, 0.50f, 0.18f));
                        break;

                    case MarlEventKind.Disaster:
                        MarlVisualPalette.Disaster(e.disaster, out Color a, out _);
                        fx?.PlayDisaster(DisasterSite(e), e.disaster);
                        banner?.Post($"Week {week}  —  {MarlVisualPalette.DisasterLabel(e.disaster)}", a);
                        break;

                    case MarlEventKind.Death:
                        fx?.PlayHouseholdDeath(HouseholdPos(e));
                        break;
                }
            }
        }

        private void FireRateAndPolicy(UnitySnapshot snap)
        {
            var prev = playback.PreviousSnapshot;
            if (prev == null || prev == snap) return;

            if (!Mathf.Approximately(prev.interest_rate, snap.interest_rate))
            {
                Color c = snap.interest_rate > prev.interest_rate
                    ? new Color(0.95f, 0.55f, 0.25f) : new Color(0.45f, 0.85f, 1.00f);
                banner?.Post($"Week {snap.week}  —  INTEREST RATE  {prev.interest_rate:0.000} → {snap.interest_rate:0.000}", c);
            }

            PolicyData a = prev.policy, b = snap.policy;
            if (a == null || b == null) return;

            string changed = null; float pv = 0, cv = 0, best = 0.0005f;
            void Check(string name, float x, float y)
            {
                float d = Mathf.Abs(x - y);
                if (d > best) { best = d; changed = name; pv = x; cv = y; }
            }
            Check("TAX", a.tax_rate, b.tax_rate);
            Check("WELFARE", a.welfare, b.welfare);
            Check("POLICING", a.policing, b.policing);
            Check("HEALTHCARE", a.healthcare_access, b.healthcare_access);
            Check("EDUCATION", a.education_quality, b.education_quality);
            Check("MIN WAGE", a.min_wage_floor, b.min_wage_floor);

            if (changed != null)
                banner?.Post($"Week {snap.week}  —  POLICY {changed}  {pv:0.00} → {cv:0.00}",
                    new Color(0.95f, 0.85f, 0.35f));
        }

        private void FireCrime(UnitySnapshot snap)
        {
            int fired = 0;
            foreach (var h in snap.households)
            {
                if (h.action_label != "steal") continue;
                var go = playback.GetHouseholdObject(h.id);
                Vector3 p = go != null ? go.transform.position
                    : (h.position != null ? new Vector3(h.position.x, h.position.y, h.position.z) : MarlEconomyPlayback.RiskZonePos);
                fx?.PlayCrime(p + Vector3.up * 1.2f);
                if (++fired >= 12) break;
            }
        }

        private Vector3 FirmPos(MarlEvent e)
        {
            // Firms live on business-district lots, so the live object (or its
            // lot) wins over the stale data-space position in the event.
            var go = playback.GetFirmObject(e.firmId);
            if (go != null) return go.transform.position;
            if (playback.TryGetFirmAnchor(e.firmId, out Vector3 p)) return p;
            if (e.hasPosition) return e.position;
            return MarlEconomyPlayback.MarketPos;
        }

        private Vector3 DisasterSite(MarlEvent e)
        {
            var win = playback.EventBus.WindowStartingAt(e.disaster, e.week);
            if (win != null)
            {
                if (win.firmId >= 0 && playback.TryGetFirmAnchor(win.firmId, out Vector3 p))
                    return p + Vector3.up * 1.5f;
                if (win.hasSite) return win.site + Vector3.up * 1.5f;
            }
            if (e.hasPosition) return e.position;
            return MarlEconomyPlayback.MarketPos + Vector3.up * 1.5f;
        }

        private Vector3 HouseholdPos(MarlEvent e)
        {
            var go = playback.GetHouseholdObject(e.householdId);
            if (go != null) return go.transform.position + Vector3.up;
            if (e.hasPosition) return e.position;
            return Vector3.zero;
        }
    }
}
