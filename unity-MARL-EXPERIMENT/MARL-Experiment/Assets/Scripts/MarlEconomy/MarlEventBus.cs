using System.Collections.Generic;
using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Canonical, de-noised view over the playback's `events` lists.
    //
    // The exported data has two quirks this layer hides from everyone else:
    //   1. Event names differ from what the old Unity code expected
    //      (firm_created/firm_closed/central_bank_rate_changed/... ).
    //   2. Each week's `events` list ACCUMULATES every prior week's events.
    //      We dedupe by (type, week, id) so each event is counted exactly once,
    //      then key everything off the event's own `week`.
    //   3. Disaster events carry no usable position (origin), so we derive a
    //      site from the affected firms / displaced or sick households.

    public enum MarlEventKind
    {
        FirmOpen, FirmClose, RateChange, PolicyChange,
        Death, HungerCrisis, FatigueCrisis, SevereIllness, Infected, Recovered,
        Disaster, Other
    }

    public enum DisasterKind { None, Fire, Flood, Earthquake, SupplyShock, EpidemicWave }

    public class MarlEvent
    {
        public MarlEventKind kind;
        public DisasterKind disaster = DisasterKind.None;
        public int week;
        public int firmId;
        public int householdId;
        public string rawType;
        public Vector3 position;
        public bool hasPosition;
        public float previousRate;
        public float interestRate;
    }

    public class DisasterWindow
    {
        public DisasterKind kind;
        public int startWeek;
        public int endWeek;     // inclusive
        public Vector3 site;
        public bool hasSite;
        public int firmId = -1; // victim firm, when one was derived — resolve its
                                // CURRENT position through the playback, since
                                // firms live on lots, not at the data positions

        public bool IsActive(int week) => week >= startWeek && week <= endWeek;
        public float Progress(int week) =>
            endWeek <= startWeek ? 1f : Mathf.Clamp01((week - startWeek) / (float)(endWeek - startWeek));
    }

    public class MarlEventBus
    {
        // Visual durations for inferred disaster windows (weeks). These mirror the
        // simulation constants (fire downtime 4, flood 6, supply shock 8, epidemic 12);
        // earthquake is instantaneous but we keep rubble visible briefly.
        public const int FireWeeks = 4;
        public const int FloodWeeks = 6;
        public const int SupplyShockWeeks = 8;
        public const int EpidemicWeeks = 12;
        public const int EarthquakeWeeks = 2;

        public readonly List<MarlEvent> Log = new List<MarlEvent>();
        public readonly List<DisasterWindow> DisasterWindows = new List<DisasterWindow>();

        private readonly Dictionary<int, List<MarlEvent>> _byWeek = new Dictionary<int, List<MarlEvent>>();

        public static MarlEventBus Build(UnityPlaybackRoot playback)
        {
            var bus = new MarlEventBus();
            if (playback == null || playback.snapshots == null) return bus;

            // Snapshot lookup by week for disaster-site derivation.
            var snapByWeek = new Dictionary<int, UnitySnapshot>();
            foreach (var s in playback.snapshots)
                snapByWeek[s.week] = s;

            var seen = new HashSet<string>();
            foreach (var snap in playback.snapshots)
            {
                if (snap.events == null) continue;
                foreach (var raw in snap.events)
                {
                    if (raw == null || string.IsNullOrEmpty(raw.type)) continue;
                    string key = raw.type + "|" + raw.week + "|" +
                                 raw.firm_id + "|" + raw.household_id;
                    if (!seen.Add(key)) continue;

                    var evt = Normalize(raw);
                    bus.Log.Add(evt);
                    if (!bus._byWeek.TryGetValue(evt.week, out var list))
                    {
                        list = new List<MarlEvent>();
                        bus._byWeek[evt.week] = list;
                    }
                    list.Add(evt);
                }
            }

            bus.Log.Sort((a, b) => a.week.CompareTo(b.week));
            bus.BuildDisasterWindows(snapByWeek);
            return bus;
        }

        private static MarlEvent Normalize(UnityEventRecord raw)
        {
            var e = new MarlEvent
            {
                week = raw.week,
                firmId = raw.firm_id,
                householdId = raw.household_id,
                rawType = raw.type,
                previousRate = raw.previous_rate,
                interestRate = raw.interest_rate,
            };

            if (raw.position != null &&
                !(raw.position.x == 0f && raw.position.y == 0f && raw.position.z == 0f))
            {
                e.position = new Vector3(raw.position.x, raw.position.y, raw.position.z);
                e.hasPosition = true;
            }

            switch (raw.type)
            {
                case "firm_created":
                case "firm_entry_normal":
                case "firm_entry_emergency":
                    e.kind = MarlEventKind.FirmOpen; break;

                case "firm_closed":
                case "firm_exit":
                case "firm_insolvent":
                    e.kind = MarlEventKind.FirmClose; break;

                case "central_bank_rate_changed":
                case "interest_rate_change":
                    e.kind = MarlEventKind.RateChange; break;

                case "government_policy_changed":
                    e.kind = MarlEventKind.PolicyChange; break;

                case "household_death":
                    e.kind = MarlEventKind.Death; break;
                case "household_hunger_crisis":
                    e.kind = MarlEventKind.HungerCrisis; break;
                case "household_fatigue_crisis":
                    e.kind = MarlEventKind.FatigueCrisis; break;
                case "household_severe_illness":
                    e.kind = MarlEventKind.SevereIllness; break;
                case "household_infected":
                    e.kind = MarlEventKind.Infected; break;
                case "household_recovered":
                    e.kind = MarlEventKind.Recovered; break;

                case "disaster_fire":
                    e.kind = MarlEventKind.Disaster; e.disaster = DisasterKind.Fire; break;
                case "disaster_flood":
                    e.kind = MarlEventKind.Disaster; e.disaster = DisasterKind.Flood; break;
                case "disaster_earthquake":
                    e.kind = MarlEventKind.Disaster; e.disaster = DisasterKind.Earthquake; break;
                case "disaster_supply_shock":
                    e.kind = MarlEventKind.Disaster; e.disaster = DisasterKind.SupplyShock; break;
                case "disaster_epidemic_wave":
                    e.kind = MarlEventKind.Disaster; e.disaster = DisasterKind.EpidemicWave; break;

                default:
                    e.kind = MarlEventKind.Other; break;
            }
            return e;
        }

        private void BuildDisasterWindows(Dictionary<int, UnitySnapshot> snapByWeek)
        {
            foreach (var e in Log)
            {
                if (e.kind != MarlEventKind.Disaster) continue;
                int dur;
                switch (e.disaster)
                {
                    case DisasterKind.Fire: dur = FireWeeks; break;
                    case DisasterKind.Flood: dur = FloodWeeks; break;
                    case DisasterKind.SupplyShock: dur = SupplyShockWeeks; break;
                    case DisasterKind.EpidemicWave: dur = EpidemicWeeks; break;
                    case DisasterKind.Earthquake: dur = EarthquakeWeeks; break;
                    default: dur = 2; break;
                }

                var win = new DisasterWindow
                {
                    kind = e.disaster,
                    startWeek = e.week,
                    endWeek = e.week + dur - 1,
                };
                DeriveSite(win, e, snapByWeek);
                DisasterWindows.Add(win);
            }
        }

        // Disaster events have no real position, so look at the snapshot at the
        // start week to find where the damage landed.
        private void DeriveSite(DisasterWindow win, MarlEvent e, Dictionary<int, UnitySnapshot> snapByWeek)
        {
            if (e.hasPosition) { win.site = e.position; win.hasSite = true; return; }
            if (!snapByWeek.TryGetValue(e.week, out var snap) || snap == null) return;

            switch (win.kind)
            {
                case DisasterKind.Fire:
                case DisasterKind.Earthquake:
                {
                    // The firm with the lowest cash / in downtime is the most likely victim.
                    FirmSnapshot worst = null;
                    foreach (var f in snap.firms)
                        if (worst == null || f.cash < worst.cash) worst = f;
                    if (worst != null)
                    {
                        win.firmId = worst.id;
                        win.site = ToVec(worst.position); // fallback only
                        win.hasSite = true;
                    }
                    break;
                }
                case DisasterKind.Flood:
                {
                    // Centroid of displaced households, falling back to firm centroid.
                    Vector3 sum = Vector3.zero; int n = 0;
                    foreach (var h in snap.households)
                        if (h.visual_state == "displaced" && h.position != null)
                        { sum += ToVec(h.position); n++; }
                    if (n == 0)
                        foreach (var f in snap.firms)
                            if (f.position != null) { sum += ToVec(f.position); n++; }
                    if (n > 0) { win.site = sum / n; win.hasSite = true; }
                    break;
                }
                case DisasterKind.EpidemicWave:
                    // Clinic anchor (constants.UNITY_CLINIC_POSITION).
                    win.site = new Vector3(-14f, 0f, 5f); win.hasSite = true;
                    break;
                case DisasterKind.SupplyShock:
                    win.hasSite = false; // global effect, no single site
                    break;
            }
        }

        private static Vector3 ToVec(Vec3Data v) => v == null ? Vector3.zero : new Vector3(v.x, v.y, v.z);

        public List<MarlEvent> EventsAtWeek(int week) =>
            _byWeek.TryGetValue(week, out var list) ? list : null;

        public DisasterWindow WindowStartingAt(DisasterKind kind, int week)
        {
            foreach (var w in DisasterWindows)
                if (w.kind == kind && w.startWeek == week) return w;
            return null;
        }

        public List<DisasterWindow> ActiveWindows(int week, List<DisasterWindow> buffer)
        {
            buffer.Clear();
            foreach (var w in DisasterWindows)
                if (w.IsActive(week)) buffer.Add(w);
            return buffer;
        }
    }
}
