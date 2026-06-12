using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // Keyboard transport: Space = pause, comma/period = slower/faster.
    public class MarlPlaybackControls : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public Text weekLabel;

        private static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f, 4f, 8f };
        private int speedIndex = 2;

        private void Update()
        {
            if (playback == null) return;

            if (Input.GetKeyDown(KeyCode.Space)) playback.TogglePlayback();
            if (Input.GetKeyDown(KeyCode.Period) && speedIndex < Speeds.Length - 1)
            { speedIndex++; playback.playbackSpeed = Speeds[speedIndex]; }
            if (Input.GetKeyDown(KeyCode.Comma) && speedIndex > 0)
            { speedIndex--; playback.playbackSpeed = Speeds[speedIndex]; }

            if (weekLabel != null && playback.CurrentSnapshot != null && playback.Playback != null)
            {
                int total = playback.TotalFrames - 1;
                string paused = playback.autoPlay ? "" : "  [PAUSED]";
                weekLabel.text = $"Week {playback.CurrentSnapshot.week} / {total}   {Speeds[speedIndex]}x{paused}";
            }
        }
    }
}
