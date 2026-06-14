using UnityEngine;
using UnityEngine.UI;

namespace MarlEconomy.UnityViz
{
    // Keyboard transport: Space = pause, comma/period = slower/faster (preset
    // stops). Speed is read/written through playback.playbackSpeed — the same
    // value the on-screen MarlSpeedSlider drives — so the keys and the slider
    // stay in sync even after the other one is used.
    public class MarlPlaybackControls : MonoBehaviour
    {
        public MarlEconomyPlayback playback;
        public Text weekLabel;

        private static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f, 4f, 8f };

        private void Update()
        {
            if (playback == null) return;

            if (Input.GetKeyDown(KeyCode.Space)) playback.TogglePlayback();
            if (Input.GetKeyDown(KeyCode.Period)) playback.playbackSpeed = Step(+1);
            if (Input.GetKeyDown(KeyCode.Comma))  playback.playbackSpeed = Step(-1);

            if (weekLabel != null && playback.CurrentSnapshot != null && playback.Playback != null)
            {
                int total = playback.TotalFrames - 1;
                string paused = playback.autoPlay ? "" : "   [PAUSED]";
                weekLabel.text = $"Week {playback.CurrentSnapshot.week} / {total}{paused}";
            }
        }

        // Nearest preset to the current speed, moved one stop in `dir`. Reading the
        // live speed first means a press still behaves sensibly after a slider drag.
        private float Step(int dir)
        {
            int idx = 0;
            float best = Mathf.Abs(playback.playbackSpeed - Speeds[0]);
            for (int i = 1; i < Speeds.Length; i++)
            {
                float d = Mathf.Abs(playback.playbackSpeed - Speeds[i]);
                if (d < best) { best = d; idx = i; }
            }
            return Speeds[Mathf.Clamp(idx + dir, 0, Speeds.Length - 1)];
        }
    }
}
