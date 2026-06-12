"""
Pygame Real-Time Viewer (Section 15.1).
2x2 grid showing all 4 regimes simultaneously.

Controls:
    Space: pause/resume
    1/2/3: speed (0.5x / 1x / 2x)
    F: focus on one regime (cycle)
    R: start/stop recording frames
    Esc: exit

Usage:
    python -m marl_economy.viz.viewer_pygame
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    REGIME_NAMES, REGIME_FREE_MARKET, REGIME_RESTRICTED, REGIME_SOCIAL_DEM, REGIME_COMMAND,
    DEFAULT_EPISODE_WEEKS, DEFAULT_RANDOM_SEED,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD, RUN_PROGRESS_INTERVAL_WEEKS,
    NUM_HOUSEHOLD_ACTIONS, FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS,
    FIRM_CAPACITY_DELTAS, GOV_TAX_DELTAS, GOV_WELFARE_DELTAS,
    GOV_POLICING_DELTAS, GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS,
    GOV_MINWAGE_DELTAS, CB_RATE_DELTAS,
    VIEWER_SPEED_SLOW, VIEWER_SPEED_DEFAULT, VIEWER_SPEED_FAST,
    VIEWER_BASE_FPS, VIEWER_WINDOW_WIDTH, VIEWER_WINDOW_HEIGHT,
    VIEWER_FONT_SMALL_SIZE, VIEWER_FONT_MED_SIZE, VIEWER_PANEL_MARGIN,
    VIEWER_HOUSEHOLD_AREA_RIGHT_MARGIN, VIEWER_HOUSEHOLD_AREA_BOTTOM_MARGIN,
    VIEWER_HOUSEHOLD_COLS, VIEWER_HOUSEHOLD_DOT_RADIUS, VIEWER_HEALTH_RING_RADIUS,
    VIEWER_HEALTH_RING_THRESHOLD, VIEWER_FIRM_X_OFFSET, VIEWER_FIRM_Y_START,
    VIEWER_FIRM_Y_SPACING, VIEWER_FIRM_SIZE_SCALE, VIEWER_FIRM_SIZE_MIN,
    VIEWER_FIRM_SIZE_MAX, VIEWER_FIRM_COLOR_SCALE,
    VIEWER_OVERLAY_OFFSET_Y, VIEWER_STATUS_OFFSET_Y,
    VIEWER_COLORS, VIEWER_REGIME_COLORS,
)
from marl_economy.env.world import World

COLORS = VIEWER_COLORS
REGIME_COLORS = VIEWER_REGIME_COLORS


class SimViewer:
    """Real-time viewer for MARL economy simulation."""

    def __init__(self, episode_weeks: int = DEFAULT_EPISODE_WEEKS, seed: int = DEFAULT_RANDOM_SEED):
        self.episode_weeks = episode_weeks
        self.seed = seed
        self.regimes = list(REGIME_NAMES)

        # World per regime
        self.worlds = {}
        self.rng = np.random.default_rng(seed)

        for regime in self.regimes:
            self.worlds[regime] = World(
                regime_name=regime,
                episode_weeks=episode_weeks,
                seed=seed,
                curriculum_phase=1,
            )
            self.worlds[regime].reset()

        # State
        self.paused = False
        self.speed = VIEWER_SPEED_DEFAULT
        self.focused_regime = None
        self.recording = False
        self.frame_count = 0

    def run_headless(self, steps: int = 156):
        """Run without pygame for testing."""
        for step in range(steps):
            for regime in self.regimes:
                world = self.worlds[regime]
                if world.done:
                    continue

                actions = {
                    int(world.h.ids[i]): self.rng.integers(0, NUM_HOUSEHOLD_ACTIONS)
                    for i in range(world.h.size)
                }

                firm_actions = None
                if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
                    alive_slots = world.f.alive_slots()
                    firm_actions = {
                        int(world.f.id[slot]): (
                            self.rng.integers(0, len(FIRM_PRICE_DELTAS)),
                            self.rng.integers(0, len(FIRM_WAGE_DELTAS)),
                            self.rng.integers(0, len(FIRM_CAPACITY_DELTAS)),
                        )
                        for slot in alive_slots.tolist()
                    }

                gov_action = None
                if step > 0 and step % GOV_UPDATE_PERIOD == 0:
                    gov_action = tuple(
                        self.rng.integers(0, len(d))
                        for d in [GOV_TAX_DELTAS, GOV_WELFARE_DELTAS,
                                  GOV_POLICING_DELTAS, GOV_HEALTHCARE_DELTAS,
                                  GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS]
                    )

                cb_action = None
                if step > 0 and step % CB_UPDATE_PERIOD == 0:
                    cb_action = self.rng.integers(0, len(CB_RATE_DELTAS))

                world.step(actions, firm_actions, gov_action, cb_action)

            if step % RUN_PROGRESS_INTERVAL_WEEKS == 0:
                print(f"\nWeek {step}:")
                for regime in self.regimes:
                    snap = self.worlds[regime].get_snapshot()
                    m = snap["macro"]
                    print(f"  {regime:<14} GDP={m['gdp']:.3f} Gini={m['gini']:.3f} "
                          f"Health={m['avg_health']:.3f} Firms={m['num_firms']}")

    def run_pygame(self):
        """Run with pygame visualization."""
        try:
            import pygame
        except ImportError:
            print("Pygame not installed. Install with: pip install pygame")
            print("Running headless instead...")
            self.run_headless()
            return

        pygame.init()
        WIDTH, HEIGHT = VIEWER_WINDOW_WIDTH, VIEWER_WINDOW_HEIGHT
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MARL Economy Viewer")
        clock = pygame.time.Clock()
        font_small = pygame.font.SysFont("monospace", VIEWER_FONT_SMALL_SIZE)
        font_med = pygame.font.SysFont("monospace", VIEWER_FONT_MED_SIZE)

        running = True
        step = 0

        while running and step < self.episode_weeks:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_1:
                        self.speed = VIEWER_SPEED_SLOW
                    elif event.key == pygame.K_2:
                        self.speed = VIEWER_SPEED_DEFAULT
                    elif event.key == pygame.K_3:
                        self.speed = VIEWER_SPEED_FAST
                    elif event.key == pygame.K_f:
                        if self.focused_regime is None:
                            self.focused_regime = 0
                        else:
                            self.focused_regime = (self.focused_regime + 1) % len(self.regimes)
                            if self.focused_regime == 0:
                                self.focused_regime = None

            if not self.paused:
                # Step all regimes
                for regime in self.regimes:
                    world = self.worlds[regime]
                    if world.done:
                        continue

                    actions = {
                        int(world.h.ids[i]): self.rng.integers(0, NUM_HOUSEHOLD_ACTIONS)
                        for i in range(world.h.size)
                    }
                    firm_actions = None
                    if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
                        alive_slots = world.f.alive_slots()
                        firm_actions = {
                            int(world.f.id[slot]): (
                                self.rng.integers(0, len(FIRM_PRICE_DELTAS)),
                                self.rng.integers(0, len(FIRM_WAGE_DELTAS)),
                                self.rng.integers(0, len(FIRM_CAPACITY_DELTAS)),
                            )
                            for slot in alive_slots.tolist()
                        }

                    gov_action = None
                    cb_action = None
                    if step > 0 and step % GOV_UPDATE_PERIOD == 0:
                        gov_action = tuple(
                            self.rng.integers(0, len(d))
                            for d in [GOV_TAX_DELTAS, GOV_WELFARE_DELTAS,
                                      GOV_POLICING_DELTAS, GOV_HEALTHCARE_DELTAS,
                                      GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS]
                        )
                    if step > 0 and step % CB_UPDATE_PERIOD == 0:
                        cb_action = self.rng.integers(0, len(CB_RATE_DELTAS))

                    world.step(actions, firm_actions, gov_action, cb_action)

                step += 1

            # Draw
            screen.fill(COLORS["bg"])

            # 2x2 grid
            panel_w = WIDTH // 2 - 2 * VIEWER_PANEL_MARGIN
            panel_h = HEIGHT // 2 - 2 * VIEWER_PANEL_MARGIN
            positions = [
                (VIEWER_PANEL_MARGIN, VIEWER_PANEL_MARGIN),
                (WIDTH // 2 + VIEWER_PANEL_MARGIN, VIEWER_PANEL_MARGIN),
                (VIEWER_PANEL_MARGIN, HEIGHT // 2 + VIEWER_PANEL_MARGIN),
                (WIDTH // 2 + VIEWER_PANEL_MARGIN, HEIGHT // 2 + VIEWER_PANEL_MARGIN),
            ]

            for i, regime in enumerate(self.regimes):
                if self.focused_regime is not None and i != self.focused_regime:
                    continue

                px, py = positions[i] if self.focused_regime is None else (VIEWER_PANEL_MARGIN, VIEWER_PANEL_MARGIN)
                pw = panel_w if self.focused_regime is None else WIDTH - 2 * VIEWER_PANEL_MARGIN
                ph = panel_h if self.focused_regime is None else HEIGHT - 2 * VIEWER_PANEL_MARGIN

                snap = self.worlds[regime].get_snapshot()
                self._draw_panel(screen, px, py, pw, ph, regime, snap,
                                 font_small, font_med)

            # Week counter
            week_text = font_med.render(f"Week: {step} | Speed: {self.speed}x | "
                                         f"{'PAUSED' if self.paused else 'RUNNING'}",
                                         True, COLORS["text"])
            screen.blit(week_text, (VIEWER_PANEL_MARGIN, HEIGHT - VIEWER_STATUS_OFFSET_Y))

            pygame.display.flip()
            clock.tick(int(VIEWER_BASE_FPS * self.speed))

        pygame.quit()

    def _draw_panel(self, screen, px, py, pw, ph, regime, snapshot,
                    font_small, font_med):
        """Draw one regime panel."""
        import pygame

        # Panel background
        pygame.draw.rect(screen, COLORS["panel_bg"],
                         (px, py, pw, ph))
        pygame.draw.rect(screen, REGIME_COLORS.get(regime, (200, 200, 200)),
                         (px, py, pw, ph), 2)

        # Title
        title = font_med.render(regime, True, REGIME_COLORS.get(regime, COLORS["fallback_regime"]))
        screen.blit(title, (px + VIEWER_PANEL_MARGIN, py + 3))

        # Households as dots
        h_area_x = px + 2 * VIEWER_PANEL_MARGIN
        h_area_y = py + VIEWER_STATUS_OFFSET_Y
        h_area_w = pw - VIEWER_HOUSEHOLD_AREA_RIGHT_MARGIN
        h_area_h = ph - VIEWER_HOUSEHOLD_AREA_BOTTOM_MARGIN

        households = snapshot["households"]
        cols = VIEWER_HOUSEHOLD_COLS
        for idx, h in enumerate(households):
            row = idx // cols
            col = idx % cols
            x = h_area_x + col * (h_area_w // cols)
            y = h_area_y + row * (h_area_h // max(1, len(households) // cols + 1))
            color = COLORS.get(h["wealth_class"], COLORS["fallback_household"])
            pygame.draw.circle(screen, color, (int(x), int(y)), VIEWER_HOUSEHOLD_DOT_RADIUS)
            if h["health"] < VIEWER_HEALTH_RING_THRESHOLD:
                pygame.draw.circle(screen, COLORS["health_ring"],
                                   (int(x), int(y)), VIEWER_HEALTH_RING_RADIUS, 1)

        # Firms as nodes
        firms = snapshot["firms"]
        firm_x = px + pw - VIEWER_FIRM_X_OFFSET
        for j, f in enumerate(firms):
            fy = py + VIEWER_FIRM_Y_START + j * VIEWER_FIRM_Y_SPACING
            size = max(VIEWER_FIRM_SIZE_MIN, min(VIEWER_FIRM_SIZE_MAX, int(f["capacity"] * VIEWER_FIRM_SIZE_SCALE)))
            color_val = min(255, max(0, int(f["price"] * VIEWER_FIRM_COLOR_SCALE)))
            firm_color = (color_val, 100, 255 - color_val)
            pygame.draw.circle(screen, firm_color, (firm_x, fy), size)
            pygame.draw.circle(screen, COLORS["firm_outline"],
                               (firm_x, fy), size, 1)
            label = font_small.render(f"p={f['price']:.2f}", True, COLORS["text"])
            screen.blit(label, (firm_x + size + 3, fy - 6))

        # Macro overlay
        m = snapshot["macro"]
        overlay_y = py + ph - VIEWER_OVERLAY_OFFSET_Y
        metrics_text = (f"GDP:{m['gdp']:.2f} Gini:{m['gini']:.2f} "
                        f"HP:{m['avg_health']:.2f} Inf:{m['inflation']:.3f} "
                        f"F:{m['num_firms']}")
        overlay = font_small.render(metrics_text, True, COLORS["text"])
        screen.blit(overlay, (px + VIEWER_PANEL_MARGIN, overlay_y))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MARL Economy Viewer")
    parser.add_argument("--weeks", type=int, default=DEFAULT_EPISODE_WEEKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    viewer = SimViewer(args.weeks, args.seed)
    if args.headless:
        viewer.run_headless(args.weeks)
    else:
        viewer.run_pygame()


if __name__ == "__main__":
    main()
