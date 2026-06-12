"""
Regime evaluation: run all 4 regimes, produce comparison tables,
similarity scores, and rankings (Section 14.7).
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import DEFAULT_EPISODE_WEEKS, DEFAULT_RANDOM_SEED, REGIME_NAMES
from marl_economy.scripts.run_one_episode import run_episode


def eval_all_regimes(episode_weeks: int = DEFAULT_EPISODE_WEEKS, seed: int = DEFAULT_RANDOM_SEED,
                     base_dir: str = None) -> dict:
    """Run all 4 regimes and produce comparison outputs."""
    regimes = list(REGIME_NAMES)

    if base_dir is None:
        base_dir = f"runs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)

    summaries = {}
    for regime in regimes:
        print(f"\n{'='*60}")
        print(f"Running regime: {regime}")
        print(f"{'='*60}")
        output_dir = f"{base_dir}/{regime}"
        summary = run_episode(regime, episode_weeks, seed, output_dir)
        summaries[regime] = summary

    # Regime summary table.
    csv_path = f"{base_dir}/reports/regime_summary.csv"
    fields = ["regime", "mean_gini", "mean_health", "mean_inflation",
              "mean_poverty", "mean_crime", "final_gdp", "firm_entries", "firm_exits"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for regime, s in summaries.items():
            writer.writerow({k: s.get(k, "") for k in fields})

    # Pretty print
    print(f"\n{'='*60}")
    print("REGIME COMPARISON")
    print(f"{'='*60}")
    print(f"{'Regime':<14} {'GDP':>8} {'Gini':>7} {'Health':>8} "
          f"{'Infltn':>8} {'Poverty':>8} {'Crime':>8}")
    print("-" * 65)
    for regime, s in summaries.items():
        print(f"{regime:<14} {s['final_gdp']:8.4f} {s['mean_gini']:7.4f} "
              f"{s['mean_health']:8.4f} {s['mean_inflation']:8.4f} "
              f"{s['mean_poverty']:8.4f} {s['mean_crime']:8.4f}")

    # Pairwise similarity scores between regimes.
    sim_scores = compute_similarity_scores(summaries)
    sim_path = f"{base_dir}/reports/similarity_scores.csv"
    with open(sim_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(regimes))
        for r1 in regimes:
            row = [r1] + [f"{sim_scores[r1][r2]:.4f}" for r2 in regimes]
            writer.writerow(row)

    print(f"\nSimilarity scores saved to {sim_path}")

    # Composite rankings.
    rankings = compute_rankings(summaries)
    rank_path = f"{base_dir}/reports/rankings.json"
    with open(rank_path, "w") as f:
        json.dump(rankings, f, indent=2)

    print(f"Rankings saved to {rank_path}")
    print(f"\nAll outputs in {base_dir}/")

    return summaries


def compute_similarity_scores(summaries: dict) -> dict:
    """
    Compute pairwise similarity between regimes based on normalized metrics.
    Uses cosine similarity on the metric vectors.
    """
    metrics_keys = ["mean_gini", "mean_health", "mean_inflation",
                    "mean_poverty", "mean_crime", "final_gdp"]

    regimes = list(summaries.keys())
    vectors = {}
    for regime in regimes:
        s = summaries[regime]
        vectors[regime] = np.array([s.get(k, 0.0) for k in metrics_keys])

    # Normalize
    all_vecs = np.array(list(vectors.values()))
    means = all_vecs.mean(axis=0)
    stds = all_vecs.std(axis=0) + 1e-8
    for regime in regimes:
        vectors[regime] = (vectors[regime] - means) / stds

    # Cosine similarity
    scores = {}
    for r1 in regimes:
        scores[r1] = {}
        for r2 in regimes:
            v1 = vectors[r1]
            v2 = vectors[r2]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            scores[r1][r2] = float(cos_sim)

    return scores


def compute_rankings(summaries: dict) -> dict:
    """
    Rank regimes under different weight schemes (Section 14.7).
    """
    regimes = list(summaries.keys())

    # Weight schemes
    schemes = {
        "equal_weights": {
            "final_gdp": 1.0, "mean_gini": -1.0, "mean_health": 1.0,
            "mean_inflation": -1.0, "mean_poverty": -1.0, "mean_crime": -1.0,
        },
        "social_priority": {
            "final_gdp": 0.5, "mean_gini": -2.0, "mean_health": 2.0,
            "mean_inflation": -0.5, "mean_poverty": -2.0, "mean_crime": -1.0,
        },
        "economic_priority": {
            "final_gdp": 2.0, "mean_gini": -0.5, "mean_health": 0.5,
            "mean_inflation": -2.0, "mean_poverty": -0.5, "mean_crime": -0.5,
        },
    }

    rankings = {}
    for scheme_name, weights in schemes.items():
        scores = {}
        for regime in regimes:
            s = summaries[regime]
            score = sum(weights[k] * s.get(k, 0.0) for k in weights)
            scores[regime] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings[scheme_name] = [
            {"rank": i + 1, "regime": r, "score": round(s, 4)}
            for i, (r, s) in enumerate(ranked)
        ]

    return rankings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate all regimes")
    parser.add_argument("--weeks", type=int, default=DEFAULT_EPISODE_WEEKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    eval_all_regimes(args.weeks, args.seed, args.output)


if __name__ == "__main__":
    main()
