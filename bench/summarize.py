"""Summarize Jessamine runs against the published eggp_paper_GECCO results.

Usage: python bench/summarize.py results/*.jsonl > summary.md

For each dataset and lambda_op, reports Jessamine's median score with a
distribution-free confidence interval for the median (order statistics), and
compares it with each reference method by a two-sided Mann-Whitney U test.
Across datasets, compares per-dataset medians by a Wilcoxon signed-rank test.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu, wilcoxon

BASELINE_URL = (
    "https://raw.githubusercontent.com/folivetti/eggp_paper_GECCO/HEAD/"
    "srbench_experiment/r2_perf.csv"
)
METHODS = ["Operon", "PySR", "eggp_mo", "eggp_so", "tinyGP"]
ALPHA = 0.05


def median_ci(values, level=0.95):
    """Order-statistic CI for the median: the widest symmetric (k, n-k+1) pair
    whose coverage is still at least `level`. Returns (lo, hi, coverage)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    best = None
    for k in range(1, n // 2 + 1):
        coverage = 1 - 2 * binom.cdf(k - 1, n, 0.5)
        if coverage >= level:
            best = (x[k - 1], x[n - k], coverage)
    return best if best else (x[0], x[-1], 1 - 2 * 0.5**n)


def main(paths):
    runs = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    runs["dataset_id"] = runs["dataset"].str.split("_").str[0].astype(int)
    base = pd.read_csv(BASELINE_URL)

    print("# Jessamine preview on the SRBench 2.0 black-box selection\n")
    print(
        f"{len(runs)} Jessamine runs. Score = test R² (0 if the model has more than"
        " 50 nodes or the run returned no model), as in eggp_paper_GECCO. Reference methods: 30 runs each"
        " (published results).\n"
    )

    for config, jl in runs.groupby("config"):
        print(f"## {config} (lambda_op = {jl['lambda_op'].iloc[0]:g})\n")
        print(
            "| dataset | Jessamine median [CI] | runs | failed | over size cap | "
            + " | ".join(METHODS)
            + " |"
        )
        print("|---|---|---|---|---|" + "---|" * len(METHODS))
        medians = {m: [] for m in ["Jessamine"] + METHODS}
        record = {m: [0, 0, 0] for m in METHODS}  # better, tied, worse
        for ds, j in jl.groupby("dataset"):
            ds_id = j["dataset_id"].iloc[0]
            scores = j["score"].to_numpy()
            lo, hi, cov = median_ci(scores)
            med = float(np.median(scores))
            medians["Jessamine"].append(med)
            cells = []
            for m in METHODS:
                b = base[(base.algorithm == m) & (base.dataset == ds_id)]["r2_test"]
                b_med = float(np.median(b))
                medians[m].append(b_med)
                p = mannwhitneyu(scores, b, alternative="two-sided").pvalue
                if p < ALPHA:
                    mark = "▲" if med > b_med else "▼"
                    record[m][0 if med > b_med else 2] += 1
                else:
                    mark = "≈"
                    record[m][1] += 1
                cells.append(f"{b_med:.3f} {mark}")
            over = int((j["size"] > 50).sum())
            failed = int(j["failed"].sum())
            print(
                f"| {ds} | {med:.3f} [{lo:.3f}, {hi:.3f}] | {len(j)} | {failed} | {over} | "
                + " | ".join(cells)
                + " |"
            )
        print(
            f"\nCI: order-statistic interval for the median, coverage ≥ 95%"
            f" (exact {cov:.1%} for n = {len(j)} runs). ▲/▼: Jessamine significantly"
            f" better/worse than that method (Mann-Whitney, p < {ALPHA}); ≈: no"
            " significant difference.\n"
        )
        print(
            f"| vs | better / tied / worse datasets | Wilcoxon p"
            f" (medians, {len(medians['Jessamine'])} datasets) |"
        )
        print("|---|---|---|")
        for m in METHODS:
            diff = np.array(medians["Jessamine"]) - np.array(medians[m])
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            b, t, w = record[m]
            print(f"| {m} | {b} / {t} / {w} | {p:.3f} |")
        print(
            f"\nJessamine median size {jl['size'].median():.0f} nodes; median fit time"
            f" {jl['fit_seconds'].median():.0f} s.\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
