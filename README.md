# Jessamine preview on the SRBench 2.0 black-box selection

A small, unofficial preview of how [Jessamine](https://github.com/wgm-applied-math/Jessamine.jl)
(via [jessaminescikitlearn](https://github.com/wgm-applied-math/jessaminescikitlearn) v0.3.0) compares with
other symbolic regression methods. It uses the 12 black-box datasets selected for SRBench 2.0
([Imai Aldeia et al., 2025](https://arxiv.org/abs/2505.03977)). The train/test splits, scoring rule and
reference results come from de França & Kronberger's
[eggp_paper_GECCO](https://github.com/folivetti/eggp_paper_GECCO), which covers Operon, PySR, eggp and
tinyGP with 30 runs each.

This is not an SRBench result. It is a quick check run on GitHub Actions.

## Protocol

- **Data:** the 12 datasets in `eggp_paper_GECCO/srbench_experiment/datasets`, 3 train/test folds each,
  downloaded at run time.
- **Runs:** 3 per fold (9 per dataset) per configuration. Each fit is limited to 600 s on one CPU core.
- **Configurations:** operation inventory `+ - * / sqrt exp log sin cos`, with `lambda_op` = 1e-3.
  There is no hyperparameter tuning, as in the reference runs.
  - `default`: jessaminescikitlearn's genome sizes, which grow with the number of features.
  - `compact`: `output_size` 3, `scratch_size` 2, `num_time_steps` 2. Smaller models, to respect the
    50-node cap, as the reference methods were configured to (e.g. PySR `maxsize=50`).
- **Score:** test R² of the returned model, or 0 if the model has more than 50 SymPy nodes or the run
  returned no model. This is the rule in `eggp_paper_GECCO/srbench_experiment/perfplot.py`, with failures
  counted as 0.
- **Statistics:** for each dataset, the median score with an order-statistic 95% confidence interval and
  a Mann-Whitney test against each reference method. Across datasets, a Wilcoxon signed-rank test on the
  medians.

## Caveats

- The reference methods ran under budgets set by iterations or generations (about 1–30 min per run),
  not a fixed 600 s.
- Model size is counted on the SymPy expression. Each reference method reports its own size measure.
- 9 runs per dataset detect large differences but not small ones.

## Runs

1. **Preview, 2026-10-24** ([run](https://github.com/riordanaa/jessamine-srbench-preview/actions/runs/40141043758),
   [report](reports/2026-10-24-preview.md)): 9 runs per dataset, `default` and `compact` with
   `lambda_op` = 1e-3.
   - `compact` was not significantly different from PySR, tinyGP, eggp_so or Operon across the 12
     datasets, and was significantly worse than eggp_mo.
   - 24 of 108 `compact` runs scored 0 for exceeding the 50-node cap.
   - The `compact` configuration was chosen after a smoke run on these same datasets.
2. **Confirmation, setup fixed before running** (2026-10-25): one configuration, chosen before seeing
   any results with it.
   - `compact` genome with `lambda_op` = 1e-2: the operand penalty the Jessamine article recommends for
     noisy data, based on its sin experiment.
   - 10 runs per fold, so 30 per dataset (as in the reference results), 600 s each.
   - No other changes.

## Running

Actions → *Jessamine preview benchmark* → Run workflow. `bench/run_one.py` runs a single fit, and
`bench/summarize.py` builds the results table.
