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
- **Configurations:** operation inventory `+ - * / sqrt exp log sin cos`. The operand penalty `lambda_op`
  is 1e-10 (the default) or 1e-3. Everything else uses jessaminescikitlearn defaults. There is no
  hyperparameter tuning, as in the reference runs.
- **Score:** test R² of the returned model, or 0 if the model has more than 50 SymPy nodes. This is the
  rule in `eggp_paper_GECCO/srbench_experiment/perfplot.py`.
- **Statistics:** for each dataset, the median score with an order-statistic 95% confidence interval and
  a Mann-Whitney test against each reference method. Across datasets, a Wilcoxon signed-rank test on the
  medians.

## Caveats

- The reference methods ran under budgets set by iterations or generations (about 1–30 min per run),
  not a fixed 600 s.
- Model size is counted on the SymPy expression. Each reference method reports its own size measure.
- 9 runs per dataset detect large differences but not small ones.

## Running

Actions → *Jessamine preview benchmark* → Run workflow. `bench/run_one.py` runs a single fit, and
`bench/summarize.py` builds the results table.
