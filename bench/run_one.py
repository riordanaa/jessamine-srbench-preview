"""Run Jessamine once on one fold of one dataset and print the result as JSON.

Datasets and scoring follow de Franca & Kronberger's eggp_paper_GECCO
(srbench_experiment/perfplot.py): the score of a run is the test R^2 of the
returned model, counted as 0 if the model has more than 50 nodes.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import sympy
from jessaminescikitlearn.Regression import Regressor
from sklearn.metrics import r2_score

MAX_SIZE = 50


def node_count(expr):
    """Nodes in the SymPy expression tree (operators, symbols, and constants)."""
    return sum(1 for _ in sympy.preorder_traversal(expr))


def r2(est, X, y):
    # A constant model predicts a scalar, so broadcast to one value per row.
    y_hat = np.broadcast_to(np.asarray(est.predict(X), dtype=float), y.shape)
    if not np.all(np.isfinite(y_hat)):
        return float("nan")
    return float(r2_score(y, y_hat))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("fold", type=int)
    parser.add_argument("run", type=int)
    parser.add_argument("--lambda-op", type=float, required=True)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--op-inventory", default="+ - * / sqrt exp log sin cos")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    train = pd.read_csv(f"{args.data_dir}/{args.dataset}_train{args.fold}.csv")
    test = pd.read_csv(f"{args.data_dir}/{args.dataset}_test{args.fold}.csv")
    X_train, y_train = train.drop(columns="target"), train["target"].to_numpy()
    X_test, y_test = test.drop(columns="target"), test["target"].to_numpy()

    est = Regressor(
        op_inventory=args.op_inventory,
        lambda_op=args.lambda_op,
        max_time=args.max_time,
        random_state=1000 * args.fold + args.run,
    )
    start = time.perf_counter()
    est.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    r2_train = r2(est, X_train, y_train)
    r2_test = r2(est, X_test, y_test)
    size = node_count(est.model_sympy())
    valid = size <= MAX_SIZE and np.isfinite(r2_test)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "fold": args.fold,
                "run": args.run,
                "lambda_op": args.lambda_op,
                "op_inventory": args.op_inventory,
                "max_time": args.max_time,
                "fit_seconds": round(fit_seconds, 1),
                "size": size,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "score": r2_test if valid else 0.0,
                "model": est.model_str(),
            }
        )
    )


if __name__ == "__main__":
    main()
