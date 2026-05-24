"""
Lambda tuning experiment for VS-iCNLR.

The script tunes lambda_center and lambda_radius by validation error.  For each
scenario and replicate, one synthetic data set is generated, split into
training/validation parts, and every lambda pair is evaluated on the same split.

Main score:
    0.5 * MSE_center_val / Var(center_train)
  + 0.5 * MSE_log_radius_val / Var(log_radius_train)

Run examples:
    python3 -B run_vs_icnlr_lambda_experiment.py --scenario-group nonlinear_nonlinear
    python3 -B run_vs_icnlr_lambda_experiment.py --scenario-ids 19 20 21
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from run_vs_icnlr_parameter_experiment import (
    candidate_functions_for_model,
    selection_metrics,
)
from vs_icnlr import VSICNLR, interval_to_center_radius
from vs_icnlr_simulation_data import SimulationData, generate_vs_icnlr_data, scenario_grid


SCENARIO_GROUPS: Dict[str, Tuple[int, ...]] = {
    "linear_linear": tuple(range(1, 7)),
    "linear_nonlinear": tuple(range(7, 13)),
    "nonlinear_linear": tuple(range(13, 19)),
    "nonlinear_nonlinear": tuple(range(19, 25)),
    "all": tuple(range(1, 25)),
}


def parse_float_grid(value: str) -> Tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("Lambda grid values must be non-negative.")
    return values


def train_validation_indices(
    labels: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stratified train/validation split so every true cluster appears in train."""
    labels = np.asarray(labels, dtype=int)
    train_parts: List[np.ndarray] = []
    validation_parts: List[np.ndarray] = []

    for cluster in np.unique(labels):
        cluster_indices = np.where(labels == cluster)[0]
        shuffled = cluster_indices.copy()
        rng.shuffle(shuffled)
        n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
        n_validation = min(n_validation, len(shuffled) - 1)
        validation_parts.append(shuffled[:n_validation])
        train_parts.append(shuffled[n_validation:])

    train_index = np.concatenate(train_parts)
    validation_index = np.concatenate(validation_parts)
    rng.shuffle(train_index)
    rng.shuffle(validation_index)
    return train_index, validation_index


def subset_data(data: SimulationData, index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return data.X_lower[index], data.X_upper[index], data.y_lower[index], data.y_upper[index]


def validation_score(
    data: SimulationData,
    model: VSICNLR,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    center_weight: float = 0.5,
    radius_weight: float = 0.5,
) -> Dict[str, float]:
    """Assign validation observations to the best fitted cluster and score both responses."""
    Xc_train, Xr_train = data.X_center[train_index], data.X_radius[train_index]
    yc_train, yr_train = data.y_center[train_index], data.y_radius[train_index]
    Xc_val, Xr_val = data.X_center[validation_index], data.X_radius[validation_index]
    yc_val, yr_val = data.y_center[validation_index], data.y_radius[validation_index]

    center_var = float(np.var(yc_train, ddof=1))
    radius_var = float(np.var(np.log(np.maximum(yr_train, 1e-8)), ddof=1))
    center_var = max(center_var, 1e-12)
    radius_var = max(radius_var, 1e-12)

    total_loss = np.empty((len(validation_index), model.n_clusters), dtype=float)
    center_loss = np.empty_like(total_loss)
    radius_loss = np.empty_like(total_loss)

    for k, cluster_model in enumerate(model.models_):
        center_loss[:, k] = cluster_model.center.loss(Xc_val, yc_val)
        radius_loss[:, k] = cluster_model.radius.loss(Xr_val, yr_val)
        total_loss[:, k] = center_loss[:, k] + radius_loss[:, k]

    assigned = np.argmin(total_loss, axis=1)
    rows = np.arange(len(validation_index))
    center_mse = float(np.mean(center_loss[rows, assigned]))
    radius_log_mse = float(np.mean(radius_loss[rows, assigned]))
    center_standardized = center_mse / center_var
    radius_standardized = radius_log_mse / radius_var
    score = center_weight * center_standardized + radius_weight * radius_standardized

    return {
        "validation_score": float(score),
        "validation_center_mse": center_mse,
        "validation_radius_log_mse": radius_log_mse,
        "validation_center_standardized": float(center_standardized),
        "validation_radius_log_standardized": float(radius_standardized),
    }


def average_selection_metrics(model: VSICNLR, data: SimulationData) -> Dict[str, float]:
    center_values = [
        selection_metrics(cluster_model.center.feature_mask, data.center_informative_mask)
        for cluster_model in model.models_
    ]
    radius_values = [
        selection_metrics(cluster_model.radius.feature_mask, data.radius_informative_mask)
        for cluster_model in model.models_
    ]
    center = np.asarray(center_values, dtype=float)
    radius = np.asarray(radius_values, dtype=float)
    return {
        "center_selection_tpr": float(np.mean(center[:, 0])),
        "center_selection_fpr": float(np.mean(center[:, 1])),
        "center_selection_exact": float(np.mean(center[:, 2])),
        "radius_selection_tpr": float(np.mean(radius[:, 0])),
        "radius_selection_fpr": float(np.mean(radius[:, 1])),
        "radius_selection_exact": float(np.mean(radius[:, 2])),
    }


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, float, float], List[Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["scenario_id"]), float(row["lambda_center"]), float(row["lambda_radius"]))
        grouped.setdefault(key, []).append(row)

    metric_names = [
        "validation_score",
        "validation_center_standardized",
        "validation_radius_log_standardized",
        "validation_center_mse",
        "validation_radius_log_mse",
        "train_objective",
        "train_sse",
        "center_selection_tpr",
        "center_selection_fpr",
        "center_selection_exact",
        "radius_selection_tpr",
        "radius_selection_fpr",
        "radius_selection_exact",
    ]

    summaries: List[Dict[str, object]] = []
    for (scenario_id, lambda_center, lambda_radius), group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summary: Dict[str, object] = {
            "scenario_id": scenario_id,
            "n_clusters": first["n_clusters"],
            "relation": first["relation"],
            "center_model": first["center_model"],
            "radius_model": first["radius_model"],
            "lambda_center": lambda_center,
            "lambda_radius": lambda_radius,
            "n_replicates": len(group_rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                summary[f"{metric}_mean"] = float("nan")
                summary[f"{metric}_sd"] = float("nan")
            else:
                summary[f"{metric}_mean"] = float(np.mean(finite))
                summary[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        summaries.append(summary)
    return summaries


def best_lambda_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(row)

    best_rows = []
    for scenario_id, rows in sorted(grouped.items()):
        finite_rows = [
            row for row in rows if np.isfinite(float(row["validation_score_mean"]))
        ]
        if not finite_rows:
            continue
        best = min(
            finite_rows,
            key=lambda row: (
                float(row["validation_score_mean"]),
                float(row["lambda_center"]) + float(row["lambda_radius"]),
            ),
        )
        best_rows.append(dict(best))
    return best_rows


def selected_scenario_ids(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.scenario_ids:
        return tuple(int(value) for value in args.scenario_ids)
    return SCENARIO_GROUPS[args.scenario_group]


def run_lambda_experiment(
    scenario_ids: Iterable[int],
    lambda_center_grid: Sequence[float],
    lambda_radius_grid: Sequence[float],
    n_replicates: int = 10,
    n_samples: int = 100,
    n_informative: int = 1,
    n_noise: int = 2,
    validation_fraction: float = 0.2,
    selection_scope: str = "group",
    optimizers: Sequence[str] = ("L-BFGS-B",),
    n_init: int = 3,
    max_iter: int = 10,
    max_feature_subsets: int = 100,
    output_dir: Path = Path("out") / "vs_icnlr_lambda_experiment",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    scenarios = scenario_grid()
    selected_ids = set(int(scenario_id) for scenario_id in scenario_ids)

    for scenario_id, scenario in enumerate(scenarios, start=1):
        if scenario_id not in selected_ids:
            continue
        for replicate_id in range(1, n_replicates + 1):
            seed = 710000 + scenario_id * 1000 + replicate_id
            split_rng = np.random.default_rng(seed + 137)
            data = generate_vs_icnlr_data(
                n_samples=n_samples,
                n_informative=n_informative,
                n_noise=n_noise,
                random_state=seed,
                **scenario,
            )
            train_index, validation_index = train_validation_indices(
                data.labels,
                validation_fraction=validation_fraction,
                rng=split_rng,
            )
            X_lower_train, X_upper_train, y_lower_train, y_upper_train = subset_data(data, train_index)

            for lambda_center in lambda_center_grid:
                for lambda_radius in lambda_radius_grid:
                    model = VSICNLR(
                        n_clusters=int(scenario["n_clusters"]),
                        selection_scope=selection_scope,
                        selection_criterion="penalty",
                        lambda_center=float(lambda_center),
                        lambda_radius=float(lambda_radius),
                        max_selected_features=n_informative,
                        max_feature_subsets=max_feature_subsets,
                        optimizers=optimizers,
                        center_candidate_functions=candidate_functions_for_model(
                            str(scenario["center_model"])
                        ),
                        radius_candidate_functions=candidate_functions_for_model(
                            str(scenario["radius_model"])
                        ),
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=seed,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        model.fit(X_lower_train, X_upper_train, y_lower_train, y_upper_train)

                    score_values = validation_score(data, model, train_index, validation_index)
                    selection_values = average_selection_metrics(model, data)
                    row: Dict[str, object] = {
                        "scenario_id": scenario_id,
                        "replicate_id": replicate_id,
                        "n_clusters": scenario["n_clusters"],
                        "relation": scenario["relation"],
                        "center_model": scenario["center_model"],
                        "radius_model": scenario["radius_model"],
                        "lambda_center": float(lambda_center),
                        "lambda_radius": float(lambda_radius),
                        "n_train": int(len(train_index)),
                        "n_validation": int(len(validation_index)),
                        "train_objective": float(model.objective_),
                        "train_sse": float(model.sse_),
                        **score_values,
                        **selection_values,
                    }
                    rows.append(row)
                    print(
                        f"scenario={scenario_id:02d} rep={replicate_id:02d} "
                        f"lambda_c={lambda_center:g} lambda_r={lambda_radius:g} "
                        f"score={row['validation_score']:.4f}"
                    )

    summary_rows = summarize(rows)
    best_rows = best_lambda_rows(summary_rows)
    write_csv(rows, output_dir / "lambda_replicate_results.csv")
    write_csv(summary_rows, output_dir / "lambda_summary.csv")
    write_csv(best_rows, output_dir / "lambda_best_by_scenario.csv")
    return rows, summary_rows, best_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune lambda_center/lambda_radius for VS-iCNLR.")
    parser.add_argument(
        "--scenario-group",
        choices=sorted(SCENARIO_GROUPS),
        default="nonlinear_nonlinear",
        help="Scenario group to run when --scenario-ids is not provided.",
    )
    parser.add_argument(
        "--scenario-ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit scenario ids. Overrides --scenario-group.",
    )
    parser.add_argument(
        "--lambda-center-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.001,0.003,0.01,0.03,0.1"),
        help="Comma-separated lambda_center values.",
    )
    parser.add_argument(
        "--lambda-radius-grid",
        type=parse_float_grid,
        default=parse_float_grid("0,0.0003,0.001,0.003,0.01,0.03"),
        help="Comma-separated lambda_radius values.",
    )
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-informative", type=int, default=1)
    parser.add_argument("--n-noise", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--selection-scope", choices=("group", "global"), default="group")
    parser.add_argument("--optimizers", nargs="+", default=["L-BFGS-B"])
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--max-feature-subsets", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out") / "vs_icnlr_lambda_experiment",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, summary_rows, best_rows = run_lambda_experiment(
        scenario_ids=selected_scenario_ids(args),
        lambda_center_grid=args.lambda_center_grid,
        lambda_radius_grid=args.lambda_radius_grid,
        n_replicates=args.n_replicates,
        n_samples=args.n_samples,
        n_informative=args.n_informative,
        n_noise=args.n_noise,
        validation_fraction=args.validation_fraction,
        selection_scope=args.selection_scope,
        optimizers=tuple(args.optimizers),
        n_init=args.n_init,
        max_iter=args.max_iter,
        max_feature_subsets=args.max_feature_subsets,
        output_dir=args.output_dir,
    )
    print(f"\nFinished {len(rows)} lambda replicate runs.")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Best-by-scenario rows: {len(best_rows)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
