"""
Parameter-estimation experiment for VS-iCNLR.

This script mirrors the synthetic parameter-estimation module in de Carvalho
et al. (2021), with an added variable-selection target:

1. Generate interval-valued synthetic data under known cluster-specific
   center/radius functions.
2. Add useless interval-valued explanatory variables after the informative
   variables.
3. Fit VSICNLR.
4. Align estimated clusters to true clusters.
5. Compute parameter RMSE and variable-selection metrics.
   Radius parameters are compared on the log-radius scale, matching the
   VSICNLR radius model.

    By default, this parameter-estimation script runs scenarios 1--24,
    covering both linear and nonlinear center/radius cases.  A non-empty
    optimizer list is required for the nonlinear candidate functions.

    The script writes two CSV files:

- out/vs_icnlr_parameter_experiment/replicate_results.csv
- out/vs_icnlr_parameter_experiment/scenario_summary.csv

Run:
    python3 -B run_vs_icnlr_parameter_experiment.py
"""

from __future__ import annotations

import csv
import warnings
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vs_icnlr import FittedModel, VSICNLR
from vs_icnlr_simulation_data import SimulationData, generate_vs_icnlr_data, scenario_grid

#把聚类标签和真实标签映射起来
def align_clusters(true_labels: np.ndarray, estimated_labels: np.ndarray, n_clusters: int) -> Dict[int, int]:
    """
    Align estimated cluster ids to true cluster ids by maximizing label matches.

    Returns a dictionary: estimated_cluster -> true_cluster.
    """
    best_mapping: Dict[int, int] = {}
    best_score = -1
    for perm in permutations(range(n_clusters)):
        mapping = {estimated: true for estimated, true in enumerate(perm)}
        aligned = np.array([mapping[int(label)] for label in estimated_labels], dtype=int)
        score = int(np.sum(aligned == true_labels))
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping

#计算聚类准确率
def clustering_accuracy(true_labels: np.ndarray, estimated_labels: np.ndarray, mapping: Dict[int, int]) -> float:
    aligned = np.array([mapping[int(label)] for label in estimated_labels], dtype=int)
    return float(np.mean(aligned == true_labels))

#把估计参数补充成完整长度
def full_linear_beta(model: FittedModel, n_features: int) -> Optional[np.ndarray]:
    """Expand a fitted linear model's beta to [intercept, all feature betas]."""
    if model.function.name != "linear":
        return None
    beta = np.asarray(model.beta, dtype=float)
    full = np.zeros(n_features + 1, dtype=float)
    full[0] = beta[0]
    selected = np.where(model.feature_mask)[0]
    if len(selected) != len(beta) - 1:
        return None
    full[selected + 1] = beta[1:]
    return full


def true_linear_beta(params: Dict[str, object], n_features: int) -> np.ndarray:
    """Expand true linear beta to [intercept, informative betas, noise zeros]."""
    beta = np.zeros(n_features + 1, dtype=float)
    beta[0] = float(params["intercept"])
    true_beta = np.asarray(params["beta"], dtype=float)
    beta[1 : 1 + len(true_beta)] = true_beta
    return beta

def nonlinear_beta(model: FittedModel, n_features: int) -> Optional[np.ndarray]:
    """Expand supported nonlinear parameter estimates to a comparable vector."""
    if model.function.name == "additive_paper_f1":
        beta = np.asarray(model.beta, dtype=float)
        full = np.zeros(n_features + 3, dtype=float)
        full[:3] = beta[:3]
        selected = np.where(model.feature_mask)[0]
        if len(selected) != len(beta) - 3:
            return None
        full[3 + selected] = beta[3:]
        return full
    if model.function.name != "paper_f1":
        return None
    if model.n_selected_features != 1:
        return None
    return np.asarray(model.beta, dtype=float)


def true_nonlinear_beta(params: Dict[str, object], n_features: int) -> np.ndarray:
    if "weights" in params:
        beta = np.zeros(n_features + 3, dtype=float)
        beta[0] = float(params["intercept"])
        beta[1] = float(params["beta0"])
        beta[2] = float(params.get("eta1", np.log(float(params["beta1"]))))
        weights = np.asarray(params["weights"], dtype=float)
        beta[3 : 3 + len(weights)] = weights
        return beta
    return np.array([float(params["beta0"]), float(params["beta1"])], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def finite_mean(values: Sequence[float]) -> float:
    finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))

#计算参数估计误差
def model_parameter_rmse(
    model: FittedModel,
    true_params: Dict[str, object],
    n_features: int,
) -> float:
    if true_params["type"] == "linear":
        estimated = full_linear_beta(model, n_features)
        if estimated is None:
            return float("nan")
        truth = true_linear_beta(true_params, n_features)
        return rmse(estimated, truth)

    estimated = nonlinear_beta(model, n_features)
    if estimated is None:
        return float("nan")
    truth = true_nonlinear_beta(true_params, n_features)
    return rmse(estimated, truth)

#TPR：真实有用变量中，有多少被选出来了；
#FPR：真实无用变量中，有多少被错误选出来了；
#exact：是否变量集合完全正确。
def selection_metrics(selected_mask: np.ndarray, true_mask: np.ndarray) -> Tuple[float, float, float]:
    selected_mask = np.asarray(selected_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    tp = np.sum(selected_mask & true_mask)
    fp = np.sum(selected_mask & ~true_mask)
    fn = np.sum(~selected_mask & true_mask)
    tn = np.sum(~selected_mask & ~true_mask)

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    exact = float(np.array_equal(selected_mask, true_mask))
    return float(tpr), float(fpr), exact

#单次重复实验的评价
def evaluate_replicate(
    data: SimulationData,
    model: VSICNLR,
    scenario_id: int,
    replicate_id: int,
) -> Dict[str, object]:
    n_clusters = int(data.metadata["n_clusters"])
    n_features = data.X_lower.shape[1]
    mapping = align_clusters(data.labels, model.labels_, n_clusters)

    center_rmse_values = []
    radius_rmse_values = []
    center_tpr_values = []
    center_fpr_values = []
    center_exact_values = []
    radius_tpr_values = []
    radius_fpr_values = []
    radius_exact_values = []

    for estimated_cluster, true_cluster in mapping.items():
        cluster_model = model.models_[estimated_cluster]
        center_true = data.metadata["center_params"][true_cluster]
        radius_true = data.metadata["radius_params"][true_cluster]

        center_rmse_values.append(model_parameter_rmse(cluster_model.center, center_true, n_features))
        radius_rmse_values.append(model_parameter_rmse(cluster_model.radius, radius_true, n_features))

        center_tpr, center_fpr, center_exact = selection_metrics(
            cluster_model.center.feature_mask,
            data.center_informative_mask,
        )
        radius_tpr, radius_fpr, radius_exact = selection_metrics(
            cluster_model.radius.feature_mask,
            data.radius_informative_mask,
        )
        center_tpr_values.append(center_tpr)
        center_fpr_values.append(center_fpr)
        center_exact_values.append(center_exact)
        radius_tpr_values.append(radius_tpr)
        radius_fpr_values.append(radius_fpr)
        radius_exact_values.append(radius_exact)

    return {
        "scenario_id": scenario_id,
        "replicate_id": replicate_id,
        "n_clusters": n_clusters,
        "relation": data.metadata["relation"],
        "center_model": data.metadata["center_model"],
        "radius_model": data.metadata["radius_model"],
        "objective": model.objective_,
        "sse": model.sse_,
        "clustering_accuracy": clustering_accuracy(data.labels, model.labels_, mapping),
        "center_param_rmse": finite_mean(center_rmse_values),
        "radius_param_rmse": finite_mean(radius_rmse_values),
        "center_selection_tpr": float(np.mean(center_tpr_values)),
        "center_selection_fpr": float(np.mean(center_fpr_values)),
        "center_selection_exact": float(np.mean(center_exact_values)),
        "radius_selection_tpr": float(np.mean(radius_tpr_values)),
        "radius_selection_fpr": float(np.mean(radius_fpr_values)),
        "radius_selection_exact": float(np.mean(radius_exact_values)),
    }


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(row)

    summaries = []
    metric_names = [
        "objective",
        "sse",
        "clustering_accuracy",
        "center_param_rmse",
        "radius_param_rmse",
        "center_selection_tpr",
        "center_selection_fpr",
        "center_selection_exact",
        "radius_selection_tpr",
        "radius_selection_fpr",
        "radius_selection_exact",
    ]
    for scenario_id, scenario_rows in sorted(grouped.items()):
        first = scenario_rows[0]
        summary = {
            "scenario_id": scenario_id,
            "n_clusters": first["n_clusters"],
            "relation": first["relation"],
            "center_model": first["center_model"],
            "radius_model": first["radius_model"],
            "n_replicates": len(scenario_rows),
        }
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in scenario_rows], dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                summary[f"{metric}_mean"] = float("nan")
                summary[f"{metric}_sd"] = float("nan")
            else:
                summary[f"{metric}_mean"] = float(np.mean(finite))
                summary[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_parameter_experiment(
    n_replicates: int = 10,
    n_samples: int = 120,
    n_informative: int = 2,
    n_noise: int = 4,
    selection_scope: str = "group",
    scenario_ids: Optional[Iterable[int]] = None,
    optimizers: Sequence[str] = ("RANDOM",),
    output_dir: Path = Path("out") / "vs_icnlr_parameter_experiment",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    scenarios = scenario_grid()
    if scenario_ids is None:
        scenario_ids = range(1, 25)
    selected_scenario_ids = set(int(scenario_id) for scenario_id in scenario_ids)

    for scenario_id, scenario in enumerate(scenarios, start=1):
        if scenario_id not in selected_scenario_ids:
            continue
        for replicate_id in range(1, n_replicates + 1):
            seed = 310000 + scenario_id * 1000 + replicate_id
            data = generate_vs_icnlr_data(
                n_samples=n_samples,
                n_informative=n_informative,
                n_noise=n_noise,
                random_state=seed,
                **scenario,
            )
            model = VSICNLR(
                n_clusters=int(scenario["n_clusters"]),
                selection_scope=selection_scope,
                selection_criterion="penalty",
                lambda_center=0.05,
                lambda_radius=0.05,
                max_selected_features=n_informative,
                max_feature_subsets=100,
                optimizers=optimizers,
                n_init=15,
                max_iter=30,
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(data.X_lower, data.X_upper, data.y_lower, data.y_upper)
            rows.append(evaluate_replicate(data, model, scenario_id, replicate_id))
            print(
                f"scenario={scenario_id:02d} replicate={replicate_id:02d} "
                f"acc={rows[-1]['clustering_accuracy']:.3f} "
                f"center_rmse={rows[-1]['center_param_rmse']:.4f} "
                f"radius_rmse={rows[-1]['radius_param_rmse']:.4f}"
            )

    summaries = summarize(rows)
    write_csv(rows, output_dir / "replicate_results.csv")
    write_csv(summaries, output_dir / "scenario_summary.csv")
    return rows, summaries


def main() -> None:
    rows, summaries = run_parameter_experiment()
    print(f"\nFinished {len(rows)} replicate runs.")
    print("Summary rows:", len(summaries))
    print("Output directory: out/vs_icnlr_parameter_experiment")


if __name__ == "__main__":
    main()
