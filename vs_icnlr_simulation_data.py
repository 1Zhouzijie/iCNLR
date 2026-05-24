"""
Synthetic interval-valued data generator for VS-iCNLR experiments.

The design follows the simulation spirit of de Carvalho et al. (2021):

- generate interval-valued explanatory variables through centers and radii;
- create K latent clusters;
- control the cluster geometry as disjoint, intersecting, or overlapping;
- generate the response center and response radius from linear or nonlinear
  cluster-specific functions;
- reconstruct lower/upper interval bounds.

The VS-iCNLR extension adds irrelevant variables.  The first
``n_informative`` explanatory variables are useful and enter the response
generating functions.  The last ``n_noise`` explanatory variables are generated
in the same interval-valued format but do not affect the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

Array = np.ndarray
RelationType = Literal["disjoint", "intersecting", "overlapping"]
ModelType = Literal["linear", "nonlinear"]


@dataclass(frozen=True)
class SimulationData:
    X_lower: Array
    X_upper: Array
    y_lower: Array
    y_upper: Array
    labels: Array
    informative_mask: Array
    center_informative_mask: Array
    radius_informative_mask: Array
    X_center: Array
    X_radius: Array
    y_center: Array
    y_radius: Array
    metadata: Dict[str, object]


def paper_nonlinear_function(x: Array, beta0: float, beta1: float) -> Array:
    """
    Nonlinear function used in the paper-like synthetic scenarios.

    f(x; beta0, beta1) = x^(beta0 - 1) exp(-x / beta1)

    The input is clipped to a small positive value because the function is
    defined for positive x.
    """
    x = np.maximum(np.asarray(x, dtype=float), 1e-8)
    return np.power(x, beta0 - 1.0) * np.exp(-x / beta1)


def generate_vs_icnlr_data(
    n_samples: int = 180,
    n_clusters: int = 2,
    n_informative: int = 2,
    n_noise: int = 4,
    relation: RelationType = "intersecting",
    center_model: ModelType = "linear",
    radius_model: ModelType = "linear",
    noise_scale_center: float = 0.05,
    noise_scale_radius: float = 0.02,
    random_state: Optional[int] = None,
) -> SimulationData:
    """
    Generate one synthetic interval-valued dataset.

    Parameters
    ----------
    n_samples:
        Total sample size.
    n_clusters:
        Number of latent clusters.
    n_informative:
        Number of useful explanatory interval variables.  These variables are
        placed first in X.
    n_noise:
        Number of useless explanatory interval variables.  These variables are
        placed after the useful variables and do not enter y.
    relation:
        Cluster geometry for informative variables:
        "disjoint", "intersecting", or "overlapping".
    center_model:
        True function type for response centers.
    radius_model:
        True function type for response radii.
    noise_scale_center, noise_scale_radius:
        Gaussian observational noise added to y center and log y radius.
    random_state:
        Optional random seed.
    """
    if n_samples < n_clusters:
        raise ValueError("n_samples must be at least n_clusters.")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    if n_informative < 1:
        raise ValueError("n_informative must be at least 1.")
    if n_noise < 0:
        raise ValueError("n_noise must be non-negative.")

    rng = np.random.default_rng(random_state)
    n_features = n_informative + n_noise
    labels = _balanced_labels(n_samples, n_clusters)

    X_center = np.empty((n_samples, n_features), dtype=float)
    X_radius = np.empty((n_samples, n_features), dtype=float)

    for k in range(n_clusters):
        row_mask = labels == k
        n_k = int(np.sum(row_mask))
        center_low, center_high = _cluster_range(k, n_clusters, relation, "center", center_model)
        radius_low, radius_high = _cluster_range(k, n_clusters, relation, "radius", radius_model)

        X_center[row_mask, :n_informative] = rng.uniform(
            center_low,
            center_high,
            size=(n_k, n_informative),
        )
        X_radius[row_mask, :n_informative] = rng.uniform(
            radius_low,
            radius_high,
            size=(n_k, n_informative),
        )

    if n_noise:
        X_center[:, n_informative:] = rng.uniform(0.5, 8.0, size=(n_samples, n_noise))
        X_radius[:, n_informative:] = rng.uniform(0.5, 10.0, size=(n_samples, n_noise))

    y_center = np.empty(n_samples, dtype=float)
    y_radius_log = np.empty(n_samples, dtype=float)
    center_params: Dict[int, Dict[str, object]] = {}
    radius_params: Dict[int, Dict[str, object]] = {}

    for k in range(n_clusters):
        row_mask = labels == k
        Xk_center = X_center[row_mask, :n_informative]
        Xk_radius = X_radius[row_mask, :n_informative]

        center_values, center_params[k] = _generate_response_component(
            Xk_center,
            k,
            n_clusters,
            center_model,
            response_kind="center",
            rng=rng,
        )
        radius_log_values, radius_params[k] = _generate_response_component(
            Xk_radius,
            k,
            n_clusters,
            radius_model,
            response_kind="radius",
            rng=rng,
        )

        y_center[row_mask] = center_values
        y_radius_log[row_mask] = radius_log_values

    y_center = y_center + rng.normal(0.0, noise_scale_center, size=n_samples)
    y_radius_log = y_radius_log + rng.normal(0.0, noise_scale_radius, size=n_samples)
    y_radius = np.exp(np.clip(y_radius_log, -745.0, 709.0))
    y_radius = np.maximum(y_radius, 0.05)

    X_lower = X_center - X_radius
    X_upper = X_center + X_radius
    y_lower = y_center - y_radius
    y_upper = y_center + y_radius

    informative_mask = np.zeros(n_features, dtype=bool)
    informative_mask[:n_informative] = True

    metadata: Dict[str, object] = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_informative": n_informative,
        "n_noise": n_noise,
        "relation": relation,
        "center_model": center_model,
        "radius_model": radius_model,
        "center_params": center_params,
        "radius_params": radius_params,
        "noise_scale_center": noise_scale_center,
        "noise_scale_radius": noise_scale_radius,
        "radius_response_scale": "log",
        "random_state": random_state,
    }

    return SimulationData(
        X_lower=X_lower,
        X_upper=X_upper,
        y_lower=y_lower,
        y_upper=y_upper,
        labels=labels,
        informative_mask=informative_mask,
        center_informative_mask=informative_mask.copy(),
        radius_informative_mask=informative_mask.copy(),
        X_center=X_center,
        X_radius=X_radius,
        y_center=y_center,
        y_radius=y_radius,
        metadata=metadata,
    )


def save_simulation_csv(data: SimulationData, path: str | Path) -> None:
    """Save a generated dataset as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_features = data.X_lower.shape
    header = []
    for j in range(n_features):
        header.extend([f"X{j + 1}_lower", f"X{j + 1}_upper"])
    header.extend(["y_lower", "y_upper", "true_group"])

    rows = np.empty((n_samples, 2 * n_features + 3), dtype=float)
    for j in range(n_features):
        rows[:, 2 * j] = data.X_lower[:, j]
        rows[:, 2 * j + 1] = data.X_upper[:, j]
    rows[:, 2 * n_features] = data.y_lower
    rows[:, 2 * n_features + 1] = data.y_upper
    rows[:, 2 * n_features + 2] = data.labels

    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10f",
    )


def scenario_grid() -> list[Dict[str, object]]:
    """
    Return a 24-scenario grid analogous to the paper's simulation structure.

    The grid crosses:
    - K in {2, 3};
    - relation in {disjoint, intersecting, overlapping};
    - center/radius model type in
      {(linear, linear), (linear, nonlinear),
       (nonlinear, linear), (nonlinear, nonlinear)}.
    """
    scenarios = []
    model_pairs = [
        ("linear", "linear"),
        ("linear", "nonlinear"),
        ("nonlinear", "linear"),
        ("nonlinear", "nonlinear"),
    ]
    relations: list[RelationType] = ["disjoint", "intersecting", "overlapping"]
    for center_model, radius_model in model_pairs:
        for n_clusters in (2, 3):
            for relation in relations:
                scenarios.append(
                    {
                        "n_clusters": n_clusters,
                        "relation": relation,
                        "center_model": center_model,
                        "radius_model": radius_model,
                    }
                )
    return scenarios


def _balanced_labels(n_samples: int, n_clusters: int) -> Array:
    labels = np.arange(n_samples) % n_clusters
    return labels.astype(int)


def _cluster_range(
    k: int,
    n_clusters: int,
    relation: RelationType,
    response_kind: Literal["center", "radius"],
    model_type: ModelType,
) -> Tuple[float, float]:
    """
    Paper-inspired fixed covariate ranges.

    The original implementation used one narrow rule for every scenario, e.g.
    disjoint clusters were U(1, 2), U(4, 5), ... .  That made some regression
    clusters hard to identify even when the variable selection was correct.
    These ranges keep the VS extension but use wider, scenario-specific ranges
    closer to the synthetic designs in de Carvalho et al. (2021).
    """
    if n_clusters == 2:
        if relation == "disjoint":
            ranges = [(0.5, 3.0), (4.0, 8.0)]
        elif relation == "intersecting":
            ranges = [(0.5, 3.0), (2.0, 5.0)]
        elif relation == "overlapping":
            ranges = [(0.5, 4.0), (0.5, 4.0)]
        else:
            raise ValueError(f"Unknown relation: {relation}")
    elif n_clusters == 3:
        if relation == "disjoint":
            if response_kind == "radius" and model_type == "nonlinear":
                ranges = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0)]
            else:
                ranges = [(0.5, 2.0), (3.0, 5.0), (6.0, 9.0)]
        elif relation == "intersecting":
            if response_kind == "radius" and model_type == "nonlinear":
                ranges = [(0.5, 4.0), (2.0, 8.0), (5.0, 10.0)]
            else:
                ranges = [(0.5, 4.0), (3.0, 6.0), (5.0, 8.0)]
        elif relation == "overlapping":
            if response_kind == "radius" and model_type == "nonlinear":
                ranges = [(0.5, 10.0), (0.5, 10.0), (0.5, 10.0)]
            else:
                ranges = [(0.5, 4.0), (0.5, 4.0), (0.5, 4.0)]
        else:
            raise ValueError(f"Unknown relation: {relation}")
    else:
        raise ValueError("Only n_clusters in {2, 3} is supported by the scenario grid.")
    return ranges[k]


def _coefficient_vector(base_slope: float, n_features: int) -> Array:
    multipliers = np.linspace(1.0, 0.6, n_features)
    return base_slope * multipliers


def _weight_vector(base_weight: float, n_features: int) -> Array:
    multipliers = np.linspace(1.0, 0.7, n_features)
    return base_weight * multipliers


def _linear_parameters(
    cluster_id: int,
    n_clusters: int,
    response_kind: Literal["center", "radius"],
    n_features: int,
) -> Tuple[float, Array]:
    if response_kind == "center":
        if n_clusters == 2:
            intercepts = [3.0, 0.5]
            slopes = [-1.0, 1.0]
        else:
            intercepts = [4.0, 2.0, 1.0]
            slopes = [1.0, 2.0, 3.0]
    else:
        # Log-radius scale: keep slopes identifiable without creating huge
        # intervals after exponentiation.
        if n_clusters == 2:
            intercepts = [-2.0, -1.8]
            slopes = [0.25, -0.20]
        else:
            intercepts = [-2.0, -1.8, -1.6]
            slopes = [0.22, -0.18, 0.15]
    return intercepts[cluster_id], _coefficient_vector(slopes[cluster_id], n_features)


def _nonlinear_parameters(
    cluster_id: int,
    n_clusters: int,
    response_kind: Literal["center", "radius"],
    n_features: int,
) -> Tuple[float, float, float, Array]:
    if n_clusters == 2:
        beta0_values = [0.5, 1.0]
        beta1_values = [2.0, 3.0]
        if response_kind == "center":
            intercepts = [0.5, 0.75]
            base_weights = [1.5, -1.2]
        else:
            intercepts = [-2.0, -1.8]
            base_weights = [1.2, -1.0]
    else:
        beta0_values = [0.5, 0.75, 0.75]
        beta1_values = [1.0, 4.0, 6.0]
        if response_kind == "center":
            intercepts = [0.5, 1.0, 1.5]
            base_weights = [1.5, -1.2, -1.4]
        else:
            intercepts = [-2.0, -1.8, -1.6]
            base_weights = [1.2, -1.0, -1.2]
    return (
        intercepts[cluster_id],
        beta0_values[cluster_id],
        beta1_values[cluster_id],
        _weight_vector(base_weights[cluster_id], n_features),
    )


def _generate_response_component(
    X: Array,
    cluster_id: int,
    n_clusters: int,
    model_type: ModelType,
    response_kind: Literal["center", "radius"],
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, object]]:
    del rng
    n_features = X.shape[1]
    if model_type == "linear":
        intercept, beta = _linear_parameters(cluster_id, n_clusters, response_kind, n_features)
        values = intercept + X @ beta
        response_scale = "identity" if response_kind == "center" else "log"
        return values, {
            "type": "linear",
            "response_scale": response_scale,
            "intercept": intercept,
            "beta": beta.tolist(),
        }

    if model_type == "nonlinear":
        intercept, beta0, beta1, weights = _nonlinear_parameters(
            cluster_id,
            n_clusters,
            response_kind,
            n_features,
        )
        transformed = paper_nonlinear_function(X, beta0, beta1)
        values = intercept + transformed @ weights
        response_scale = "identity" if response_kind == "center" else "log"
        return values, {
            "type": "nonlinear",
            "form": "additive_paper_f1",
            "response_scale": response_scale,
            "intercept": intercept,
            "beta0": beta0,
            "beta1": beta1,
            "eta1": float(np.log(beta1)),
            "weights": weights.tolist(),
            "uses_variables": list(range(X.shape[1])),
        }

    raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    dataset = generate_vs_icnlr_data(
        n_samples=180,
        n_clusters=2,
        n_informative=2,
        n_noise=4,
        relation="intersecting",
        center_model="linear",
        radius_model="nonlinear",
        random_state=2026,
    )
    output_path = Path("out") / "vs_icnlr_synthetic_example.csv"
    save_simulation_csv(dataset, output_path)
    print(f"Saved {output_path}")
    print("Informative variables:", np.where(dataset.informative_mask)[0].tolist())
    print("Noise variables:", np.where(~dataset.informative_mask)[0].tolist())
