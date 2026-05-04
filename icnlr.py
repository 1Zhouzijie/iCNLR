"""
iCNLR: interval center and range clusterwise nonlinear regression.

This file implements Algorithm 1 in de Carvalho et al. (2021), pp. 364-365.
It focuses on the algorithm itself, not on the simulation or real-data
experiments from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import dual_annealing, minimize
except Exception:  # pragma: no cover - scipy is optional here
    dual_annealing = None
    minimize = None


Array = np.ndarray


@dataclass(frozen=True)
class CandidateFunction:
    """A candidate regression function y = f(X, beta)."""

    name: str
    n_params: int
    func: Callable[[Array, Array], Array]
    is_linear: bool = False
    bounds: Optional[Sequence[Tuple[float, float]]] = None

    def initial_params(self, rng: np.random.Generator) -> Array:
        return rng.normal(loc=0.0, scale=1.0, size=self.n_params)


@dataclass
class FittedModel:
    function: CandidateFunction
    beta: Array
    sse: float

    def predict(self, X: Array) -> Array:
        return self.function.func(X, self.beta)


@dataclass
class ClusterModel:
    center: FittedModel
    radius: FittedModel


def interval_to_center_radius(lower: Array, upper: Array) -> Tuple[Array, Array]:
    """Convert interval lower/upper bounds to center/half-range."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")
    if np.any(upper < lower):
        raise ValueError("All interval upper bounds must be >= lower bounds.")
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    return center, radius


def _safe_positive(x: Array) -> Array:
    return np.maximum(np.asarray(x, dtype=float), 1e-8)


def make_default_functions(n_features: int) -> List[CandidateFunction]:
    """
    Candidate functions used by iCNLR.

    For one explanatory variable, this includes the linear model and nonlinear
    shapes close to the list used in the paper. For several variables, the
    default is a linear model plus a log-linear nonlinear form. You can pass
    your own CandidateFunction list to ICNLR if a paper-specific function set
    is needed.
    """
    funcs: List[CandidateFunction] = []

    def linear(X: Array, beta: Array) -> Array:
        return beta[0] + X @ beta[1:]

    funcs.append(
        CandidateFunction(
            name="linear",
            n_params=n_features + 1,
            func=linear,
            is_linear=True,
        )
    )

    if n_features == 1:

        def f1(X: Array, b: Array) -> Array:
            x = _safe_positive(X[:, 0])
            return np.power(x, b[0] - 1.0) * np.exp(-x / (b[1] + 1e-8))

        def f2(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return b[0] - b[1] / (b[2] + x + 1e-8)

        def f3(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return b[1] * x / (b[0] + x + 1e-8)

        def f5(X: Array, b: Array) -> Array:
            x = _safe_positive(X[:, 0])
            return b[2] + (1.0 - b[2]) / (1.0 + np.exp(-b[0] - b[1] * np.log(x)))

        def f7(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return b[2] + (1.0 - b[2]) / (1.0 + np.exp(-b[0] - b[1] * x))

        def f9(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return b[0] + (1.0 - b[0]) * (1.0 - np.exp(-b[1] * x))

        def f11(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return b[0] + (1.0 - b[0]) * (1.0 - np.exp(-b[1] * x - b[2] * x * x))

        def f13(X: Array, b: Array) -> Array:
            x = X[:, 0]
            return 1.0 - np.exp(-b[0] * x - b[1] * x * x - b[2] * x * x * x)

        funcs.extend(
            [
                CandidateFunction("paper_f1", 2, f1, bounds=[(-10, 10), (-10, 10)]),
                CandidateFunction("paper_f2", 3, f2, bounds=[(-100, 100)] * 3),
                CandidateFunction("paper_f3", 2, f3, bounds=[(-100, 100)] * 2),
                CandidateFunction("paper_f5", 3, f5, bounds=[(-100, 100)] * 3),
                CandidateFunction("paper_f7", 3, f7, bounds=[(-100, 100)] * 3),
                CandidateFunction("paper_f9", 2, f9, bounds=[(-100, 100)] * 2),
                CandidateFunction("paper_f11", 3, f11, bounds=[(-100, 100)] * 3),
                CandidateFunction("paper_f13", 3, f13, bounds=[(-100, 100)] * 3),
            ]
        )
    else:

        def log_linear(X: Array, beta: Array) -> Array:
            return beta[0] + np.log(_safe_positive(X)) @ beta[1:]

        funcs.append(
            CandidateFunction(
                name="log_linear",
                n_params=n_features + 1,
                func=log_linear,
                bounds=[(-100, 100)] * (n_features + 1),
            )
        )

    return funcs


class ICNLR:
    """
    Interval center and range clusterwise nonlinear regression.

    Parameters
    ----------
    n_clusters:
        Fixed number of non-empty clusters K.
    functions:
        Candidate function set H. If None, make_default_functions is used.
    optimizers:
        Optimization methods O. Supported: "BFGS", "CG", "SANN".
    n_init:
        Number of random restarts. The solution with the smallest objective is kept.
    max_iter:
        Maximum alternating optimization iterations per restart.
    tol:
        Minimum objective improvement required to continue.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        functions: Optional[Sequence[CandidateFunction]] = None,
        optimizers: Sequence[str] = ("BFGS", "CG", "SANN"),
        n_init: int = 20,
        max_iter: int = 100,
        tol: float = 1e-7,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        self.n_clusters = n_clusters
        self.functions = list(functions) if functions is not None else None
        self.optimizers = tuple(optimizers)
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.labels_: Optional[Array] = None #保存样本是属于那个cluster
        self.models_: Optional[List[ClusterModel]] = None #保存每个cluster的模型
        self.objective_: Optional[float] = None #保存总误差
        self.n_iter_: Optional[int] = None #保存迭代的轮数
        self.X_center_: Optional[Array] = None
        self.X_radius_: Optional[Array] = None

    def fit(self, X_lower: Array, X_upper: Array, y_lower: Array, y_upper: Array) -> "ICNLR":
        Xc, Xr = interval_to_center_radius(X_lower, X_upper)
        yc, yr = interval_to_center_radius(y_lower, y_upper)
        yc = yc.reshape(-1) #把yc,yr变成一维数组：[[1],[2],[3]] -> [1,2,3]
        yr = yr.reshape(-1)
        if Xc.ndim != 2:
            raise ValueError("X_lower and X_upper must be 2D arrays with shape (n, p).")
        if len(yc) != Xc.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if self.n_clusters > Xc.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of observations.")

        funcs = self.functions or make_default_functions(Xc.shape[1])
        rng = np.random.default_rng(self.random_state)

        best_objective = np.inf
        best_labels = None
        best_models = None
        best_iter = 0

        for _ in range(self.n_init):
            labels = self._init_labels(Xc.shape[0], rng)
            models = self._init_models(funcs, Xc, Xr, yc, yr, labels, rng)
            prev_objective = np.inf

            for iteration in range(1, self.max_iter + 1):
                models = self._fit_all_clusters(funcs, Xc, Xr, yc, yr, labels, models, rng)
                new_labels = self._assign_clusters(Xc, Xr, yc, yr, models)
                new_labels = self._repair_empty_clusters(new_labels, Xc, Xr, yc, yr, models)
                objective = self._objective(Xc, Xr, yc, yr, new_labels, models)

                if np.array_equal(new_labels, labels) or abs(prev_objective - objective) <= self.tol:
                    labels = new_labels
                    break

                labels = new_labels
                prev_objective = objective

            objective = self._objective(Xc, Xr, yc, yr, labels, models)
            if objective < best_objective:
                best_objective = objective
                best_labels = labels.copy()
                best_models = models
                best_iter = iteration

        self.labels_ = best_labels
        self.models_ = best_models
        self.objective_ = float(best_objective)
        self.n_iter_ = best_iter
        self.X_center_ = Xc
        self.X_radius_ = Xr
        return self

    def predict_with_labels(
        self,
        X_lower: Array,
        X_upper: Array,
        labels: Array,
    ) -> Tuple[Array, Array]:
        """Predict interval lower/upper bounds when cluster labels are known."""
        self._check_fitted()
        Xc, Xr = interval_to_center_radius(X_lower, X_upper)
        labels = np.asarray(labels, dtype=int)
        yc_hat = np.empty(Xc.shape[0], dtype=float)
        yr_hat = np.empty(Xc.shape[0], dtype=float)

        for k in range(self.n_clusters):
            mask = labels == k
            if not np.any(mask):
                continue
            model = self.models_[k]
            yc_hat[mask] = model.center.predict(Xc[mask])
            yr_hat[mask] = np.maximum(model.radius.predict(Xr[mask]), 0.0)

        return yc_hat - yr_hat, yc_hat + yr_hat

    def _init_labels(self, n_samples: int, rng: np.random.Generator) -> Array:
        labels = np.arange(n_samples) % self.n_clusters
        rng.shuffle(labels)
        return labels

    def _init_models(
        self,
        funcs: Sequence[CandidateFunction],
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        models = []
        base_func = funcs[0]
        for k in range(self.n_clusters):
            mask = labels == k
            center = self._fit_one_response(funcs, Xc[mask], yc[mask], None, rng)
            radius = self._fit_one_response(funcs, Xr[mask], yr[mask], None, rng)
            if center is None:
                beta = base_func.initial_params(rng)
                center = FittedModel(base_func, beta, self._sse(base_func, beta, Xc[mask], yc[mask]))
            if radius is None:
                beta = base_func.initial_params(rng)
                radius = FittedModel(base_func, beta, self._sse(base_func, beta, Xr[mask], yr[mask]))
            models.append(ClusterModel(center=center, radius=radius))
        return models

    def _fit_all_clusters(
        self,
        funcs: Sequence[CandidateFunction],
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        previous: List[ClusterModel],
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        models = []
        for k in range(self.n_clusters):
            mask = labels == k
            prev_center = previous[k].center
            prev_radius = previous[k].radius
            center = self._fit_one_response(funcs, Xc[mask], yc[mask], prev_center, rng)
            radius = self._fit_one_response(funcs, Xr[mask], yr[mask], prev_radius, rng)
            models.append(
                ClusterModel(
                    center=center if center is not None else prev_center,
                    radius=radius if radius is not None else prev_radius,
                )
            )
        return models

    def _fit_one_response(
        self,
        funcs: Sequence[CandidateFunction],
        X: Array,
        y: Array,
        previous: Optional[FittedModel],
        rng: np.random.Generator,
    ) -> Optional[FittedModel]:
        best: Optional[FittedModel] = None

        for function in funcs:
            fitted = self._fit_candidate(function, X, y, previous, rng)
            if fitted is not None and (best is None or fitted.sse < best.sse):
                best = fitted

        if best is None and previous is not None:
            return previous
        return best

    def _fit_candidate(
        self,
        function: CandidateFunction,
        X: Array,
        y: Array,
        previous: Optional[FittedModel],
        rng: np.random.Generator,
    ) -> Optional[FittedModel]:
        if len(y) == 0:
            return None

        if function.is_linear and function.n_params == X.shape[1] + 1:
            design = np.column_stack([np.ones(X.shape[0]), X])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            return FittedModel(function, beta, self._sse(function, beta, X, y))

        starts = [function.initial_params(rng) for _ in range(3)]
        if previous is not None and previous.function.name == function.name:
            starts.insert(0, previous.beta)

        for optimizer in self.optimizers:
            for start in starts:
                fitted = self._optimize(function, X, y, start, optimizer, rng)
                if fitted is not None:
                    return fitted

        return None

    def _optimize(
        self,
        function: CandidateFunction,
        X: Array,
        y: Array,
        start: Array,
        optimizer: str,
        rng: np.random.Generator,
    ) -> Optional[FittedModel]:
        objective = lambda beta: self._sse(function, beta, X, y)
        optimizer = optimizer.upper()

        if minimize is not None and optimizer in {"BFGS", "CG"}:
            result = minimize(objective, start, method=optimizer, options={"maxiter": 500})
            if result.success and np.isfinite(result.fun):
                return FittedModel(function, np.asarray(result.x), float(result.fun))
            return None

        if dual_annealing is not None and optimizer in {"SANN", "ANNEALING"}:
            bounds = function.bounds or [(-100.0, 100.0)] * function.n_params
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            result = dual_annealing(objective, bounds=bounds, maxiter=300, seed=seed)
            if result.success and np.isfinite(result.fun):
                return FittedModel(function, np.asarray(result.x), float(result.fun))
            return None

        if optimizer in {"BFGS", "CG", "SANN", "ANNEALING", "RANDOM"}:
            beta, value = self._fallback_random_search(objective, start, rng)
            if np.isfinite(value):
                return FittedModel(function, beta, float(value))

        return None

    @staticmethod
    def _fallback_random_search(
        objective: Callable[[Array], float],
        start: Array,
        rng: np.random.Generator,
    ) -> Tuple[Array, float]:
        best_beta = np.asarray(start, dtype=float).copy()
        best_value = objective(best_beta)
        step = 1.0
        for _ in range(250):
            candidate = best_beta + rng.normal(0.0, step, size=best_beta.shape)
            value = objective(candidate)
            if value < best_value:
                best_beta = candidate
                best_value = value
            step *= 0.98
        return best_beta, float(best_value)

    @staticmethod
    def _sse(function: CandidateFunction, beta: Array, X: Array, y: Array) -> float:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = function.func(X, np.asarray(beta, dtype=float))
                value = np.sum((y - pred) ** 2)
        except Exception:
            return np.inf
        if not np.isfinite(value):
            return np.inf
        return float(value)

    def _assign_clusters(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        models: List[ClusterModel],
    ) -> Array:
        losses = np.empty((Xc.shape[0], self.n_clusters), dtype=float)
        for k, model in enumerate(models):
            center_res = yc - model.center.predict(Xc)
            radius_res = yr - model.radius.predict(Xr)
            losses[:, k] = center_res * center_res + radius_res * radius_res
        return np.argmin(losses, axis=1)

    def _repair_empty_clusters(
        self,
        labels: Array,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        models: List[ClusterModel],
    ) -> Array:
        labels = labels.copy()
        counts = np.bincount(labels, minlength=self.n_clusters)
        empty = np.where(counts == 0)[0]
        if len(empty) == 0:
            return labels

        current_loss = np.zeros(Xc.shape[0], dtype=float)
        for i, k in enumerate(labels):
            model = models[k]
            current_loss[i] = (
                (yc[i] - model.center.predict(Xc[i : i + 1])[0]) ** 2
                + (yr[i] - model.radius.predict(Xr[i : i + 1])[0]) ** 2
            )

        for k_empty in empty:
            donor_clusters = np.where(counts > 1)[0]
            if len(donor_clusters) == 0:
                break
            donor_mask = np.isin(labels, donor_clusters)
            donor_indices = np.where(donor_mask)[0]
            move_index = donor_indices[np.argmax(current_loss[donor_indices])]
            counts[labels[move_index]] -= 1
            labels[move_index] = k_empty
            counts[k_empty] += 1

        return labels

    def _objective(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        models: List[ClusterModel],
    ) -> float:
        total = 0.0
        for k, model in enumerate(models):
            mask = labels == k
            total += self._sse(model.center.function, model.center.beta, Xc[mask], yc[mask])
            total += self._sse(model.radius.function, model.radius.beta, Xr[mask], yr[mask])
        return float(total)

    def _check_fitted(self) -> None:
        if self.models_ is None:
            raise RuntimeError("ICNLR has not been fitted yet.")
