"""
VS-iCNLR: variable-selection interval center and range clusterwise nonlinear
regression.

This module is intentionally independent from icnlr.py.  It implements the
algorithmic extension that adds global or cluster-specific variable selection
to iCNLR, without adding simulation or real-data experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple

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
    """A fitted response model plus the selected feature mask."""

    function: CandidateFunction
    beta: Array
    sse: float
    feature_mask: Array
    response_scale: str = "identity"

    def predict(self, X: Array) -> Array:
        raw = self.raw_predict(X)
        if self.response_scale == "log":
            return np.exp(np.clip(raw, -745.0, 709.0))
        return raw

    def raw_predict(self, X: Array) -> Array:
        return self.function.func(X[:, self.feature_mask], self.beta)

    def target(self, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        if self.response_scale == "log":
            return np.log(_safe_positive(y))
        return y

    def loss(self, X: Array, y: Array) -> Array:
        residual = self.target(y) - self.raw_predict(X)
        return residual * residual

    @property
    def n_selected_features(self) -> int:
        return int(np.sum(self.feature_mask))

    @property
    def n_effective_params(self) -> int:
        return int(self.function.n_params)


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


def make_default_functions(
    n_features: int,
    allowed_names: Optional[Sequence[str]] = None,
) -> List[CandidateFunction]:
    """
    Default candidate functions for a selected feature subset.

    For zero selected features, the only available model is an intercept-only
    linear model.  For any non-empty selected feature set, an additive
    paper-like nonlinear function is available.  For one selected feature, the
    original one-dimensional paper-like nonlinear functions are also included.
    For several selected features, a log-linear model is also available.
    """
    if n_features < 0:
        raise ValueError("n_features must be non-negative.")

    allowed = None if allowed_names is None else set(allowed_names)
    funcs: List[CandidateFunction] = []

    def add_function(function: CandidateFunction) -> None:
        if allowed is None or function.name in allowed:
            funcs.append(function)

    def linear(X: Array, beta: Array) -> Array:
        if X.shape[1] == 0:
            return np.full(X.shape[0], beta[0], dtype=float)
        return beta[0] + X @ beta[1:]

    linear_function = CandidateFunction(
        name="linear",
        n_params=n_features + 1,
        func=linear,
        is_linear=True,
    )
    if n_features == 0:
        funcs.append(linear_function)
        return funcs

    add_function(
        linear_function
    )

    if not funcs and allowed is not None and "linear" in allowed:
        funcs.append(linear_function)

    def additive_f1(X: Array, b: Array) -> Array:
        x = _safe_positive(X)
        beta1 = np.exp(np.clip(float(b[2]), -50.0, 50.0))
        transformed = np.power(x, b[1] - 1.0) * np.exp(-x / beta1)
        return b[0] + transformed @ b[3:]

    add_function(
        CandidateFunction(
            "additive_paper_f1",
            n_features + 3,
            additive_f1,
            bounds=[(-3, 3), (0.2, 2), (-0.7, 2)] + [(-2.5, 2.5)] * n_features,
        )
    )

    paper_function_names = {
        "paper_f1",
        "paper_f2",
        "paper_f3",
        "paper_f5",
        "paper_f7",
        "paper_f9",
        "paper_f11",
        "paper_f13",
    }
    include_paper_functions = allowed is None or bool(allowed & paper_function_names)
    include_log_linear = allowed is None or "log_linear" in allowed

    if n_features == 1 and include_paper_functions:

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

        for function in [
            CandidateFunction("paper_f1", 2, f1, bounds=[(-10, 10), (-10, 10)]),
            CandidateFunction("paper_f2", 3, f2, bounds=[(-100, 100)] * 3),
            CandidateFunction("paper_f3", 2, f3, bounds=[(-100, 100)] * 2),
            CandidateFunction("paper_f5", 3, f5, bounds=[(-100, 100)] * 3),
            CandidateFunction("paper_f7", 3, f7, bounds=[(-100, 100)] * 3),
            CandidateFunction("paper_f9", 2, f9, bounds=[(-100, 100)] * 2),
            CandidateFunction("paper_f11", 3, f11, bounds=[(-100, 100)] * 3),
            CandidateFunction("paper_f13", 3, f13, bounds=[(-100, 100)] * 3),
        ]:
            add_function(function)
    elif n_features > 1 and include_log_linear:

        def log_linear(X: Array, beta: Array) -> Array:
            return beta[0] + np.log(_safe_positive(X)) @ beta[1:]

        add_function(
            CandidateFunction(
                name="log_linear",
                n_params=n_features + 1,
                func=log_linear,
                bounds=[(-100, 100)] * (n_features + 1),
            )
        )

    return funcs


class VSICNLR:
    """
    Variable-selection iCNLR.

    Parameters
    ----------
    n_clusters:
        Fixed number of non-empty clusters K.
    selection_scope:
        "global" means all clusters share one selected feature set for center
        and one selected feature set for radius.  "group" means each cluster
        has its own selected center/radius feature sets.
    selection_criterion:
        "penalty" uses SSE + lambda * number_of_selected_features.
        "bic" uses n log(SSE / n) + d log(n).
    lambda_center, lambda_radius:
        Selection strengths for center and radius responses.
    max_selected_features:
        Optional upper bound on selected feature count when enumerating masks.
    max_feature_subsets:
        Optional cap on the number of masks.  If enumeration is larger, all
        one-feature, empty, and full masks are kept, then the rest are sampled.
    optimizers:
        Optimization methods.  Supported: "L-BFGS-B", "BFGS", "CG", "SANN",
        "RANDOM".
    center_candidate_functions, radius_candidate_functions:
        Optional response-specific candidate function names.  Use this to
        restrict linear scenarios to linear models, or nonlinear synthetic
        scenarios to additive_paper_f1.
    n_init:
        Number of random restarts.
    max_iter:
        Maximum alternating iterations per restart.
    tol:
        Minimum objective improvement required to continue.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        selection_scope: str = "group", #是否为全局的变量选择
        selection_criterion: str = "penalty", #用BIC准则选择变量还是 sse+lambda*变量个数
        lambda_center: float = 0.0,
        lambda_radius: float = 0.0,
        max_selected_features: Optional[int] = None,
        max_feature_subsets: Optional[int] = None,
        optimizers: Sequence[str] = ("BFGS", "CG", "SANN"),
        center_candidate_functions: Optional[Sequence[str]] = None,
        radius_candidate_functions: Optional[Sequence[str]] = None,
        n_init: int = 20,
        max_iter: int = 100,
        tol: float = 1e-7,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if selection_scope not in {"global", "group"}:
            raise ValueError('selection_scope must be "global" or "group".')
        if selection_criterion not in {"penalty", "bic"}:
            raise ValueError('selection_criterion must be "penalty" or "bic".')
        if lambda_center < 0 or lambda_radius < 0:
            raise ValueError("lambda_center and lambda_radius must be non-negative.")

        self.n_clusters = n_clusters
        self.selection_scope = selection_scope
        self.selection_criterion = selection_criterion
        self.lambda_center = float(lambda_center)
        self.lambda_radius = float(lambda_radius)
        self.max_selected_features = max_selected_features
        self.max_feature_subsets = max_feature_subsets
        self.optimizers = tuple(optimizers)
        self.center_candidate_functions = (
            None if center_candidate_functions is None else tuple(center_candidate_functions)
        )
        self.radius_candidate_functions = (
            None if radius_candidate_functions is None else tuple(radius_candidate_functions)
        )
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.labels_: Optional[Array] = None
        self.models_: Optional[List[ClusterModel]] = None
        self.objective_: Optional[float] = None
        self.sse_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.X_center_: Optional[Array] = None
        self.X_radius_: Optional[Array] = None
        self.selected_center_features_: Optional[Array] = None
        self.selected_radius_features_: Optional[Array] = None

    def fit(self, X_lower: Array, X_upper: Array, y_lower: Array, y_upper: Array) -> "VSICNLR":
        Xc, Xr = interval_to_center_radius(X_lower, X_upper)
        yc, yr = interval_to_center_radius(y_lower, y_upper)
        yc = yc.reshape(-1)
        yr = yr.reshape(-1)

        if Xc.ndim != 2:
            raise ValueError("X_lower and X_upper must be 2D arrays with shape (n, p).")
        if len(yc) != Xc.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")
        if self.n_clusters > Xc.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of observations.")

        rng = np.random.default_rng(self.random_state)
        feature_masks = self._generate_feature_masks(Xc.shape[1], rng) #区别于icnlr的部分

        best_objective = np.inf
        best_sse = np.inf
        best_labels = None
        best_models = None
        best_iter = 0

        for _ in range(self.n_init):
            labels = self._init_labels(Xc.shape[0], rng)
            models = self._init_models(Xc, Xr, yc, yr, labels, feature_masks, rng)
            prev_objective = np.inf

            for iteration in range(1, self.max_iter + 1):
                models = self._fit_all_clusters(Xc, Xr, yc, yr, labels, models, feature_masks, rng)
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
                best_sse = self._total_sse(Xc, Xr, yc, yr, labels, models)
                best_labels = labels.copy()
                best_models = models
                best_iter = iteration

        self.labels_ = best_labels
        self.models_ = best_models
        self.objective_ = float(best_objective)
        self.sse_ = float(best_sse)
        self.n_iter_ = best_iter
        self.X_center_ = Xc
        self.X_radius_ = Xr
        self._store_selected_features()
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
            yr_hat[mask] = model.radius.predict(Xr[mask])

        return yc_hat - yr_hat, yc_hat + yr_hat

    def _init_labels(self, n_samples: int, rng: np.random.Generator) -> Array:
        labels = np.arange(n_samples) % self.n_clusters
        rng.shuffle(labels)
        return labels

    def _init_models(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        feature_masks: Sequence[Array],
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        previous = []
        empty_mask = np.zeros(Xc.shape[1], dtype=bool)
        for k in range(self.n_clusters):
            row_mask = labels == k
            center = self._fit_one_response_with_mask(
                Xc[row_mask],
                yc[row_mask],
                empty_mask,
                None,
                rng,
                response_scale="identity",
                candidate_function_names=self.center_candidate_functions,
            )
            radius = self._fit_one_response_with_mask(
                Xr[row_mask],
                yr[row_mask],
                empty_mask,
                None,
                rng,
                response_scale="log",
                candidate_function_names=self.radius_candidate_functions,
            )
            previous.append(ClusterModel(center=center, radius=radius))
        return self._fit_all_clusters(Xc, Xr, yc, yr, labels, previous, feature_masks, rng)

    def _fit_all_clusters(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        previous: List[ClusterModel],
        feature_masks: Sequence[Array],
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        if self.selection_scope == "global":
            return self._fit_all_clusters_global_selection(
                Xc, Xr, yc, yr, labels, previous, feature_masks, rng
            )
        return self._fit_all_clusters_group_selection(
            Xc, Xr, yc, yr, labels, previous, feature_masks, rng
        )

    def _fit_all_clusters_group_selection(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        previous: List[ClusterModel],
        feature_masks: Sequence[Array],
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        models = []
        for k in range(self.n_clusters):
            row_mask = labels == k
            center = self._fit_one_response_with_selection(
                Xc[row_mask],
                yc[row_mask],
                previous[k].center,
                feature_masks,
                self.lambda_center,
                rng,
                response_scale="identity",
                candidate_function_names=self.center_candidate_functions,
            )
            radius = self._fit_one_response_with_selection(
                Xr[row_mask],
                yr[row_mask],
                previous[k].radius,
                feature_masks,
                self.lambda_radius,
                rng,
                response_scale="log",
                candidate_function_names=self.radius_candidate_functions,
            )
            models.append(ClusterModel(center=center, radius=radius))
        return models

    def _fit_all_clusters_global_selection(
        self,
        Xc: Array,
        Xr: Array,
        yc: Array,
        yr: Array,
        labels: Array,
        previous: List[ClusterModel],
        feature_masks: Sequence[Array],
        rng: np.random.Generator,
    ) -> List[ClusterModel]:
        centers = self._fit_global_response_selection(
            Xc,
            yc,
            labels,
            [model.center for model in previous],
            feature_masks,
            self.lambda_center,
            rng,
            response_scale="identity",
            candidate_function_names=self.center_candidate_functions,
        )
        radii = self._fit_global_response_selection(
            Xr,
            yr,
            labels,
            [model.radius for model in previous],
            feature_masks,
            self.lambda_radius,
            rng,
            response_scale="log",
            candidate_function_names=self.radius_candidate_functions,
        )
        return [ClusterModel(center=centers[k], radius=radii[k]) for k in range(self.n_clusters)]

    def _fit_global_response_selection(
        self,
        X: Array,
        y: Array,
        labels: Array,
        previous: Sequence[FittedModel],
        feature_masks: Sequence[Array],
        lambda_: float,
        rng: np.random.Generator,
        response_scale: str,
        candidate_function_names: Optional[Sequence[str]],
    ) -> List[FittedModel]:
        best_models: Optional[List[FittedModel]] = None
        best_score = np.inf

        for feature_mask in feature_masks:
            models = []
            total_sse = 0.0
            total_params = 0

            for k in range(self.n_clusters):
                row_mask = labels == k
                fitted = self._fit_one_response_with_mask(
                    X[row_mask],
                    y[row_mask],
                    feature_mask,
                    previous[k],
                    rng,
                    response_scale=response_scale,
                    candidate_function_names=candidate_function_names,
                )
                models.append(fitted)
                total_sse += fitted.sse
                total_params += fitted.n_effective_params

            score = self._selection_score(
                total_sse,
                n_samples=len(y),
                n_selected=int(np.sum(feature_mask)),
                n_params=total_params,
                lambda_=lambda_,
            )
            if score < best_score:
                best_score = score
                best_models = models

        if best_models is None:
            raise RuntimeError("Global variable selection failed.")
        return best_models

    def _fit_one_response_with_selection(
        self,
        X: Array,
        y: Array,
        previous: FittedModel,
        feature_masks: Sequence[Array],
        lambda_: float,
        rng: np.random.Generator,
        response_scale: str,
        candidate_function_names: Optional[Sequence[str]],
    ) -> FittedModel:
        best: Optional[FittedModel] = None
        best_score = np.inf

        for feature_mask in feature_masks:
            fitted = self._fit_one_response_with_mask(
                X,
                y,
                feature_mask,
                previous,
                rng,
                response_scale=response_scale,
                candidate_function_names=candidate_function_names,
            )
            score = self._selection_score(
                fitted.sse,
                n_samples=len(y),
                n_selected=fitted.n_selected_features,
                n_params=fitted.n_effective_params,
                lambda_=lambda_,
            )
            if score < best_score:
                best = fitted
                best_score = score

        if best is None:
            return previous
        return best

    def _fit_one_response_with_mask(
        self,
        X: Array,
        y: Array,
        feature_mask: Array,
        previous: Optional[FittedModel],
        rng: np.random.Generator,
        response_scale: str = "identity",
        candidate_function_names: Optional[Sequence[str]] = None,
    ) -> FittedModel:
        X_selected = X[:, feature_mask]
        funcs = make_default_functions(X_selected.shape[1], candidate_function_names)
        previous_for_mask = (
            previous
            if previous is not None
            and np.array_equal(previous.feature_mask, feature_mask)
            and previous.response_scale == response_scale
            else None
        )
        best: Optional[FittedModel] = None

        for function in funcs:
            fitted = self._fit_candidate(
                function, X_selected, y, previous_for_mask, feature_mask, rng, response_scale
            )
            if fitted is not None and (best is None or fitted.sse < best.sse):
                best = fitted

        if best is not None:
            return best
        if previous is not None:
            return previous

        base_func = make_default_functions(0)[0]
        empty_mask = np.zeros(X.shape[1], dtype=bool)
        target_y = self._transform_response(y, response_scale)
        beta = np.array([float(np.mean(target_y))]) if len(target_y) else np.array([0.0])
        return FittedModel(
            base_func,
            beta,
            self._sse(base_func, beta, X[:, empty_mask], target_y),
            empty_mask,
            response_scale,
        )

    def _fit_candidate(
        self,
        function: CandidateFunction,
        X_selected: Array,
        y: Array,
        previous: Optional[FittedModel],
        feature_mask: Array,
        rng: np.random.Generator,
        response_scale: str,
    ) -> Optional[FittedModel]:
        if len(y) == 0:
            return None
        target_y = self._transform_response(y, response_scale)

        if function.is_linear and function.n_params == X_selected.shape[1] + 1:
            design = np.column_stack([np.ones(X_selected.shape[0]), X_selected])
            beta, *_ = np.linalg.lstsq(design, target_y, rcond=None)
            return FittedModel(
                function,
                beta,
                self._sse(function, beta, X_selected, target_y),
                feature_mask.copy(),
                response_scale,
            )

        starts = [function.initial_params(rng) for _ in range(3)]
        if previous is not None and previous.function.name == function.name:
            starts.insert(0, previous.beta)

        for optimizer in self.optimizers:
            for start in starts:
                fitted = self._optimize(
                    function, X_selected, target_y, start, optimizer, feature_mask, rng, response_scale
                )
                if fitted is not None:
                    return fitted

        return None

    def _optimize(
        self,
        function: CandidateFunction,
        X_selected: Array,
        y: Array,
        start: Array,
        optimizer: str,
        feature_mask: Array,
        rng: np.random.Generator,
        response_scale: str,
    ) -> Optional[FittedModel]:
        objective = lambda beta: self._sse(function, beta, X_selected, y)
        optimizer = optimizer.upper()

        if optimizer == "LBFGSB":
            optimizer = "L-BFGS-B"

        if minimize is not None and optimizer in {"L-BFGS-B", "BFGS", "CG"}:
            bounds = function.bounds if optimizer == "L-BFGS-B" else None
            start = self._clip_to_bounds(start, bounds)
            result = minimize(objective, start, method=optimizer, bounds=bounds, options={"maxiter": 500})
            if result.success and np.isfinite(result.fun):
                return FittedModel(
                    function,
                    np.asarray(result.x),
                    float(result.fun),
                    feature_mask.copy(),
                    response_scale,
                )
            return None

        if dual_annealing is not None and optimizer in {"SANN", "ANNEALING"}:
            bounds = function.bounds or [(-100.0, 100.0)] * function.n_params
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            result = dual_annealing(objective, bounds=bounds, maxiter=300, seed=seed)
            if result.success and np.isfinite(result.fun):
                return FittedModel(
                    function,
                    np.asarray(result.x),
                    float(result.fun),
                    feature_mask.copy(),
                    response_scale,
                )
            return None

        if optimizer in {"L-BFGS-B", "BFGS", "CG", "SANN", "ANNEALING", "RANDOM"}:
            beta, value = self._fallback_random_search(objective, start, rng, function.bounds)
            if np.isfinite(value):
                return FittedModel(
                    function,
                    beta,
                    float(value),
                    feature_mask.copy(),
                    response_scale,
                )

        return None

    @staticmethod
    def _clip_to_bounds(start: Array, bounds: Optional[Sequence[Tuple[float, float]]]) -> Array:
        beta = np.asarray(start, dtype=float).copy()
        if bounds is None:
            return beta
        lower = np.array([bound[0] for bound in bounds], dtype=float)
        upper = np.array([bound[1] for bound in bounds], dtype=float)
        return np.clip(beta, lower, upper)

    @staticmethod
    def _fallback_random_search(
        objective: Callable[[Array], float],
        start: Array,
        rng: np.random.Generator,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Tuple[Array, float]:
        best_beta = VSICNLR._clip_to_bounds(start, bounds)
        best_value = objective(best_beta)
        step = 1.0
        for _ in range(250):
            candidate = best_beta + rng.normal(0.0, step, size=best_beta.shape)
            candidate = VSICNLR._clip_to_bounds(candidate, bounds)
            value = objective(candidate)
            if value < best_value:
                best_beta = candidate
                best_value = value
            step *= 0.98
        return best_beta, float(best_value)

    @staticmethod
    def _transform_response(y: Array, response_scale: str) -> Array:
        y = np.asarray(y, dtype=float)
        if response_scale == "identity":
            return y
        if response_scale == "log":
            return np.log(_safe_positive(y))
        raise ValueError('response_scale must be "identity" or "log".')

    @staticmethod
    def _sse(function: CandidateFunction, beta: Array, X_selected: Array, y: Array) -> float:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = function.func(X_selected, np.asarray(beta, dtype=float))
                value = np.sum((y - pred) ** 2)
        except Exception:
            return np.inf
        if not np.isfinite(value):
            return np.inf
        return float(value)
#样本分配的逻辑没有太大变化
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
            losses[:, k] = model.center.loss(Xc, yc) + model.radius.loss(Xr, yr)
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
                model.center.loss(Xc[i : i + 1], yc[i : i + 1])[0]
                + model.radius.loss(Xr[i : i + 1], yr[i : i + 1])[0]
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
        sse = self._total_sse(Xc, Xr, yc, yr, labels, models)
        if self.selection_criterion == "bic":
            n_params = sum(model.center.n_effective_params + model.radius.n_effective_params for model in models)
            return self._bic_score(sse, len(yc), n_params)
        return sse + self._selection_penalty(models)

    def _total_sse(
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
            row_mask = labels == k
            total += self._model_sse(model.center, Xc[row_mask], yc[row_mask])
            total += self._model_sse(model.radius, Xr[row_mask], yr[row_mask])
        return float(total)

    def _model_sse(self, model: FittedModel, X: Array, y: Array) -> float:
        value = np.sum(model.loss(X, y))
        if not np.isfinite(value):
            return np.inf
        return float(value)

    def _selection_penalty(self, models: List[ClusterModel]) -> float:
        if self.selection_scope == "global":
            center_features = models[0].center.n_selected_features
            radius_features = models[0].radius.n_selected_features
            return self.lambda_center * center_features + self.lambda_radius * radius_features

        penalty = 0.0
        for model in models:
            penalty += self.lambda_center * model.center.n_selected_features
            penalty += self.lambda_radius * model.radius.n_selected_features
        return float(penalty)

    def _selection_score(
        self,
        sse: float,
        n_samples: int,
        n_selected: int,
        n_params: int,
        lambda_: float,
    ) -> float:
        if self.selection_criterion == "bic":
            return self._bic_score(sse, n_samples, n_params)
        return float(sse + lambda_ * n_selected)

    @staticmethod
    def _bic_score(sse: float, n_samples: int, n_params: int) -> float:
        n = max(int(n_samples), 1)
        safe_sse = max(float(sse), np.finfo(float).tiny)
        return float(n * np.log(safe_sse / n) + n_params * np.log(n))

    def _generate_feature_masks(self, n_features: int, rng: np.random.Generator) -> List[Array]:
        max_size = self.max_selected_features
        if max_size is None:
            max_size = n_features
        max_size = max(0, min(max_size, n_features))

        masks: List[Array] = []
        for size in range(0, max_size + 1):
            for cols in combinations(range(n_features), size):
                mask = np.zeros(n_features, dtype=bool)
                mask[list(cols)] = True
                masks.append(mask)

        if self.max_selected_features is None or self.max_selected_features >= n_features:
            full_mask = np.ones(n_features, dtype=bool)
            if not any(np.array_equal(mask, full_mask) for mask in masks):
                masks.append(full_mask)

        if self.max_feature_subsets is not None and len(masks) > self.max_feature_subsets:
            masks = self._sample_feature_masks(masks, self.max_feature_subsets, rng)

        return masks

    @staticmethod
    def _sample_feature_masks(
        masks: Sequence[Array],
        max_feature_subsets: int,
        rng: np.random.Generator,
    ) -> List[Array]:
        if max_feature_subsets < 2:
            raise ValueError("max_feature_subsets must be at least 2.")

        mandatory: List[Array] = []
        for mask in masks:
            if mask.sum() in {0, 1, mask.size}:
                mandatory.append(mask)

        unique = []
        seen = set()
        for mask in mandatory:
            key = tuple(mask.tolist())
            if key not in seen:
                unique.append(mask)
                seen.add(key)

        if len(unique) >= max_feature_subsets:
            return unique[:max_feature_subsets]

        remaining = [mask for mask in masks if tuple(mask.tolist()) not in seen]
        n_extra = max_feature_subsets - len(unique)
        indices = rng.choice(len(remaining), size=n_extra, replace=False)
        unique.extend(remaining[int(i)] for i in indices)
        return unique

    def _store_selected_features(self) -> None:
        if self.models_ is None:
            return
        center_masks = np.vstack([model.center.feature_mask for model in self.models_])
        radius_masks = np.vstack([model.radius.feature_mask for model in self.models_])
        if self.selection_scope == "global":
            self.selected_center_features_ = center_masks[0].copy()
            self.selected_radius_features_ = radius_masks[0].copy()
        else:
            self.selected_center_features_ = center_masks
            self.selected_radius_features_ = radius_masks

    def _check_fitted(self) -> None:
        if self.models_ is None:
            raise RuntimeError("VSICNLR has not been fitted yet.")


VS_iCNLR = VSICNLR
