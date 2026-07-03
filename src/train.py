"""
model_training.py

OOP structure for benchmarking multiple regression models on the
Belgian real estate dataset (data/Dataframe_Clean_encoded.csv).

  BaseModel        -> Shared logic across every model: data loading,
                       train/test split, imputing + scaling (via a
                       ColumnTransformer fit only on the training set to
                       avoid leakage), generic scoring (R2, adjusted R2,
                       RMSE, MAE, cross-validation), a reusable generic
                       + two-pass GridSearchCV routine, and a generic
                       RandomizedSearchCV routine.

  RidgeModel        -> Ridge-specific logic: builds the estimator and
                        exposes a plain single-alpha train().

  DecisionTreeModel  -> Decision Tree-specific logic: single max_depth
                        hyperparameter, tuned via the inherited
                        two_pass_tune().

  RandomForestModel  -> Random Forest-specific logic. Unlike Ridge/Tree,
                        RF has SEVERAL interacting hyperparameters
                        (n_estimators, max_depth, max_features,
                        min_samples_split, min_samples_leaf) so a
                        single-parameter two-pass sweep is not
                        appropriate. See the class docstring below for
                        the tuning strategy used instead.

Usage:
    ridge = RidgeModel("data/Dataframe_Clean_encoded.csv")
    ridge.load_data().split_data()
    ridge.train(alpha=1.0)
    ridge.evaluate()
    ridge.two_pass_tune(param_name="alpha", coarse_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000])

    rf = RandomForestModel("data/Dataframe_Clean_encoded.csv")
    rf.load_data().split_data()
    rf.find_n_estimators_elbow()
    rf.random_search_tune(n_iter=40)
    rf.evaluate()
"""

import contextlib
import time
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost is an optional third-party dependency (not part of sklearn) ->
# imported lazily/guarded so the rest of this file still works (Ridge,
# DecisionTree, RandomForest, SVM) even if it isn't installed. Install
# with: pip install xgboost --break-system-packages
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ====================================================================
#  BASE CLASS - shared across all models
# ====================================================================
class BaseModel:
    """
    Handles everything that is NOT specific to a given algorithm:
      - loading the CSV already preprocessed/encoded by data_preprocessing.py
      - train/test split
      - building the ColumnTransformer (impute + scale)
      - generic scoring / reporting
      - generic + two-pass GridSearchCV hyperparameter tuning
      - generic RandomizedSearchCV hyperparameter tuning

    Subclasses (RidgeModel, LassoModel, RandomForestModel, ...) only need
    to implement `_build_estimator()` and their own `.train()`.
    """

    # Continuous columns -> median imputation + scaling.
    # Everything else (binary flags, one-hot, ordinal encodings) passes
    # through unchanged: already numeric, no missing values, and scaling
    # one-hot columns adds no value for a linear model.
    NUMERIC_FEATURES = [
        "living_area_m2", "bedrooms", "bathrooms", "latitude", "longitude",
        "building_year", "parking_count", "facades", "garden_area_m2",
        "total_area_m2", "km_from_nearby_city", "floor_number",
        "nearby_city_price_m2", "kitchen_equipped_encoded",
        "state_encoded", "epc_encoded",
    ]

    # One-hot encoded categorical columns share a common prefix, e.g.
    # pd.get_dummies(df["region"], prefix="region") -> "region_Brussels",
    # "region_Flanders", "region_Wallonia". Individually, each dummy
    # column can look unimportant to a model (it's just a 0/1 flag for
    # ONE category), but the CATEGORY as a whole (region, property
    # subtype, province...) can still be very informative once its
    # dummy columns are summed back together. Used by
    # grouped_feature_importance() below. Must match the `onehot_cols`
    # used in data_preprocessing.py's PreprocessData.preprocess_data().
    ONEHOT_GROUP_PREFIXES = ["property_subtype", "province", "region"]

    def __init__(self, filepath: str = None, df: pd.DataFrame = None,
                 target: str = "price", test_size: float = 0.2,
                 random_state: int = 42, log_target: bool = False):
        """Start from a CSV path or an already-loaded dataframe.

        log_target: True by default -- the model is fit on log1p(price)
        instead of raw price, since price is right-skewed (a long tail
        of expensive listings). This keeps squared-error-based fitting
        and tuning from being dominated by that tail. R2/RMSE/MAE
        reported by .evaluate() are ALWAYS converted back to real EUR
        first, so results stay directly interpretable. Set False to
        revert to fitting on raw price.
        """
        self.filepath = filepath
        self.df = df.copy() if df is not None else None
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.log_target = log_target

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_train_raw = self.y_test_raw = None
        self.preprocessor = None
        self.model = None          # set by .train() / .grid_search()
        self.pipeline = None       # preprocessor + model
        self.results_ = {}         # last computed evaluation scores

    # ---------- Loading ----------
    def load_data(self, **read_csv_kwargs):
        if self.df is None:
            self.df = pd.read_csv(self.filepath, **read_csv_kwargs)
        print(f"Data loaded -> shape {self.df.shape}")
        return self

    # ---------- Split ----------
    def split_data(self):
        df = self._check_loaded()
        X = df.drop(columns=[self.target])
        y = df[self.target]

        self.X_train, self.X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Original EUR values kept around no matter what: evaluate()
        # always scores against these, never against the log scale.
        self.y_train_raw, self.y_test_raw = y_train_raw, y_test_raw

        if self.log_target:
            self.y_train = np.log1p(y_train_raw)
            self.y_test = np.log1p(y_test_raw)
        else:
            self.y_train = y_train_raw
            self.y_test = y_test_raw

        print(f"Split -> train {self.X_train.shape}, test {self.X_test.shape}"
              + (" (log1p target)" if self.log_target else ""))
        self._build_preprocessor(X.columns)
        return self

    # ---------- Preprocessing (imputing + scaling) ----------
    def _build_preprocessor(self, all_columns):
        """
        Builds the ColumnTransformer without fitting it here: it is fit
        ONLY on X_train when the full pipeline is fitted, preventing any
        leakage from the test set into the training statistics.
        Continuous numeric columns -> impute (median) + StandardScaler.
        Remaining columns -> passthrough.
        """
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in all_columns]
        other_cols = [c for c in all_columns if c not in numeric_cols]

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("passthrough", "passthrough", other_cols),
        ])
        print(f"Preprocessor built -> {len(numeric_cols)} numeric cols "
              f"(impute+scale), {len(other_cols)} passthrough cols")
        return self

    # ---------- Estimator factory (must be implemented per model) ----------
    def _build_estimator(self, **kwargs):
        """
        Returns a fresh, unfitted estimator instance for this model
        (e.g. Ridge(...), Lasso(...), RandomForestRegressor(...)).
        Must be overridden by every subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_estimator()."
        )

    # ---------- Nested-parallelism guard (no-op unless overridden) ----------
    @contextlib.contextmanager
    def _avoid_nested_parallelism(self):
        """
        No-op by default: models like Ridge/DecisionTree have no
        internal n_jobs of their own, so there's nothing to protect
        against here.

        Overridden by RandomForestModel (and any future model whose
        estimator has its own internal n_jobs), which forces the
        estimator single-threaded for the duration of any OUTER
        cross-validation loop (GridSearchCV, RandomizedSearchCV,
        cross_val_score) that already parallelizes across
        folds/candidates with n_jobs=-1. Without this, two layers of
        parallelism compete for the same CPU cores (nested
        oversubscription), which can be dramatically SLOWER than no
        parallelism at all — this is used by grid_search(),
        random_search(), and cross_validate() below.
        """
        yield

    def _maybe_subsample(self, sample_frac: float = None):
        """
        Returns (X, y) -> either the full training set (sample_frac is
        None) or a random subsample of it, for use by grid_search()/
        random_search() when sample_frac is given.
        """
        if sample_frac is None:
            return self.X_train, self.y_train
        X_sub, _, y_sub, _ = train_test_split(
            self.X_train, self.y_train, train_size=sample_frac,
            random_state=self.random_state,
        )
        return X_sub, y_sub

    # ---------- Generic GridSearchCV ----------
    def grid_search(self, param_grid: dict, cv: int = 5, scoring: str = "r2",
                     sample_frac: float = None):
        """
        Generic GridSearchCV runner, usable by any subclass:
          - builds a fresh preprocessor + estimator pipeline
          - searches the given param_grid (keys must be prefixed with the
            sklearn step name, e.g. "ridge__alpha", "lasso__alpha")
          - stores the best pipeline in self.pipeline / self.model

        sample_frac: if given (0 < sample_frac < 1), the search itself
        runs on a random SUBSAMPLE of the training set instead of all
        of it, then automatically refits the winning pipeline on the
        FULL training set afterwards. Useful for estimators whose
        training cost scales badly with n (e.g. SVR with a non-linear
        kernel, roughly O(n^2)-O(n^3)) -- exploring many hyperparameter
        candidates on the full set can be extremely slow, while a
        subsample is usually enough to find the right region of the
        search space. Leave as None (default) for estimators that don't
        need this (Ridge, DecisionTree, RandomForest).

        Progress is printed as a single "start" line (with the total
        number of fits) and a single "done" line (with elapsed time) --
        sklearn's own per-fit verbose logging is kept OFF on purpose,
        since with 50-200+ fits it floods the console with one line per
        fit/fold and makes it harder to see what's actually going on.
        """
        self._check_split()
        X_search, y_search = self._maybe_subsample(sample_frac)

        n_candidates = 1
        for v in param_grid.values():
            n_candidates *= len(v)
        total_fits = n_candidates * cv
        print(f"[{self.__class__.__name__}] grid search starting -> "
              f"{n_candidates} candidate(s) x {cv} folds = {total_fits} fits"
              f"{f' (on a {len(X_search)}-row subsample)' if sample_frac else ''}...")
        t0 = time.time()

        with self._avoid_nested_parallelism():
            estimator = self._build_estimator()
            step_name = type(estimator).__name__.lower()
            base_pipeline = make_pipeline(self.preprocessor, estimator)

            grid = GridSearchCV(base_pipeline, param_grid, cv=cv,
                                 scoring=scoring, n_jobs=-1, verbose=0)
            grid.fit(X_search, y_search)

            self.pipeline = grid.best_estimator_
            if sample_frac is not None:
                print(f"[{self.__class__.__name__}] refitting best params "
                      f"on the FULL training set ({len(self.X_train)} rows)...")
                self.pipeline.fit(self.X_train, self.y_train)

        self.model = self.pipeline.named_steps[step_name]
        print(f"[{self.__class__.__name__}] grid search done in "
              f"{time.time() - t0:.1f}s -> best={grid.best_params_} "
              f"CV {scoring}={grid.best_score_:.4f}")
        return grid

    # ---------- Generic RandomizedSearchCV ----------
    def random_search(self, param_distributions: dict, n_iter: int = 30,
                       cv: int = 5, scoring: str = "r2",
                       sample_frac: float = None):
        """
        Generic RandomizedSearchCV runner, usable by any subclass:
          - builds a fresh preprocessor + estimator pipeline
          - samples `n_iter` random combinations from param_distributions
            (keys must be prefixed with the sklearn step name, e.g.
            "randomforestregressor__max_depth"; values can be a list
            OR a scipy.stats distribution such as randint/uniform)
          - stores the best pipeline in self.pipeline / self.model

        Why this exists (as opposed to only grid_search):
        For estimators with several interacting hyperparameters (typically
        ensembles like RandomForest/XGBoost), an exhaustive grid explodes
        combinatorially (e.g. 5 params x 5 values each = 3125 fits x cv).
        RandomizedSearchCV samples a fixed budget of combinations instead,
        which in practice finds near-optimal regions much faster.

        sample_frac: see grid_search() -- same idea, subsample during the
        search, then auto-refit the winner on the full training set.

        Progress is printed as a single "start" line (with the total
        number of fits) and a single "done" line (with elapsed time) --
        sklearn's own per-fit verbose logging is kept OFF on purpose,
        since with n_iter x cv fits (e.g. 200) it floods the console
        with one line per fit and makes it harder to see what's
        actually going on.
        """
        self._check_split()
        X_search, y_search = self._maybe_subsample(sample_frac)

        total_fits = n_iter * cv
        print(f"[{self.__class__.__name__}] random search starting -> "
              f"{n_iter} candidate(s) x {cv} folds = {total_fits} fits"
              f"{f' (on a {len(X_search)}-row subsample)' if sample_frac else ''}...")
        t0 = time.time()

        with self._avoid_nested_parallelism():
            estimator = self._build_estimator()
            step_name = type(estimator).__name__.lower()
            base_pipeline = make_pipeline(self.preprocessor, estimator)

            search = RandomizedSearchCV(
                base_pipeline, param_distributions, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, random_state=self.random_state,
                verbose=0,
            )
            search.fit(X_search, y_search)

            self.pipeline = search.best_estimator_
            if sample_frac is not None:
                print(f"[{self.__class__.__name__}] refitting best params "
                      f"on the FULL training set ({len(self.X_train)} rows)...")
                self.pipeline.fit(self.X_train, self.y_train)

        self.model = self.pipeline.named_steps[step_name]
        print(f"[{self.__class__.__name__}] random search done in "
              f"{time.time() - t0:.1f}s -> best={search.best_params_} "
              f"CV {scoring}={search.best_score_:.4f}")
        return search

    # ---------- Generic two-pass tuning ----------
    def two_pass_tune(self, param_name: str, coarse_values, cv: int = 5,
                       scoring: str = "r2"):
        """
        Reusable two-pass hyperparameter search for a single, strictly
        positive hyperparameter (e.g. Ridge/Lasso alpha, SVM C):

          Pass 1: coarse grid supplied by the caller (typically log-scale,
                  e.g. [0.001, 0.01, 0.1, 1, 10, 100, 1000]).
          Pass 2: fine grid centered on the pass-1 winner, spanning
                  best +/- 5*step, where step = best / 10.
                  (e.g. best=10 -> step=1 -> fine grid = [5, 6, ..., 15])

        param_name: the raw hyperparameter name, WITHOUT the pipeline step
                    prefix (e.g. "alpha", not "ridge__alpha") -> the step
                    prefix is derived automatically from the estimator.

        Returns (grid1, grid2), the two fitted GridSearchCV objects.
        """
        estimator = self._build_estimator()
        step_name = type(estimator).__name__.lower()
        full_param = f"{step_name}__{param_name}"

        # ---- Pass 1: coarse sweep ----
        grid1 = self.grid_search({full_param: coarse_values}, cv=cv, scoring=scoring)
        best_coarse = grid1.best_params_[full_param]
        print(f"[{self.__class__.__name__}] Pass 1 best CV {scoring}: "
              f"{grid1.best_score_:.4f} ({param_name}={best_coarse})")

        # ---- Pass 2: fine grid centered on the pass-1 winner ----
        # If every coarse value was an int (e.g. max_depth, n_estimators),
        # keep the fine grid integer too -> round + dedupe + clip to >= 1,
        # since sklearn's param validation rejects floats like 11.0 for
        # integer-only hyperparameters.
        is_int_param = all(isinstance(v, (int, np.integer)) for v in coarse_values)
        step = best_coarse / 10
        raw_fine_values = [best_coarse + i * step for i in range(-5, 6)]

        if is_int_param:
            fine_values = sorted({max(1, round(v)) for v in raw_fine_values})
        else:
            fine_values = [v for v in raw_fine_values if v > 0]  # keep strictly positive

        grid2 = self.grid_search({full_param: fine_values}, cv=cv, scoring=scoring)
        best_fine = grid2.best_params_[full_param]
        print(f"[{self.__class__.__name__}] Pass 2 best CV {scoring}: "
              f"{grid2.best_score_:.4f} ({param_name}={best_fine})")

        return grid1, grid2

    # ---------- Scoring (shared by every subclass) ----------
    def evaluate(self, on: str = "test"):
        """Scores the fitted pipeline on the 'test' or 'train' split.
        Always in real EUR, even if log_target=True (model fits/predicts
        in log space internally, but predictions are converted back with
        expm1() before R2/RMSE/MAE are computed -- so a log_target run
        and a raw run report directly comparable numbers)."""
        if self.pipeline is None:
            raise ValueError("No trained pipeline yet. Call .train() first.")

        if on == "test":
            X, y_true = self.X_test, self.y_test_raw
        else:
            X, y_true = self.X_train, self.y_train_raw
        y_pred = self.pipeline.predict(X)
        if self.log_target:
            y_pred = np.expm1(y_pred)

        r2 = r2_score(y_true, y_pred)
        n, p = X.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        self.results_[on] = {"R2": r2, "adj_R2": adj_r2, "RMSE": rmse, "MAE": mae}
        print(f"[{self.__class__.__name__}] {on:>5} -> "
              f"R2={r2:.4f}  adj_R2={adj_r2:.4f}  "
              f"RMSE={rmse:,.0f}  MAE={mae:,.0f}")
        return self.results_[on]

    def cross_validate(self, cv: int = 5, scoring: str = "r2"):
        """K-fold CV on the training set (full pipeline -> no leakage).
        Prints a single start/done line with elapsed time instead of
        sklearn's one-line-per-fold verbose output."""
        if self.pipeline is None:
            raise ValueError("No pipeline yet. Call .train() first.")
        print(f"[{self.__class__.__name__}] {cv}-fold CV starting...")
        t0 = time.time()
        with self._avoid_nested_parallelism():
            scores = cross_val_score(self.pipeline, self.X_train, self.y_train,
                                      cv=cv, scoring=scoring, n_jobs=-1,
                                      verbose=0)
        print(f"[{self.__class__.__name__}] {cv}-fold CV done in "
              f"{time.time() - t0:.1f}s -> {scoring}: "
              f"{scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    # ---------- Anti-overfitting guard (shared by every subclass) ----------
    def _current_hyperparams(self) -> dict:
        """
        Returns the CURRENTLY fitted model's hyperparameters as a plain
        dict usable directly with self.train(**params) -- i.e. using
        the same argument names train() itself declares. Must be
        overridden by every subclass (its own train() signature is the
        source of truth for which names matter).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _current_hyperparams()."
        )

    def _stronger_regularization(self, params: dict) -> dict:
        """
        Returns a MORE regularized (simpler) version of the given
        hyperparameters -- the specific knob differs per model family
        (e.g. raise alpha for Ridge, raise min_samples_leaf for trees,
        lower C for SVM, raise reg_alpha/reg_lambda for XGBoost).
        No-op by default; every subclass overrides this with its own
        recipe. Used by reduce_overfitting() below.
        """
        return dict(params)

    def reduce_overfitting(self, gap_threshold: float = 0.10,
                            max_attempts: int = 3):
        """
        Generic anti-overfitting guard, run AFTER hyperparameter tuning
        already picked a "best" set of params via cross-validation.

        Why this is needed even after CV-based tuning: cross-validation
        already reduces overfitting risk somewhat (a badly overfit
        model scores poorly on the held-out folds it wasn't trained
        on), but for some model families a single tuned hyperparameter
        doesn't fully control model complexity -- Decision Tree is the
        clearest example: max_depth alone can still allow tiny,
        hyper-specific leaves at any depth. min_samples_leaf is
        actually the stronger lever against that, but two_pass_tune()
        only searches one parameter at a time.

        This method retrains up to `max_attempts` times, each time
        calling self._stronger_regularization(params) to nudge the
        hyperparameters toward a MORE regularized (simpler) model,
        stopping as soon as the Train/Test R2 gap drops to
        gap_threshold or below, or once _stronger_regularization()
        stops changing anything (no more regularization headroom),
        whichever comes first.

        Must be called AFTER the model has already been trained once
        (self.model must exist) -- it reads the starting point via
        _current_hyperparams().

        Returns (final_params, train_stats, test_stats, attempts_used).
        """
        if self.model is None:
            raise ValueError("No trained model yet. Call .train() first.")

        current_params = self._current_hyperparams()
        train_stats = self.evaluate(on="train")
        test_stats = self.evaluate(on="test")
        gap = train_stats["R2"] - test_stats["R2"]

        attempt = 0
        while gap > gap_threshold and attempt < max_attempts:
            next_params = self._stronger_regularization(current_params)
            if next_params == current_params:
                break  # no more regularization headroom -> stop early
            current_params = next_params
            self.train(**current_params)
            train_stats = self.evaluate(on="train")
            test_stats = self.evaluate(on="test")
            gap = train_stats["R2"] - test_stats["R2"]
            attempt += 1

        return current_params, train_stats, test_stats, attempt

    # ---------- Feature importance (shared by every subclass) ----------
    def _get_feature_names_out(self):
        """
        Returns the column names AFTER the ColumnTransformer, in the
        exact order the fitted estimator sees them (numeric/scaled cols
        first, then passthrough cols) -> required to correctly label
        coef_ / feature_importances_, since those are plain arrays with
        no column names attached.
        """
        raw_names = self.preprocessor.get_feature_names_out()
        # ColumnTransformer prefixes names with the step id ("num__",
        # "passthrough__") -> strip that back off for readability.
        return [n.split("__", 1)[1] if "__" in n else n for n in raw_names]

    def _compute_importance_df(self):
        """
        Core computation shared by feature_importance() and
        grouped_feature_importance(): returns (df, metric_name) where df
        has one row per RAW (post-one-hot) feature, UNSORTED. Doing the
        dispatch/extraction once here avoids duplicating the
        feature_importances_ vs coef_ logic in both public methods.
        """
        if self.pipeline is None:
            raise ValueError("No trained pipeline yet. Call .train() first.")

        feature_names = self._get_feature_names_out()

        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
            metric = "importance"
        elif hasattr(self.model, "coef_"):
            values = np.abs(np.ravel(self.model.coef_))
            metric = "abs_coefficient"
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}'s estimator exposes neither "
                "feature_importances_ nor coef_ -> cannot rank features."
            )

        return pd.DataFrame({"feature": feature_names, metric: values}), metric

    def feature_importance(self, top_n: int = 5):
        """
        Ranks every RAW feature (each one-hot dummy counted separately,
        e.g. "region_Brussels", "region_Flanders", "region_Wallonia" as
        3 distinct rows) by how much it drives this model's predictions,
        and prints/returns the top_n MOST and top_n LEAST important ones.

        NOTE: for a one-hot encoded category, this often makes every
        individual dummy look unimportant even when the category as a
        whole matters a lot (its effect is split across N columns
        instead of concentrated in 1). Use grouped_feature_importance()
        to see the category-level picture instead.

        Dispatches automatically depending on what the fitted estimator
        exposes:
          - tree-based models (DecisionTree, RandomForest, ...) expose
            `feature_importances_` (mean impurity decrease brought by
            each feature, across all trees for RF) -> used directly.
          - linear models (Ridge, Lasso, ...) expose `coef_` -> the
            ABSOLUTE VALUE of the coefficient is used as the importance
            proxy. This is only a fair comparison because the numeric
            features are StandardScaled (mean 0, std 1) in the shared
            preprocessor, so their coefficients are already on a
            comparable scale. NOTE: the passthrough columns (one-hot
            flags, ordinal encodings) are NOT scaled, so their
            coefficient magnitude is not perfectly comparable to the
            scaled numeric ones -> treat the linear-model ranking as
            indicative, not exact, for those specific columns.

        Returns (top_df, bottom_df): two DataFrames with columns
        ["feature", <importance metric>], sorted from most to least
        important. Also stored in self.feature_importance_ (full
        ranking, every raw feature).
        """
        df_imp, metric = self._compute_importance_df()
        if metric == "abs_coefficient":
            print(f"[{self.__class__.__name__}] NOTE: ranking is based on "
                  "|coefficient|. Valid across the StandardScaled numeric "
                  "features; passthrough (one-hot/ordinal) columns are on "
                  "a different scale so treat their rank as indicative.")

        df_imp = df_imp.sort_values(metric, ascending=False).reset_index(drop=True)
        self.feature_importance_ = df_imp

        top = df_imp.head(top_n)
        bottom = df_imp.tail(top_n).sort_values(metric)  # least important first

        print(f"\n[{self.__class__.__name__}] Top {top_n} MOST important features:")
        print(top.to_string(index=False))
        print(f"\n[{self.__class__.__name__}] Top {top_n} LEAST important features:")
        print(bottom.to_string(index=False))

        return top, bottom

    def grouped_feature_importance(self, group_prefixes=None, top_n: int = 5):
        """
        Same idea as feature_importance(), but first collapses one-hot
        dummy columns that share a common categorical origin back into
        a single "category" row by SUMMING their individual importance
        (or |coefficient|) values -> e.g. "region_Brussels" (0.004) +
        "region_Flanders" (0.003) + "region_Wallonia" (0.002) all
        collapse into one "region" row (0.009), instead of 3 separately
        unimpressive rows hiding a genuinely important category.

        Why summing is the right aggregation here: for tree-based
        feature_importances_, importance is (roughly) the total
        impurity decrease attributed to splits on that column, across
        the whole forest -> these are already additive by construction,
        so summing the dummies recovers "how much impurity decrease is
        attributable to the region variable overall". For |coef_|
        (Ridge), summing is a reasonable proxy for "total influence of
        this category" though it is less exact than for tree importances
        (see the coefficient-scale caveat in feature_importance()).

        group_prefixes: list of column-name prefixes to collapse (each
        prefix matches "{prefix}_*"). Defaults to
        self.ONEHOT_GROUP_PREFIXES (["property_subtype", "province",
        "region"] -> matches the onehot_cols used in
        data_preprocessing.py). Any feature NOT matching one of these
        prefixes is kept as its own individual group (e.g. "bedrooms",
        "state_encoded" stay ungrouped, since they're not one-hot).

        Returns (top_df, bottom_df): two DataFrames with columns
        ["group", <importance metric>]. Also stored in
        self.grouped_feature_importance_ (full ranking, every group).
        """
        df_imp, metric = self._compute_importance_df()
        if metric == "abs_coefficient":
            print(f"[{self.__class__.__name__}] NOTE: ranking is based on "
                  "summed |coefficient| per category, an indicative proxy "
                  "rather than an exact category-level effect.")

        prefixes = group_prefixes or self.ONEHOT_GROUP_PREFIXES

        def assign_group(feature_name):
            for prefix in prefixes:
                if feature_name.startswith(prefix + "_"):
                    return prefix
            return feature_name  # not one-hot -> stays its own group

        df_imp["group"] = df_imp["feature"].apply(assign_group)
        grouped = (df_imp.groupby("group", as_index=False)[metric]
                   .sum()
                   .sort_values(metric, ascending=False)
                   .reset_index(drop=True))
        self.grouped_feature_importance_ = grouped

        top = grouped.head(top_n)
        bottom = grouped.tail(top_n).sort_values(metric)

        print(f"\n[{self.__class__.__name__}] Top {top_n} MOST important "
              "GROUPS (one-hot categories summed):")
        print(top.to_string(index=False))
        print(f"\n[{self.__class__.__name__}] Top {top_n} LEAST important "
              "GROUPS (one-hot categories summed):")
        print(bottom.to_string(index=False))

        return top, bottom

    # ---------- Helpers ----------
    def _check_loaded(self):
        if self.df is None:
            raise ValueError("No data loaded. Call .load_data() first.")
        return self.df

    def _check_split(self):
        if self.X_train is None:
            raise ValueError("Data not split yet. Call .split_data() first.")


# ====================================================================
#  RIDGE MODEL
# ====================================================================
class RidgeModel(BaseModel):
    """Ridge regression. Tuning logic (grid_search/two_pass_tune) is
    inherited from BaseModel."""

    def _build_estimator(self, **kwargs):
        return Ridge(random_state=self.random_state, **kwargs)

    def train(self, alpha: float = 1.0, **ridge_kwargs):
        """Trains a Ridge model with a fixed alpha."""
        self._check_split()
        self.model = self._build_estimator(alpha=alpha, **ridge_kwargs)
        self.pipeline = make_pipeline(self.preprocessor, self.model)
        self.pipeline.fit(self.X_train, self.y_train)
        print(f"[RidgeModel] trained with alpha={alpha}")
        return self

    def _current_hyperparams(self):
        return {"alpha": self.model.alpha}

    def _stronger_regularization(self, params):
        # alpha is Ridge's ONLY regularization knob -> simply raise it.
        # Capped at 1e6 so the loop naturally stops once it's clearly
        # not helping anymore (next_params == current_params).
        new_alpha = min(params["alpha"] * 3, 1e6)
        return {"alpha": new_alpha}


# ====================================================================
#  DECISION TREE MODEL
# ====================================================================
class DecisionTreeModel(BaseModel):
    """
    Decision Tree regression. Tuning logic (grid_search/two_pass_tune) is
    inherited from BaseModel.

    Note: trees are scale-invariant (splits are based on thresholds, not
    distances), so StandardScaler in the shared preprocessor has no real
    effect here — it's harmless, just unnecessary. The median-imputation
    step, however, IS still required: unlike some gradient-boosting
    implementations, sklearn's DecisionTreeRegressor cannot handle NaNs
    natively.
    """

    def _build_estimator(self, **kwargs):
        return DecisionTreeRegressor(random_state=self.random_state, **kwargs)

    def train(self, max_depth: int = None, **tree_kwargs):
        """Trains a Decision Tree with a fixed max_depth (None = unlimited).
        Other regularization knobs (min_samples_leaf, min_samples_split,
        ...) can be passed via tree_kwargs -- used by
        reduce_overfitting()'s _stronger_regularization() below, since
        max_depth alone is a fairly weak lever against tree overfitting
        compared to bounding leaf size."""
        self._check_split()
        self.model = self._build_estimator(max_depth=max_depth, **tree_kwargs)
        self.pipeline = make_pipeline(self.preprocessor, self.model)
        self.pipeline.fit(self.X_train, self.y_train)
        print(f"[DecisionTreeModel] trained with max_depth={max_depth}, "
              f"{tree_kwargs}")
        return self

    def _current_hyperparams(self):
        return {
            "max_depth": self.model.max_depth,
            "min_samples_leaf": self.model.min_samples_leaf,
            "min_samples_split": self.model.min_samples_split,
        }

    def _stronger_regularization(self, params):
        """
        min_samples_leaf is the STRONGER lever here: a tree can still
        overfit at a shallow-looking max_depth if it's allowed to carve
        out tiny, hyper-specific leaves (e.g. a leaf matching just 1-2
        training rows). Raising min_samples_leaf directly bounds how
        specific any single leaf can get, which max_depth alone doesn't
        guarantee. max_depth is also tightened as a secondary lever.
        """
        new_params = dict(params)

        leaf = params["min_samples_leaf"]
        new_leaf = max(leaf * 3, leaf + 5)
        new_params["min_samples_leaf"] = min(new_leaf, 200)
        new_params["min_samples_split"] = max(
            new_params["min_samples_leaf"] * 2, params["min_samples_split"]
        )

        depth = params["max_depth"]
        if depth is None or depth > 10:
            new_params["max_depth"] = 10
        elif depth > 3:
            new_params["max_depth"] = depth - 2
        # else: already shallow, leave max_depth as-is and rely on the
        # min_samples_leaf increase above.

        return new_params


# ====================================================================
#  RANDOM FOREST MODEL
# ====================================================================
class RandomForestModel(BaseModel):
    """
    Random Forest regression.

    Why RF needs a DIFFERENT tuning strategy than Ridge/DecisionTree
    ---------------------------------------------------------------
    Ridge has ONE hyperparameter (alpha) and DecisionTree's most
    important one is max_depth -> a single-parameter two-pass sweep
    (coarse grid, then fine grid centered on the winner) makes sense for
    both, because the search space is 1-D.

    Random Forest has SEVERAL hyperparameters that interact with each
    other (n_estimators, max_depth, max_features, min_samples_split,
    min_samples_leaf, bootstrap). Two-pass tuning one parameter at a
    time would silently assume they don't interact, which is false for
    RF (e.g. the ideal max_depth depends on min_samples_leaf, and the
    ideal max_features depends on how many correlated features exist).
    An exhaustive GridSearchCV over all of them combinatorially explodes
    (e.g. 5 values x 5 params = 3125 fits x cv folds), so this class
    instead uses a strategy in two steps, both scale-invariant since
    trees don't need feature scaling (StandardScaler in the shared
    preprocessor is harmless but unnecessary here, same as for the
    single Decision Tree):

      1) find_n_estimators_elbow(): n_estimators is special among RF
         hyperparameters -> more trees essentially never hurts
         generalization (it only reduces variance by averaging), it
         just costs more compute with diminishing returns. So instead
         of tuning it jointly with the others, we find the "elbow"
         where adding more trees stops meaningfully improving the
         score, using the model's built-in OOB (out-of-bag) score.
         OOB score reuses the ~37% of rows left out of each tree's
         bootstrap sample as a free internal validation set, so this
         needs a SINGLE fit per n_estimators value instead of a full
         cv-fold GridSearchCV -> much faster than CV for this step.

      2) random_search_tune(): once n_estimators is fixed (or capped),
         RandomizedSearchCV samples random combinations of the
         remaining, genuinely interacting hyperparameters
         (max_depth, max_features, min_samples_split, min_samples_leaf)
         from distributions rather than a fixed grid -> covers the
         joint space far more efficiently than GridSearchCV for the
         same compute budget.
    """

    def _build_estimator(self, **kwargs):
        # n_jobs is controlled via self._rf_n_jobs (default -1, i.e. "use
        # all cores"), and NOT hardcoded here, because it needs to change
        # depending on the context: a lone fit (train(),
        # find_n_estimators_elbow()) wants n_jobs=-1, but a fit happening
        # INSIDE an outer CV loop (grid_search/random_search/
        # cross_validate, all n_jobs=-1 themselves) wants n_jobs=1 to
        # avoid nested oversubscription. See _avoid_nested_parallelism()
        # below, which all three of those methods now use automatically.
        n_jobs = kwargs.pop("n_jobs", getattr(self, "_rf_n_jobs", -1))
        return RandomForestRegressor(
            random_state=self.random_state, n_jobs=n_jobs, **kwargs
        )

    @contextlib.contextmanager
    def _avoid_nested_parallelism(self):
        """
        Overrides BaseModel's no-op version. RandomForestRegressor has
        its own internal n_jobs, so this needs to handle TWO cases:

          1) self.model doesn't exist yet, or exists but isn't the one
             about to be (re)fit -> grid_search()/random_search() call
             self._build_estimator() fresh, which reads self._rf_n_jobs
             -> set it to 1 here so the freshly-built estimator is
             single-threaded for the duration of the outer search.

          2) self.model already exists AND IS FITTED (cross_validate()
             re-fits the SAME already-trained pipeline across folds) ->
             self._rf_n_jobs alone won't help, since that estimator
             object was already constructed with whatever n_jobs it had
             at train() time. n_jobs is a plain mutable attribute read
             at fit/predict time (not baked into the fitted trees), so
             it's safe to flip it in place and restore it after.
        """
        self._rf_n_jobs = 1
        had_model_n_jobs = self.model is not None and hasattr(self.model, "n_jobs")
        if had_model_n_jobs:
            prev_model_n_jobs = self.model.n_jobs
            self.model.n_jobs = 1
        try:
            yield
        finally:
            self._rf_n_jobs = -1
            if had_model_n_jobs:
                self.model.n_jobs = prev_model_n_jobs

    def train(self, n_estimators: int = 300, max_depth: int = None,
              max_features="sqrt", min_samples_split: int = 2,
              min_samples_leaf: int = 1, bootstrap: bool = True,
              oob_score: bool = True, **rf_kwargs):
        """
        Trains a Random Forest with fixed hyperparameters.
        oob_score=True (default) requires bootstrap=True and lets you
        read self.model.oob_score_ right after fitting as a free,
        CV-free estimate of generalization performance.
        """
        self._check_split()
        self.model = self._build_estimator(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
            oob_score=oob_score if bootstrap else False, **rf_kwargs
        )
        self.pipeline = make_pipeline(self.preprocessor, self.model)
        self.pipeline.fit(self.X_train, self.y_train)
        print(f"[RandomForestModel] trained with n_estimators={n_estimators}, "
              f"max_depth={max_depth}, max_features={max_features}, "
              f"min_samples_split={min_samples_split}, "
              f"min_samples_leaf={min_samples_leaf}")
        if bootstrap and oob_score:
            print(f"[RandomForestModel] OOB R2 score: {self.model.oob_score_:.4f}")
        return self

    def _current_hyperparams(self):
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "max_features": self.model.max_features,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "bootstrap": self.model.bootstrap,
        }

    def _stronger_regularization(self, params):
        """
        Same lever as DecisionTreeModel (min_samples_leaf is the
        stronger regularizer against tiny, overfit leaves), applied per
        tree in the forest. n_estimators and max_features are left
        untouched -- more trees doesn't cause overfitting (see
        find_n_estimators_elbow()'s docstring) and max_features is
        categorical, not a natural "dial up/down" knob.
        """
        new_params = dict(params)

        leaf = params["min_samples_leaf"]
        new_leaf = max(leaf * 3, leaf + 5)
        new_params["min_samples_leaf"] = min(new_leaf, 100)
        new_params["min_samples_split"] = max(
            new_params["min_samples_leaf"] * 2, params["min_samples_split"]
        )

        depth = params["max_depth"]
        if depth is None or depth > 15:
            new_params["max_depth"] = 15
        elif depth > 4:
            new_params["max_depth"] = depth - 2

        return new_params

    # ---------- Step 1: n_estimators elbow (OOB-based, no CV needed) ----------
    def find_n_estimators_elbow(self, n_values=(50, 100, 200, 300, 500, 800, 1200),
                                 min_gain: float = 0.001, patience: int = 2,
                                 **fixed_rf_kwargs):
        """
        Fits RFs with increasing n_estimators (bootstrap=True,
        oob_score=True), ONE AT A TIME IN ORDER, and STOPS EARLY as soon
        as the OOB R2 gain plateaus -> it does NOT test every value in
        n_values up front. Because OOB score only needs a single fit per
        n_estimators value (no k-fold refitting), this is already fast
        per step, but skipping the larger, more expensive values once
        they're clearly not needed anymore saves real time (no reason
        to fit 800 and 1200 trees if 300 already plateaued).

        Why stopping on the first small/negative gain is the RIGHT call
        for Random Forest specifically: adding trees can only help or be
        neutral in expectation (more trees = averaging over more
        bootstrap samples = lower variance, the underlying signal being
        modeled doesn't change) -> there is no real mechanism by which
        MORE trees produces a WORSE model. So when you see the OOB score
        dip slightly between two steps (e.g. 300 -> 500 going down),
        that is OOB-score measurement noise (each fit uses a different
        bootstrap, so the "left-out" validation rows differ each time),
        NOT the model degrading, and it should NOT be read as "still
        improving, keep pushing n higher" -- chasing that noise is
        exactly why the old version kept climbing all the way to 1200
        for a ~0.006 R2 difference that's mostly noise anyway.

        patience: how many CONSECUTIVE steps with a gain below min_gain
        are required before declaring a plateau and stopping (default 2,
        so one noisy small/negative step doesn't trigger a premature
        stop by itself, but two in a row does). Set patience=1 to stop
        on the very first small gain.

        Returns the selected n_estimators (the LAST value actually
        fitted, i.e. right where the plateau was confirmed -- there's no
        reason to step back down to an earlier, smaller n once it's been
        fitted, since it can only be equal-or-better than the smaller
        ones). Also stores the (possibly partial) sweep in
        self.n_estimators_scores_.
        """
        self._check_split()
        # The preprocessor needs to be fit once on X_train; reuse the same
        # fitted transform for every n_estimators value to keep this fast
        # and to isolate n_estimators as the only thing that changes.
        X_train_transformed = self.preprocessor.fit_transform(self.X_train)

        scores = []
        small_gain_streak = 0
        for n in n_values:
            rf = self._build_estimator(
                n_estimators=n, bootstrap=True, oob_score=True,
                **fixed_rf_kwargs
            )
            rf.fit(X_train_transformed, self.y_train)
            scores.append((n, rf.oob_score_))
            print(f"[RandomForestModel] n_estimators={n:>5} -> "
                  f"OOB R2={rf.oob_score_:.4f}")

            if len(scores) >= 2:
                gain = scores[-1][1] - scores[-2][1]
                small_gain_streak = small_gain_streak + 1 if gain < min_gain else 0
                if small_gain_streak >= patience:
                    print(f"[RandomForestModel] plateau confirmed "
                          f"({patience} consecutive step(s) with gain < "
                          f"{min_gain}) -> stopping early, skipping the "
                          f"remaining, larger n_estimators values.")
                    break

        self.n_estimators_scores_ = scores
        best_n = scores[-1][0]

        print(f"[RandomForestModel] elbow selected -> n_estimators={best_n}")
        self.selected_n_estimators_ = best_n
        return best_n

    # ---------- Step 2: joint tuning of the remaining, interacting params ----------
    def random_search_tune(self, n_iter: int = 40, cv: int = 5,
                            scoring: str = "r2", n_estimators_search: int = 200,
                            param_distributions: dict = None):
        """
        RandomizedSearchCV over the RF hyperparameters that genuinely
        interact with each other. n_estimators is fixed (not searched)
        here on purpose -> use find_n_estimators_elbow() first, since
        searching it jointly would waste search budget on a parameter
        whose effect (more trees = less variance, more compute) is
        already well understood and monotonic, unlike the others.

        n_estimators_search: number of trees used for EVERY fit DURING
        the search (default 200), deliberately NOT the full elbow value
        (e.g. 1200). This is the second big speed fix: this search does
        n_iter * cv fits (e.g. 40 * 5 = 200 fits) -> running all of them
        with 1200 trees each multiplies the total time by ~6x compared
        to 200 trees, for basically no benefit, since which max_depth /
        max_features / min_samples_* combination wins barely changes
        with the number of trees (more trees only reduces variance, it
        doesn't change the ranking of hyperparameter combinations).
        The full elbow n_estimators is added back afterwards, once, for
        the final production fit -> call train_best() right after this.

        param_distributions: override the default search space if
        needed. Must use the "randomforestregressor__" prefix. Default
        space below mixes discrete choices and continuous/integer
        distributions (scipy.stats), which RandomizedSearchCV samples
        from directly -> a genuinely finer-grained search than any
        fixed grid of the same size.
        """
        if param_distributions is None:
            param_distributions = {
                "randomforestregressor__n_estimators": [n_estimators_search],
                "randomforestregressor__max_depth": [None, 5, 10, 15, 20, 25, 30],
                "randomforestregressor__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
                "randomforestregressor__min_samples_split": randint(2, 20),
                "randomforestregressor__min_samples_leaf": randint(1, 10),
                "randomforestregressor__bootstrap": [True],
            }

        # Nested-parallelism protection (single-threaded RF fits while
        # RandomizedSearchCV parallelizes across n_iter * cv fits) is
        # handled automatically inside random_search() via
        # self._avoid_nested_parallelism() -> nothing to do here.
        search = self.random_search(param_distributions, n_iter=n_iter,
                                     cv=cv, scoring=scoring)

        self.best_params_ = search.best_params_
        return search

    # ---------- Step 3: final production fit with the full n_estimators ----------
    def train_best(self, n_estimators: int = None):
        """
        Retrains the final model using the winning hyperparameters found
        by random_search_tune() (self.best_params_), but swapping back
        in the FULL n_estimators from find_n_estimators_elbow() (or an
        explicit value) instead of the smaller n_estimators_search used
        during the search itself.

        n_estimators: defaults to self.selected_n_estimators_ (set by
        find_n_estimators_elbow()), falling back to 300 if that was
        never run.

        This is a single fit (n_jobs=-1 is safe and fast here, no outer
        CV parallelism to collide with).
        """
        if not hasattr(self, "best_params_"):
            raise ValueError(
                "No tuned hyperparameters yet. Call random_search_tune() first."
            )
        final_n = n_estimators or getattr(self, "selected_n_estimators_", 300)

        # self.best_params_ keys are prefixed ("randomforestregressor__max_depth")
        # because they come straight out of the pipeline's GridSearchCV/
        # RandomizedSearchCV -> strip the prefix to pass them as plain
        # train() kwargs, and drop n_estimators since we override it here.
        clean_params = {
            k.split("__", 1)[1]: v for k, v in self.best_params_.items()
            if not k.endswith("n_estimators")
        }

        print(f"[RandomForestModel] final fit -> n_estimators={final_n}, "
              f"{clean_params}")
        self.train(n_estimators=final_n, **clean_params)
        return self


# ====================================================================
#  SVM MODEL (Support Vector Regression)
# ====================================================================
class SVMModel(BaseModel):
    """
    Support Vector Regression (SVR).

    Why SVM needs a DIFFERENT tuning strategy than the others
    ---------------------------------------------------------
    SVR training cost scales roughly O(n^2) to O(n^3) with the number
    of training rows for non-linear kernels (rbf/poly) -- unlike trees
    (closer to O(n log n) per tree) or Ridge (closed-form solve). With
    ~10k training rows, running a full RandomizedSearchCV(cv=5) over
    kernel/C/epsilon/gamma on the ENTIRE training set can be extremely
    slow -- and, same as RandomForest, kernel/C/epsilon/gamma interact
    with each other, so a single-param two-pass sweep isn't appropriate
    either.

    Strategy used here ("search cheap, finalize expensive", same spirit
    as RandomForestModel's elbow -> random search -> final fit, but
    adapted to SVM's actual bottleneck which is TRAINING COST, not a
    monotonic n_estimators-like parameter):
      1) quick_kernel_search(): tests a handful of kernels with cheap
         cross-validation on a small SUBSAMPLE of the training data
         (default 2,000 rows) to see which kernel family is even
         competitive, before spending real time on it.
      2) random_search_tune(): RandomizedSearchCV over C / epsilon /
         gamma (and degree for poly) for the winning kernel, ALSO run
         on a subsample by default (sample_frac) for speed, then
         automatically refits the winning pipeline on the FULL training
         set for the final, production-quality model (see
         BaseModel.random_search's sample_frac parameter).

    Feature scaling note: SVR is a DISTANCE-based method (unlike trees),
    so it is sensitive to feature scale on every input dimension,
    including the one-hot columns. The shared preprocessor only
    StandardScales the continuous NUMERIC_FEATURES and passes the
    one-hot/ordinal columns through as raw 0/1 values -- a reasonable
    approximation (0/1 is roughly the same order of magnitude as
    standardized features) but not perfectly ideal for SVR specifically.

    feature_importance()/grouped_feature_importance() only work when
    kernel="linear" (SVR only exposes `coef_` for a linear kernel -- for
    rbf/poly/sigmoid kernels there is no per-feature coefficient or
    importance to extract; this is a fundamental property of those
    kernels, not a limitation of this code).
    """

    def _build_estimator(self, **kwargs):
        return SVR(**kwargs)

    def train(self, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1,
              gamma="scale", **svr_kwargs):
        """Trains an SVR with fixed hyperparameters."""
        self._check_split()
        self.model = self._build_estimator(
            kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, **svr_kwargs
        )
        self.pipeline = make_pipeline(self.preprocessor, self.model)
        self.pipeline.fit(self.X_train, self.y_train)
        print(f"[SVMModel] trained with kernel={kernel}, C={C}, "
              f"epsilon={epsilon}, gamma={gamma}")
        return self

    def _current_hyperparams(self):
        return {
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
            "gamma": self.model.gamma,
        }

    def _stronger_regularization(self, params):
        """
        Lower C = the model tolerates more points outside the tube
        instead of twisting itself to fit every one of them (SVM's
        main regularization knob). Raising epsilon (widening the
        no-penalty tube) is a secondary lever -- more points fall
        "close enough" and stop contributing to the fit at all.
        """
        new_params = dict(params)
        new_params["C"] = max(params["C"] / 3, 1e-3)
        new_params["epsilon"] = min(params["epsilon"] * 1.5, 2.0)
        return new_params

    # ---------- Step 1: cheap kernel screening on a subsample ----------
    def quick_kernel_search(self, kernels=("linear", "rbf", "poly"),
                             sample_size: int = 2000, cv: int = 3,
                             scoring: str = "r2"):
        """
        Cross-validates each candidate kernel (default hyperparameters
        otherwise) on a small random SUBSAMPLE of the training set --
        cheap and fast -- to see which kernel family is worth spending
        real search budget on, before running random_search_tune() on
        the full-cost search.

        sample_size: number of TRAINING rows to subsample (capped at
        len(X_train) if smaller). Kept small on purpose since this is
        only meant to rank kernels relative to each other, not to find
        final hyperparameters.

        Returns the winning kernel name (str), and stores the full
        ranking in self.kernel_scores_ (dict of kernel -> mean CV score).
        Also sets self.selected_kernel_, used automatically by
        random_search_tune() afterwards.
        """
        self._check_split()
        n = min(sample_size, len(self.X_train))
        X_sample, _, y_sample, _ = train_test_split(
            self.X_train, self.y_train, train_size=n, random_state=self.random_state
        )

        print(f"[SVMModel] quick kernel search starting -> {len(kernels)} "
              f"kernel(s) x {cv} folds on a {n}-row subsample...")
        t0 = time.time()

        scores = {}
        for kernel in kernels:
            estimator = self._build_estimator(kernel=kernel)
            pipeline = make_pipeline(self.preprocessor, estimator)
            cv_scores = cross_val_score(pipeline, X_sample, y_sample,
                                         cv=cv, scoring=scoring, n_jobs=-1)
            scores[kernel] = cv_scores.mean()

        self.kernel_scores_ = scores
        best_kernel = max(scores, key=scores.get)
        print(f"[SVMModel] quick kernel search done in {time.time() - t0:.1f}s "
              f"-> {scores} -> selected kernel={best_kernel}")
        self.selected_kernel_ = best_kernel
        return best_kernel

    # ---------- Step 2: joint tuning of the remaining, interacting params ----------
    def random_search_tune(self, n_iter: int = 30, cv: int = 5,
                            scoring: str = "r2", kernel: str = None,
                            sample_frac: float = 0.3,
                            param_distributions: dict = None):
        """
        RandomizedSearchCV over C / epsilon / gamma (and degree for
        "poly") for a FIXED kernel -- kernel itself is chosen ahead of
        time by quick_kernel_search() (or passed explicitly here), not
        searched jointly, since it changes what the other hyperparameters
        even mean (e.g. gamma is meaningless for "linear").

        kernel: defaults to self.selected_kernel_ if
        quick_kernel_search() was already run, otherwise "rbf".

        sample_frac: fraction of the training set used DURING the
        search (default 0.3 -- fits on ~30% of the rows to keep the
        O(n^2)-O(n^3) SVR training cost manageable across n_iter * cv
        fits). The winning pipeline is automatically refit on the FULL
        training set afterwards (handled by random_search()). Set to
        None to search on the full training set instead (slower, but
        more reliable if your machine can afford it).

        param_distributions: override the default search space if
        needed. Must use the "svr__" prefix.
        """
        kernel = kernel or getattr(self, "selected_kernel_", "rbf")

        if param_distributions is None:
            param_distributions = {
                "svr__kernel": [kernel],
                "svr__C": loguniform(1e-2, 1e3),
                "svr__epsilon": uniform(0.01, 0.99),
            }
            if kernel in ("rbf", "poly", "sigmoid"):
                param_distributions["svr__gamma"] = loguniform(1e-4, 1e1)
            if kernel == "poly":
                param_distributions["svr__degree"] = randint(2, 5)

        search = self.random_search(param_distributions, n_iter=n_iter, cv=cv,
                                     scoring=scoring, sample_frac=sample_frac)
        self.best_params_ = search.best_params_
        return search


# ====================================================================
#  XGBOOST MODEL (Gradient-Boosted Trees)
#  https://xgboost.readthedocs.io/en/stable/
# ====================================================================
class XGBoostModel(BaseModel):
    """
    Gradient-boosted trees via XGBoost.

    Why XGBoost needs a DIFFERENT tuning strategy than the others
    ---------------------------------------------------------------
    Like RandomForest, XGBoost has SEVERAL interacting hyperparameters
    (max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, gamma, reg_alpha, reg_lambda), so a single-param
    two-pass sweep isn't appropriate here either.

    UNLIKE RandomForest, though, n_estimators (the number of boosting
    rounds) is NOT "more is free / never hurts": each new tree is fit
    to correct the PREVIOUS trees' residuals (boosting), so too many
    rounds actively OVERFITS -- this is the core structural difference
    from RF's bagging, where trees are independent and averaging more
    of them can only reduce variance. Reusing RF's OOB-elbow approach
    here would be wrong, because XGBoost has no OOB score AND more
    rounds isn't monotonically safe. The standard, XGBoost-specific way
    to pick n_estimators is EARLY STOPPING: train with a large
    n_estimators cap and a held-out validation set, and let XGBoost
    stop automatically once the validation score hasn't improved for
    `early_stopping_rounds` consecutive rounds, then read back the
    actual best round from `best_iteration`.

    Strategy used here:
      1) find_n_estimators_early_stopping(): carves a validation split
         OUT OF the training data only (X_test stays untouched), fits
         with a large n_estimators cap (default 2000) and early
         stopping, and reads off the optimal round count into
         self.selected_n_estimators_.
      2) random_search_tune(): RandomizedSearchCV over the remaining
         interacting hyperparameters (max_depth, learning_rate,
         subsample, colsample_bytree, min_child_weight, gamma,
         reg_alpha, reg_lambda), with n_estimators fixed at a modest
         search-time value (mirrors RandomForestModel's
         n_estimators_search) -- exploring 100+ combinations at the
         full early-stopped round count would be slow for the same
         reason RF's search fixes a smaller n_estimators during search.
      3) train_best(): final production fit with the winning
         hyperparameters AND the full early-stopped n_estimators.

    Requires the 'xgboost' package (not part of scikit-learn):
        pip install xgboost --break-system-packages
    """

    def __init__(self, *args, **kwargs):
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoostModel requires the 'xgboost' package, which isn't "
                "installed. Install it with:\n"
                "    pip install xgboost --break-system-packages\n"
                "(or just `pip install xgboost` outside a sandboxed env)."
            )
        super().__init__(*args, **kwargs)

    def _build_estimator(self, **kwargs):
        # Same n_jobs pattern as RandomForestModel: read from
        # self._xgb_n_jobs so _avoid_nested_parallelism() below can
        # force single-threaded fits during an outer CV search.
        n_jobs = kwargs.pop("n_jobs", getattr(self, "_xgb_n_jobs", -1))
        return XGBRegressor(
            random_state=self.random_state, n_jobs=n_jobs,
            objective="reg:squarederror", **kwargs
        )

    @contextlib.contextmanager
    def _avoid_nested_parallelism(self):
        """Same nested-parallelism guard as RandomForestModel -- see
        that class's docstring for the full explanation. XGBRegressor
        also exposes a mutable n_jobs attribute, so the same in-place
        flip works for the already-fitted self.model case too."""
        self._xgb_n_jobs = 1
        had_model_n_jobs = self.model is not None and hasattr(self.model, "n_jobs")
        if had_model_n_jobs:
            prev_model_n_jobs = self.model.n_jobs
            self.model.n_jobs = 1
        try:
            yield
        finally:
            self._xgb_n_jobs = -1
            if had_model_n_jobs:
                self.model.n_jobs = prev_model_n_jobs

    def train(self, n_estimators: int = 300, max_depth: int = 6,
              learning_rate: float = 0.1, subsample: float = 1.0,
              colsample_bytree: float = 1.0, min_child_weight: float = 1,
              gamma: float = 0, reg_alpha: float = 0, reg_lambda: float = 1,
              **xgb_kwargs):
        """Trains an XGBRegressor with fixed hyperparameters."""
        self._check_split()
        self.model = self._build_estimator(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, gamma=gamma,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, **xgb_kwargs
        )
        self.pipeline = make_pipeline(self.preprocessor, self.model)
        self.pipeline.fit(self.X_train, self.y_train)
        print(f"[XGBoostModel] trained with n_estimators={n_estimators}, "
              f"max_depth={max_depth}, learning_rate={learning_rate}")
        return self

    def _current_hyperparams(self):
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "subsample": self.model.subsample,
            "colsample_bytree": self.model.colsample_bytree,
            "min_child_weight": self.model.min_child_weight,
            "gamma": self.model.gamma,
            "reg_alpha": self.model.reg_alpha,
            "reg_lambda": self.model.reg_lambda,
        }

    def _stronger_regularization(self, params):
        """
        XGBoost overfits by fitting deeper/more-confident trees to the
        residuals -> shrink max_depth, require more evidence per split
        (min_child_weight up), penalize large leaf weights harder
        (reg_alpha/reg_lambda up), and see fewer rows/columns per round
        (subsample/colsample_bytree down) so no single round can latch
        onto noise as easily. n_estimators is left untouched here --
        that's already handled by early stopping, a much more direct
        instrument for it than a blanket reduction would be.
        """
        new_params = dict(params)

        depth = params["max_depth"]
        new_params["max_depth"] = max(2, depth - 1) if depth and depth > 2 else depth
        new_params["min_child_weight"] = min(params["min_child_weight"] * 2 + 1, 50)
        new_params["reg_alpha"] = min(max(params["reg_alpha"], 0.01) * 3, 50)
        new_params["reg_lambda"] = min(max(params["reg_lambda"], 0.1) * 3, 50)
        new_params["subsample"] = max(0.5, params["subsample"] - 0.1)
        new_params["colsample_bytree"] = max(0.5, params["colsample_bytree"] - 0.1)

        return new_params

    # ---------- Step 1: n_estimators via early stopping (NOT an OOB elbow) ----------
    def find_n_estimators_early_stopping(self, n_estimators_cap: int = 2000,
                                          early_stopping_rounds: int = 50,
                                          val_size: float = 0.15,
                                          **fixed_xgb_kwargs):
        """
        Carves a validation split OUT OF THE TRAINING SET ONLY (X_test
        is never touched here, keeping it clean for the final unbiased
        evaluate()), fits with a large n_estimators cap and XGBoost's
        native early stopping, and reads back the optimal round count
        from `best_iteration`.

        early_stopping_rounds: number of consecutive rounds without
        validation-score improvement before stopping. Higher = more
        patient (less likely to stop on a temporary plateau) but slower.

        val_size: fraction of the TRAINING set (not the full dataset)
        held out for early-stopping validation.

        fixed_xgb_kwargs: other XGBoost params to hold fixed during this
        step (e.g. max_depth=6) if you already have a rough idea of them.

        Returns the selected n_estimators (int) and stores it in
        self.selected_n_estimators_.
        """
        self._check_split()
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train, test_size=val_size,
            random_state=self.random_state,
        )
        # Fit the preprocessor on the TRAIN-of-train split only, and
        # apply the same fitted transform to the validation split --
        # same no-leakage principle as everywhere else in this file.
        X_tr_transformed = self.preprocessor.fit_transform(X_tr)
        X_val_transformed = self.preprocessor.transform(X_val)

        print(f"[XGBoostModel] early stopping search starting -> cap="
              f"{n_estimators_cap} rounds, stopping after "
              f"{early_stopping_rounds} rounds without improvement...")
        t0 = time.time()

        xgb = self._build_estimator(
            n_estimators=n_estimators_cap,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            **fixed_xgb_kwargs,
        )
        xgb.fit(X_tr_transformed, y_tr,
                eval_set=[(X_val_transformed, y_val)], verbose=False)

        # best_iteration is 0-indexed (round 0 = first tree) -> +1 for
        # an actual tree COUNT to pass back into n_estimators later.
        best_n = xgb.best_iteration + 1
        print(f"[XGBoostModel] early stopping done in {time.time() - t0:.1f}s "
              f"-> best round={best_n} (cap was {n_estimators_cap})")
        self.selected_n_estimators_ = best_n
        return best_n

    # ---------- Step 2: joint tuning of the remaining, interacting params ----------
    def random_search_tune(self, n_iter: int = 40, cv: int = 5,
                            scoring: str = "r2", n_estimators_search: int = 200,
                            param_distributions: dict = None):
        """
        RandomizedSearchCV over the XGBoost hyperparameters that
        genuinely interact with each other. n_estimators is fixed (not
        searched) here on purpose -> use find_n_estimators_early_stopping()
        first, for the same reason RandomForestModel fixes n_estimators
        during its own random_search_tune(): searching it jointly here
        would waste search budget on a parameter that already has a
        dedicated, more reliable selection method (early stopping,
        which directly optimizes against a held-out validation set,
        rather than being guessed via cross-validated trial and error).

        n_estimators_search: number of rounds used for EVERY fit DURING
        this search (default 200), deliberately smaller than the full
        early-stopped value -- same reasoning as RandomForestModel's
        n_estimators_search: which max_depth / learning_rate / subsample
        / ... combination wins barely changes with the round count,
        so searching at a smaller, faster round count is a good trade.
        Call train_best() right after this to add back the full
        early-stopped n_estimators for the production fit.

        param_distributions: override the default search space if
        needed. Must use the "xgbregressor__" prefix.
        """
        if param_distributions is None:
            param_distributions = {
                "xgbregressor__n_estimators": [n_estimators_search],
                "xgbregressor__max_depth": randint(3, 10),
                "xgbregressor__learning_rate": loguniform(1e-3, 3e-1),
                "xgbregressor__subsample": uniform(0.5, 0.5),        # 0.5-1.0
                "xgbregressor__colsample_bytree": uniform(0.5, 0.5), # 0.5-1.0
                "xgbregressor__min_child_weight": randint(1, 10),
                "xgbregressor__gamma": uniform(0, 5),
                "xgbregressor__reg_alpha": loguniform(1e-3, 10),
                "xgbregressor__reg_lambda": loguniform(1e-3, 10),
            }

        search = self.random_search(param_distributions, n_iter=n_iter,
                                     cv=cv, scoring=scoring)
        self.best_params_ = search.best_params_
        return search

    # ---------- Step 3: final production fit with the full n_estimators ----------
    def train_best(self, n_estimators: int = None):
        """
        Retrains the final model using the winning hyperparameters found
        by random_search_tune() (self.best_params_), but swapping back
        in the FULL n_estimators from find_n_estimators_early_stopping()
        (or an explicit value) instead of the smaller n_estimators_search
        used during the search itself.

        n_estimators: defaults to self.selected_n_estimators_ (set by
        find_n_estimators_early_stopping()), falling back to 300 if that
        was never run.
        """
        if not hasattr(self, "best_params_"):
            raise ValueError(
                "No tuned hyperparameters yet. Call random_search_tune() first."
            )
        final_n = n_estimators or getattr(self, "selected_n_estimators_", 300)

        clean_params = {
            k.split("__", 1)[1]: v for k, v in self.best_params_.items()
            if not k.endswith("n_estimators")
        }

        print(f"[XGBoostModel] final fit -> n_estimators={final_n}, "
              f"{clean_params}")
        self.train(n_estimators=final_n, **clean_params)
        return self


# ====================================================================
#  MAIN - clean summary report
#
#  All the internal training/searching/tuning logic above is unchanged
#  and still runs exactly as before -- this block just SUPPRESSES its
#  internal print() calls (via redirect_stdout) and prints ONE clean,
#  plain-language report at the end instead: for each model, the
#  scores BEFORE and AFTER hyperparameter tuning (with the chosen
#  hyperparameters shown), and finally a single combined ranking of
#  the 5 most- and 5 least-important features, obtained by VOTING
#  across all 5 models.
# ====================================================================
import contextlib
import io

CSV_PATH = "data/Dataframe_Clean_encoded.csv"


def _quiet(func, *args, **kwargs):
    """Runs func(*args, **kwargs) with its internal print() output
    swallowed, and returns whatever it returns."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _fmt_comparison(train_stats: dict, test_stats: dict) -> str:
    """
    Side-by-side Train vs Test formatting, plus the R2 gap between them
    -- this is the actual over/underfitting signal:
      - Train R2 much higher than Test R2 (large positive gap) ->
        the model memorized the training data -> OVERFITTING.
      - Train R2 close to Test R2 (small gap) -> the model generalizes
        well to unseen data.
      - Both R2 low (regardless of gap) -> UNDERFITTING (the model is
        too simple / not capturing the signal at all, on either set).
    """
    gap = train_stats["R2"] - test_stats["R2"]
    lines = [
        f"    {'':<16}{'Train':>10}{'Test':>10}",
        f"    {'R2':<16}{train_stats['R2']:>10.3f}{test_stats['R2']:>10.3f}"
        f"   (gap: {gap:+.3f})",
        f"    {'RMSE (EUR)':<16}{train_stats['RMSE']:>10,.0f}{test_stats['RMSE']:>10,.0f}",
        f"    {'MAE (EUR)':<16}{train_stats['MAE']:>10,.0f}{test_stats['MAE']:>10,.0f}",
    ]
    if train_stats["R2"] < 0.4 and test_stats["R2"] < 0.4:
        lines.append("    -> UNDERFITTING: both Train and Test scores are "
                      "low, the model isn't capturing the signal well.")
    elif gap > 0.15:
        lines.append("    -> OVERFITTING WARNING: Train score is much "
                      "higher than Test score (gap > 0.15) -- the model "
                      "is memorizing the training data rather than "
                      "generalizing.")
    elif gap > 0.07:
        lines.append("    -> Mild overfitting: some gap between Train and "
                      "Test, worth keeping an eye on.")
    else:
        lines.append("    -> Good fit: Train and Test scores are close, "
                      "the model generalizes well.")
    return "\n".join(lines)


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _print_model_report(model_name: str,
                         baseline_train: dict, baseline_test: dict,
                         tuned_train: dict, tuned_test: dict,
                         tuned_params: dict, guard_attempts: int = 0):
    print(f"\n{'=' * 60}")
    print(f" MODEL: {model_name}")
    print(f"{'=' * 60}")
    print("  BEFORE hyperparameter tuning (default values):")
    print(_fmt_comparison(baseline_train, baseline_test))
    print(f"\n  AFTER hyperparameter tuning "
          f"({_fmt_params(tuned_params)}):")
    print(_fmt_comparison(tuned_train, tuned_test))
    if guard_attempts > 0:
        print(f"    (regularization auto-increased {guard_attempts} "
              f"time(s) to control overfitting)")


def _collect_votes(top_bottom_pairs):
    """
    top_bottom_pairs: list of (top_df, bottom_df) tuples, one per model
    that supports feature ranking (skip models where it's None, e.g.
    SVM with a non-linear kernel).

    Voting: within each model's own top-5 / bottom-5, rank 1 gets 5
    points, rank 2 gets 4, ... rank 5 gets 1 point. Points are summed
    across all models per feature group -> a simple, transparent
    "Borda count" style vote. Returns (top5, bottom5), each a list of
    (feature_group_name, total_points) tuples.
    """
    points_per_rank = [5, 4, 3, 2, 1]
    most_important_votes, least_important_votes = {}, {}

    for top_df, bottom_df in top_bottom_pairs:
        if top_df is None:
            continue
        for i, group in enumerate(top_df["group"].tolist()[:5]):
            most_important_votes[group] = (
                most_important_votes.get(group, 0) + points_per_rank[i])
        for i, group in enumerate(bottom_df["group"].tolist()[:5]):
            least_important_votes[group] = (
                least_important_votes.get(group, 0) + points_per_rank[i])

    top5 = sorted(most_important_votes.items(), key=lambda x: -x[1])[:5]
    bottom5 = sorted(least_important_votes.items(), key=lambda x: -x[1])[:5]
    return top5, bottom5


if __name__ == "__main__":

    feature_rankings = []  # list of (top_df, bottom_df) per model

    # ---------------- RIDGE ----------------
    ridge = RidgeModel(CSV_PATH)
    _quiet(ridge.load_data)
    _quiet(ridge.split_data)
    _quiet(ridge.train, alpha=1.0)
    ridge_baseline_train = _quiet(ridge.evaluate, on="train")
    ridge_baseline_test = _quiet(ridge.evaluate, on="test")
    _quiet(ridge.two_pass_tune, param_name="alpha",
           coarse_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
    (ridge_params, ridge_tuned_train, ridge_tuned_test,
     ridge_attempts) = _quiet(ridge.reduce_overfitting)
    _print_model_report("Ridge Regression", ridge_baseline_train, ridge_baseline_test,
                         ridge_tuned_train, ridge_tuned_test, ridge_params,
                         ridge_attempts)
    top, bottom = _quiet(ridge.grouped_feature_importance, top_n=5)
    feature_rankings.append((top, bottom))

    # ---------------- DECISION TREE ----------------
    tree = DecisionTreeModel(CSV_PATH)
    _quiet(tree.load_data)
    _quiet(tree.split_data)
    _quiet(tree.train, max_depth=None)
    tree_baseline_train = _quiet(tree.evaluate, on="train")
    tree_baseline_test = _quiet(tree.evaluate, on="test")
    _quiet(tree.two_pass_tune, param_name="max_depth",
           coarse_values=[2, 4, 6, 8, 10, 15, 20, 25, 30])
    (tree_params, tree_tuned_train, tree_tuned_test,
     tree_attempts) = _quiet(tree.reduce_overfitting)
    _print_model_report("Decision Tree", tree_baseline_train, tree_baseline_test,
                         tree_tuned_train, tree_tuned_test, tree_params,
                         tree_attempts)
    top, bottom = _quiet(tree.grouped_feature_importance, top_n=5)
    feature_rankings.append((top, bottom))

    # ---------------- RANDOM FOREST ----------------
    rf = RandomForestModel(CSV_PATH)
    _quiet(rf.load_data)
    _quiet(rf.split_data)
    _quiet(rf.train, n_estimators=300, max_depth=None)
    rf_baseline_train = _quiet(rf.evaluate, on="train")
    rf_baseline_test = _quiet(rf.evaluate, on="test")
    best_n = _quiet(rf.find_n_estimators_elbow,
                     n_values=(50, 100, 200, 300, 500, 800, 1200))
    _quiet(rf.random_search_tune, n_iter=40, cv=5, n_estimators_search=200)
    _quiet(rf.train_best, n_estimators=best_n)
    (rf_params, rf_tuned_train, rf_tuned_test,
     rf_attempts) = _quiet(rf.reduce_overfitting)
    _print_model_report("Random Forest", rf_baseline_train, rf_baseline_test,
                         rf_tuned_train, rf_tuned_test, rf_params, rf_attempts)
    top, bottom = _quiet(rf.grouped_feature_importance, top_n=5)
    feature_rankings.append((top, bottom))

    # ---------------- SVM ----------------
    svm = SVMModel(CSV_PATH)
    _quiet(svm.load_data)
    _quiet(svm.split_data)
    _quiet(svm.train, kernel="rbf", C=1.0, epsilon=0.1)
    svm_baseline_train = _quiet(svm.evaluate, on="train")
    svm_baseline_test = _quiet(svm.evaluate, on="test")
    best_kernel = _quiet(svm.quick_kernel_search, sample_size=2000)
    _quiet(svm.random_search_tune, n_iter=30, cv=5,
           kernel=best_kernel, sample_frac=0.3)
    (svm_params, svm_tuned_train, svm_tuned_test,
     svm_attempts) = _quiet(svm.reduce_overfitting)
    _print_model_report("SVM (Support Vector Regression)", svm_baseline_train,
                         svm_baseline_test, svm_tuned_train, svm_tuned_test,
                         svm_params, svm_attempts)
    if svm.model.kernel == "linear":
        top, bottom = _quiet(svm.grouped_feature_importance, top_n=5)
        feature_rankings.append((top, bottom))
    else:
        print("  (Feature ranking not available for this kernel "
              f"'{svm.model.kernel}' -- only the 'linear' kernel supports it)")

    # ---------------- XGBOOST ----------------
    xgb_model = XGBoostModel(CSV_PATH)
    _quiet(xgb_model.load_data)
    _quiet(xgb_model.split_data)
    _quiet(xgb_model.train, n_estimators=300, max_depth=6, learning_rate=0.1)
    xgb_baseline_train = _quiet(xgb_model.evaluate, on="train")
    xgb_baseline_test = _quiet(xgb_model.evaluate, on="test")
    best_n_xgb = _quiet(xgb_model.find_n_estimators_early_stopping,
                         n_estimators_cap=2000, early_stopping_rounds=50)
    _quiet(xgb_model.random_search_tune, n_iter=40, cv=5, n_estimators_search=200)
    _quiet(xgb_model.train_best, n_estimators=best_n_xgb)
    (xgb_params, xgb_tuned_train, xgb_tuned_test,
     xgb_attempts) = _quiet(xgb_model.reduce_overfitting)
    _print_model_report("XGBoost", xgb_baseline_train, xgb_baseline_test,
                         xgb_tuned_train, xgb_tuned_test, xgb_params, xgb_attempts)
    top, bottom = _quiet(xgb_model.grouped_feature_importance, top_n=5)
    feature_rankings.append((top, bottom))

    # ---------------- COMBINED FEATURE VOTE ----------------
    top5_voted, bottom5_voted = _collect_votes(feature_rankings)

    print(f"\n{'=' * 60}")
    print(" FINAL FEATURE RANKING (voted across all models)")
    print(f"{'=' * 60}")
    print("  Top 5 features that influence the price THE MOST:")
    for rank, (name, points) in enumerate(top5_voted, start=1):
        print(f"    {rank}. {name}  ({points} points)")

    print("\n  Top 5 features that influence the price THE LEAST:")
    for rank, (name, points) in enumerate(bottom5_voted, start=1):
        print(f"    {rank}. {name}  ({points} points)")