"""
Predict_Model.py

Trains EVERY regression model defined in Train_Model.py (Ridge,
DecisionTree, RandomForest, SVM, XGBoost) and reports each model's
OWN prediction for a given listing -- side by side.

No blending. No stacking. No weighted average. No auto-selected
"best" method. You get the five individual prices and their own
measured accuracy (R2 / RMSE / MAE), and you decide.

Each model is still trained and tuned using its OWN existing tuning
routine from Train_Model.py (two_pass_tune for Ridge/DecisionTree,
elbow + random search for RandomForest, kernel search + random search
for SVM, early stopping + random search for XGBoost) -- nothing about
that logic is duplicated here, it is reused as-is.

Runtime: tuning="quick" (the default) re-tunes all 5 models with lighter
search budgets; expect roughly 5-15 minutes on the full ~11k-row
dataset depending on your machine. tuning="full" mirrors the exact
search budgets used in Train_Model.py's own __main__ block (best
quality, noticeably slower -- can take well over an hour).

Usage
-----
    # Train everything and see the per-model comparison report:
    python Predict_Model.py

    # Reuse an already-trained set of models to price a new listing:
    from Predict_Model import PricePredictorEnsemble
    ens = PricePredictorEnsemble.load("models/price_ensemble.joblib")
    prices = ens.predict_price({
        "living_area_m2": 120, "bedrooms": 3, "bathrooms": 1,
        "latitude": 50.83, "longitude": 4.35, "building_year": 1975,
        "property_type": "Apartment", "property_subtype": "apartment",
        "province": "Brussels Capital Region", "region": "Brussels",
        "kitchen_equipped": "Fully equipped", "state_of_the_building": "Normal",
        "epc_score": "C", "nearby_city": "Bruxelles",
        "furnished": 0, "has_garage": 1, "parking_count": 1,
        "has_elevator": 1, "facades": 2, "has_garden": 0,
        "garden_area_m2": 0, "has_terrace": 1, "total_area_m2": 120,
        "km_from_nearby_city": 2.0, "is_nearby_city_prestigious": 0,
        "floor_number": 3,
    })
    # prices -> {"ridge": 312450.0, "decision_tree": 298000.0,
    #            "random_forest": 305120.0, "svm": 289990.0,
    #            "xgboost": 310875.0}

    # price is right-skewed -> every model fits on log1p(price) by
    # default (log_target=True); predictions are converted back to real
    # EUR automatically. Pass log_target=False to revert to raw price:
    ens = PricePredictorEnsemble(CSV_PATH, log_target=False).fit()
"""

import time
import contextlib
import io
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from train import (
    RidgeModel, DecisionTreeModel, RandomForestModel, SVMModel, XGBoostModel,
)
from data_preprocessing import PreprocessData


# ====================================================================
#  Encoding config for brand-new (raw, unencoded) listings.
#  Must exactly match the ORDINAL_SPECS / ONEHOT_COLS / BINARY_COLS used
#  in data_preprocessing.py's __main__ block, so a new listing gets
#  encoded into the exact same columns the models were trained on.
# ====================================================================
ORDINAL_SPECS = {
    "kitchen_equipped": {
        "new": "kitchen_equipped_encoded",
        "order": {"Not equipped": 0, "Partially equipped": 1,
                  "Fully equipped": 2, "Super equipped": 3},
    },
    "state_of_the_building": {
        "new": "state_encoded",
        "order": {"To renovate": 0, "Normal": 1, "Fully renovated": 2,
                  "New": 3, "Excellent": 4},
    },
    "epc_score": {
        "new": "epc_encoded",
        "order": {"G": 0, "F": 1, "E": 2, "E+": 3, "D": 4, "C": 5,
                  "B": 6, "B+": 7, "A": 8, "A+": 9, "A++": 10},
    },
}
ONEHOT_COLS = ["property_subtype", "province", "region"]
BINARY_COLS = ["is_nearby_city_prestigious", "has_garage", "has_garden",
               "has_terrace", "furnished", "has_elevator"]


# ====================================================================
#  MULTI-MODEL PREDICTOR (no blending)
# ====================================================================
class PricePredictorEnsemble:
    """
    Trains all 5 models from Train_Model.py and reports each one's own
    prediction for a listing -- nothing is combined or averaged.
    """

    MODEL_CLASSES = {
        "ridge": RidgeModel,
        "decision_tree": DecisionTreeModel,
        "random_forest": RandomForestModel,
        "svm": SVMModel,
        "xgboost": XGBoostModel,
    }

    def __init__(self, csv_path: str, target: str = "price",
                 test_size: float = 0.2, random_state: int = 42,
                 tuning: str = "quick", verbose: bool = False,
                 log_target: bool = False):
        """
        tuning: "quick" (fast, lighter search budgets -- good for
        iterating) or "full" (identical search budgets to
        train.py's own __main__ block -- best quality, slower).

        verbose: False (default) -> every intermediate print() from the
        training/tuning process (and from the raw-listing encoder) is
        swallowed; only the final PREDICTION REPORT from predict_price()
        is shown. Set True to see the full training log + the
        FINAL COMPARISON table again, e.g. while debugging.

        log_target: True by default -- passed straight through to every
        model (see BaseModel.__init__ in train.py). All 5 models fit on
        log1p(price) instead of raw price -- useful since price is
        right-skewed. Predictions are converted back to real EUR
        automatically everywhere (predict(), predict_price(), reports).
        Set False to revert to fitting on raw price.
        """
        if tuning not in ("quick", "full"):
            raise ValueError("tuning must be 'quick' or 'full'")

        self.csv_path = csv_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.tuning = tuning
        self.verbose = verbose
        self.log_target = log_target

        self.trained_models_ = {}   # name -> fitted BaseModel subclass instance
        self.scores_ = {}           # name -> {"R2":.., "RMSE":.., "MAE":..}
        self.results_ = {}

        self.X_train_ = self.y_train_ = None
        self.X_test_ = self.y_test_ = None
        self.feature_columns_ = None

    def _maybe_quiet(self):
        """Context manager: swallows stdout unless self.verbose is True."""
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    # ---------------------------------------------------------------
    #  Training every base model with ITS OWN tuning recipe
    # ---------------------------------------------------------------
    def _tune(self, name: str, model):
        q = self.tuning == "quick"

        if name == "ridge":
            coarse = [0.01, 0.1, 1, 10, 100] if q else \
                     [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            model.two_pass_tune(param_name="alpha", coarse_values=coarse,
                                 cv=3 if q else 5)
            model.reduce_overfitting()

        elif name == "decision_tree":
            coarse = [4, 8, 12, 16, 20] if q else \
                     [2, 4, 6, 8, 10, 15, 20, 25, 30]
            model.two_pass_tune(param_name="max_depth", coarse_values=coarse,
                                 cv=3 if q else 5)
            model.reduce_overfitting()

        elif name == "random_forest":
            n_values = (100, 300, 500) if q else (50, 100, 200, 300, 500, 800, 1200)
            best_n = model.find_n_estimators_elbow(
                n_values=n_values, patience=1 if q else 2)
            model.random_search_tune(
                n_iter=12 if q else 40, cv=3 if q else 5,
                n_estimators_search=150 if q else 200)
            model.train_best(n_estimators=best_n)
            model.reduce_overfitting()

        elif name == "svm":
            best_kernel = model.quick_kernel_search(
                sample_size=1200 if q else 2000, cv=3)
            model.random_search_tune(
                n_iter=12 if q else 30, cv=3 if q else 5,
                kernel=best_kernel, sample_frac=0.2 if q else 0.3)
            model.reduce_overfitting()

        elif name == "xgboost":
            best_n = model.find_n_estimators_early_stopping(
                n_estimators_cap=800 if q else 2000,
                early_stopping_rounds=30 if q else 50)
            model.random_search_tune(
                n_iter=12 if q else 40, cv=3 if q else 5,
                n_estimators_search=150 if q else 200)
            model.train_best(n_estimators=best_n)
            model.reduce_overfitting()

    def fit(self):
        with self._maybe_quiet():
            for name, cls in self.MODEL_CLASSES.items():
                print(f"\n{'=' * 70}\n TRAINING: {name}  (tuning={self.tuning})\n{'=' * 70}")
                t0 = time.time()

                model = cls(self.csv_path, target=self.target,
                            test_size=self.test_size, random_state=self.random_state,
                            log_target=self.log_target)
                model.load_data().split_data()
                self._tune(name, model)

                test_scores = model.evaluate(on="test")
                self.trained_models_[name] = model
                self.scores_[name] = test_scores
                print(f"[Predictor] {name} done in {time.time() - t0:.1f}s")

                if self.X_train_ is None:
                    self.X_train_, self.y_train_ = model.X_train, model.y_train
                    self.X_test_, self.y_test_ = model.X_test, model.y_test
                    self.feature_columns_ = list(model.X_train.columns)
                else:
                    # Every model reads the same CSV/target/test_size/random_state,
                    # so they MUST all land on the exact same split. Verified here
                    # rather than assumed, in case anything downstream ever
                    # compares predictions row-for-row across models.
                    assert list(model.X_train.index) == list(self.X_train_.index), (
                        f"{name}'s train/test split differs from the others -- "
                        "check csv_path/target/test_size/random_state."
                    )

            self._report()
        return self

    # ---------------------------------------------------------------
    #  Prediction -- one array per model, nothing merged
    # ---------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> dict:
        """
        Predict price for already-encoded rows (same columns as the
        training CSV, minus 'price').

        Returns a dict {model_name: np.ndarray of predictions} -- one
        full array per model, never combined.
        """
        if not self.trained_models_:
            raise ValueError("Not fitted yet. Call .fit() first.")
        preds = {name: model.pipeline.predict(X)
                 for name, model in self.trained_models_.items()}
        if self.log_target:
            preds = {name: np.expm1(p) for name, p in preds.items()}
        return preds

    def _encode_new_listing(self, raw: dict) -> pd.DataFrame:
        """
        Turns ONE raw (unencoded) listing dict into the exact encoded
        column format the models were trained on. Only handles ENCODING
        (ordinal / one-hot / label / nearby-city price) -- the dataset-
        level cleaning steps in data_preprocessing.py (dropping bad
        rows, group-median imputation...) don't apply to a single
        already-known listing; missing numeric values are handled by
        each pipeline's own median imputer.
        """
        row = pd.DataFrame([raw])
        pre = PreprocessData(row)
        pre.preprocess_data(
            ordinal_specs=ORDINAL_SPECS, onehot_cols=ONEHOT_COLS,
            binary_cols=BINARY_COLS, onehot_drop_first=False,
        )
        encoded = pre.get_data()
        # Add any one-hot column absent from this single row (0), drop
        # anything unexpected, and enforce the training column order.
        return encoded.reindex(columns=self.feature_columns_, fill_value=0)

    def predict_price(self, raw_listing: dict, show_report: bool = True) -> dict:
        """
        Convenience wrapper: raw listing dict -> dict of
        {model_name: predicted_price}, one price per model.

        show_report=True (default) prints each model's prediction next
        to its own measured accuracy (R2 / RMSE / MAE) on the held-out
        test set, so you can see how much the models agree/disagree
        and judge which one(s) to trust.
        """
        with self._maybe_quiet():
            encoded = self._encode_new_listing(raw_listing)

        prices = {}
        for name, model in self.trained_models_.items():
            raw_pred = model.pipeline.predict(encoded)[0]
            prices[name] = float(np.expm1(raw_pred) if self.log_target else raw_pred)

        if show_report:
            self._print_prediction_report(prices)

        return prices

    def _print_prediction_report(self, prices: dict):
        if not self.scores_:
            raise ValueError("Not fitted yet. Call .fit() first.")

        print(f"\n{'=' * 70}\n PREDICTION REPORT (per model -- no blending)\n{'=' * 70}")
        for name in sorted(prices, key=lambda n: -self.scores_[n]["R2"]):
            s = self.scores_[name]
            print(f"  {name:15s} EUR {prices[name]:>12,.0f}   "
                  f"(test R2={s['R2']:.4f}  RMSE=EUR {s['RMSE']:>10,.0f}  "
                  f"MAE=EUR {s['MAE']:>10,.0f})")
        print(f"{'=' * 70}")

    # ---------------------------------------------------------------
    #  Reporting
    # ---------------------------------------------------------------
    @staticmethod
    def _scores(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"R2": r2, "RMSE": rmse, "MAE": mae}

    def _report(self):
        print(f"\n{'=' * 70}\n FINAL COMPARISON (test set)\n{'=' * 70}")
        for name, s in sorted(self.scores_.items(), key=lambda kv: -kv[1]["R2"]):
            print(f"  {name:15s} R2={s['R2']:.4f}  RMSE={s['RMSE']:>10,.0f}  "
                  f"MAE={s['MAE']:>10,.0f}")
        print(f"{'=' * 70}")

        self.results_ = {"per_model": self.scores_}

    # ---------------------------------------------------------------
    #  Persistence
    # ---------------------------------------------------------------
    def save(self, path: str = "models/price_ensemble.joblib"):
        joblib.dump(self, path)
        if self.verbose:
            print(f"[Predictor] saved to '{path}'")
        return self

    @staticmethod
    def load(path: str = "models/price_ensemble.joblib") -> "PricePredictorEnsemble":
        return joblib.load(path)


# ====================================================================
#  MAIN
# ====================================================================
if __name__ == "__main__":
    CSV_PATH = "data/Dataframe_Clean_encoded.csv"

    ensemble = PricePredictorEnsemble(
        CSV_PATH, tuning="full", verbose=False
    )
    ensemble.fit()
    ensemble.save("models/price_ensemble.joblib")

    example_listing = {
        "living_area_m2": 120, "bedrooms": 3, "bathrooms": 1,
        "latitude": 50.83, "longitude": 4.35, "building_year": 1975,
        "property_type": "Apartment", "property_subtype": "apartment",
        "province": "Brussels Capital Region", "region": "Brussels",
        "kitchen_equipped": "Fully equipped", "state_of_the_building": "Normal",
        "epc_score": "C", "nearby_city": "Bruxelles",
        "furnished": 0, "has_garage": 1, "parking_count": 1,
        "has_elevator": 1, "facades": 2, "has_garden": 0,
        "garden_area_m2": 0, "has_terrace": 1, "total_area_m2": 120,
        "km_from_nearby_city": 2.0, "is_nearby_city_prestigious": 0,
        "floor_number": 3,
    }
    prices = ensemble.predict_price(example_listing)