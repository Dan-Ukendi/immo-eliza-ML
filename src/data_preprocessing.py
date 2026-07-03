"""
data_preprocessing.py

Two-phase structure:
  1. DataCleaning   -> duplicates, missing values, dropping columns/rows
  2. PreprocessData -> imputing, encoding, rescaling (currently: encoding only)

Usage (see __main__ at the bottom):
    cleaner = DataCleaning("data/properties_final.csv")
    cleaner.load().clean_column_names().clean_data(...)
    df_clean = cleaner.get_data()

    pre = PreprocessData(df_clean)
    pre.preprocess_data(...)
    pre.save("data/Dataframe_Clean_encoded.csv")
"""

import pandas as pd
import numpy as np


# ====================================================================
#  PHASE 1 — DATA CLEANING
#  Handling duplicates, missing values, dropping columns or rows.
# ====================================================================
class DataCleaning:
    def __init__(self, filepath: str = None, df: pd.DataFrame = None):
        """Start from a CSV path or an existing dataframe."""
        self.filepath = filepath
        self.df = df.copy() if df is not None else None
        self._raw = self.df.copy() if df is not None else None

    # ---------- Loading ----------
    def load(self, **read_csv_kwargs):
        """Load the CSV into a dataframe."""
        self.df = pd.read_csv(self.filepath, **read_csv_kwargs)
        self._raw = self.df.copy()
        print(f"Loaded '{self.filepath}' -> shape {self.df.shape}")
        return self

    def reset(self):
        """Revert to the originally loaded data."""
        self.df = self._raw.copy()
        print("Reset to raw data.")
        return self

    # ---------- Inspection ----------
    def overview(self):
        """Print a quick diagnostic snapshot of the current dataframe."""
        df = self._check_loaded()
        print("=" * 50)
        print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"Duplicated rows: {df.duplicated().sum()}")
        print("-" * 50)
        print("Dtypes:")
        print(df.dtypes)
        print("-" * 50)
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            print("Missing values : none")
        else:
            pct = (missing / len(df) * 100).round(1)
            print("Missing values (count | %):")
            print(pd.concat([missing, pct], axis=1, keys=["count", "%"]))
        print("=" * 50)
        return self

    def preview(self, n: int = 5):
        print(self._check_loaded().head(n))
        return self

    # ---------- Shared helpers ----------
    def clean_column_names(self):
        """Lowercase, strip, and replace spaces with underscores."""
        df = self._check_loaded()
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_", regex=False))
        return self

    def drop_columns(self, columns):
        """Drop a list of columns (ignores ones that don't exist)."""
        df = self._check_loaded()
        self.df = df.drop(columns=columns, errors="ignore")
        return self

    def get_data(self):
        """Return the current dataframe."""
        return self._check_loaded()

    def save(self, path: str, index: bool = False):
        self._check_loaded().to_csv(path, index=index)
        print(f"Saved data to '{path}'.")
        return self

    def _check_loaded(self):
        if self.df is None:
            raise ValueError("No data loaded yet. Call .load() first.")
        return self.df

    # ---------- Cleaning steps ----------
    def drop_duplicates(self, subset=None):
        df = self._check_loaded()
        before = len(df)
        self.df = df.drop_duplicates(subset=subset)
        print(f"Dropped {before - len(self.df)} duplicate rows.")
        return self

    def zero_areas(self, pairs):
        """For each (flag_col, area_col), set area to 0 where flag == 0."""
        df = self._check_loaded()
        for flag_col, area_col in pairs:
            if flag_col in df.columns and area_col in df.columns:
                df.loc[df[flag_col] == 0, area_col] = 0
                print(f"[zero] '{area_col}' = 0 where '{flag_col}' == 0")
            else:
                print(f"[zero] skip ({flag_col}, {area_col}) (missing column)")
        self.df = df
        return self

    def set_house_floor_zero(self, type_col="property_type",
                             floor_col="floor_number", house_label="House"):
        """Set floor_number to 0 for houses; leave apartments untouched."""
        df = self._check_loaded()
        df.loc[df[type_col] == house_label, floor_col] = 0
        remaining = df[floor_col].isna().sum()
        print(f"Set '{floor_col}' to 0 for houses. "
              f"{remaining} NaNs remain (apartments).")
        return self

    def fill_apartment_total_area(self, type_col="property_type",
                                  total_col="total_area_m2",
                                  living_col="living_area_m2",
                                  apartment_label="Apartment"):
        """For apartments, set total area = living area (no plot)."""
        df = self._check_loaded()
        mask = df[type_col] == apartment_label
        print(f"Mask matched {mask.sum()} rows for {type_col} == "
              f"'{apartment_label}'")
        df.loc[mask, total_col] = df.loc[mask, living_col].values
        remaining = df[total_col].isna().sum()
        print(f"Set '{total_col}' = '{living_col}' for apartments. "
              f"{remaining} NaNs remain.")
        return self

    def fill_building_year_by_state(self, year_col="building_year",
                                    state_col="state_of_the_building"):
        """Fill missing building_year with its building-state group median."""
        df = self._check_loaded()
        df[year_col] = df.groupby(state_col)[year_col].transform(
            lambda s: s.fillna(s.median()))
        print(f"Filled '{year_col}' by '{state_col}' group median. "
              f"NaNs remaining: {df[year_col].isna().sum()}")
        return self

    def drop_non_existing_buildings(self, column="state_of_the_building",
                                    states=("Under construction",
                                            "To demolish", "To restore")):
        """Remove non-standing stock (under construction / demolish / restore)."""
        df = self._check_loaded()
        n = len(df)
        mask = df[column].isin(states)
        self.df = df[~mask].copy()
        print(f"Dropped {mask.sum()} rows ({', '.join(states)}). "
              f"Remaining: {len(self.df)} / {n}")
        return self

    def drop_rows_low_missing(self, threshold=0.05,
                              always=("price", "living_area_m2")):
        """Drop rows missing values in critical (low-missingness) columns."""
        df = self._check_loaded()
        n = len(df)
        miss = df.isna().mean()
        critical = set(miss[miss < threshold].index) | set(always)
        critical = [c for c in critical if c in df.columns]
        mask = df[critical].isna().any(axis=1)
        self.df = df[~mask].copy()
        print(f"Dropped {mask.sum()} rows missing critical values "
              f"(< {threshold:.0%} cols + {always}). "
              f"Remaining: {len(self.df)} / {n}")
        return self

    def drop_bad_distances(self, column="km_from_nearby_city", max_km=300):
        """Drop rows with an implausible distance to the nearest city."""
        df = self._check_loaded()
        n = len(df)
        mask = df[column] > max_km
        self.df = df[~mask].copy()
        print(f"Dropped {mask.sum()} rows with {column} > {max_km} km. "
              f"Remaining: {len(self.df)} / {n}")
        return self

    # ---------- Orchestrator ----------
    def clean_data(self, zero_area_pairs, drop_states, drop_cols,
                   missing_threshold=0.05,
                   always_keep=("price", "living_area_m2"), max_km=300):
        """
        Run all cleaning steps: duplicate removal, structural missing-value
        fills, row drops, and column drops. Categorical columns stay as text
        for the PreprocessData phase to encode.
        """
        self.drop_duplicates()
        self.zero_areas(zero_area_pairs)
        self.set_house_floor_zero("property_type", "floor_number", "House")
        self.fill_apartment_total_area("property_type", "total_area_m2",
                                       "living_area_m2", "Apartment")
        self.fill_building_year_by_state("building_year",
                                         "state_of_the_building")
        self.drop_non_existing_buildings("state_of_the_building", drop_states)
        self.drop_rows_low_missing(missing_threshold, always_keep)
        self.drop_bad_distances("km_from_nearby_city", max_km)
        self.drop_columns(drop_cols)
        print(">>> clean_data complete.")
        return self


# ====================================================================
#  PHASE 2 — PREPROCESS DATA
#  Imputing missing values, encoding, rescaling.
#  (currently: encoding only — imputation/scaling go in the ML pipeline)
# ====================================================================
class PreprocessData:
    def __init__(self, df: pd.DataFrame):
        """Initialize with the cleaned dataframe from DataCleaning."""
        self.df = df.copy()

    # ---------- Shared helpers ----------
    def get_data(self):
        return self._check_loaded()

    def save(self, path: str, index: bool = False):
        self._check_loaded().to_csv(path, index=index)
        print(f"Saved data to '{path}'.")
        return self

    def overview(self):
        df = self._check_loaded()
        print("=" * 50)
        print(f"Shape          : {df.shape[0]} rows x {df.shape[1]} cols")
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            print("Missing values : none")
        else:
            pct = (miss / len(df) * 100).round(1)
            print("Missing values (count | %):")
            print(pd.concat([miss, pct], axis=1, keys=["count", "%"]))
        print("=" * 50)
        return self

    def _check_loaded(self):
        if self.df is None:
            raise ValueError("No dataframe provided.")
        return self.df

    # ---------- Encoding steps ----------
    def ordinal_encode(self, specs, drop_original=True, fill_missing=-1):
        """
        Ordinal-encode columns from a config dict.
        specs: source col -> {"order": {...}, "new": "out_name"}.
        Missing/unmapped values become `fill_missing` (default -1).
        """
        df = self._check_loaded()
        for col, spec in specs.items():
            if col not in df.columns:
                print(f"[ordinal] skip '{col}' (not found)")
                continue
            new_name = spec["new"]
            df[new_name] = (df[col].map(spec["order"])
                            .fillna(fill_missing).astype(int))
            if drop_original:
                df = df.drop(columns=[col])
            print(f"[ordinal] '{col}' -> '{new_name}'")
        self.df = df
        return self

    def encode_property_type(self, column="property_type", drop_original=True):
        """Label-encode property_type: Apartment -> 0, House -> 1."""
        df = self._check_loaded()
        mapping = {"Apartment": 0, "House": 1}
        df["property_type_encoded"] = df[column].map(mapping)
        if drop_original:
            df = df.drop(columns=[column])
        self.df = df
        print(f"'{column}' -> property_type_encoded (Apartment=0, House=1)")
        return self

    def onehot(self, columns, drop_original=True, drop_first=False):
        """One-hot encode each column in `columns`."""
        df = self._check_loaded()
        for col in columns:
            if col not in df.columns:
                print(f"[onehot] skip '{col}' (not found)")
                continue
            dummies = pd.get_dummies(df[col], prefix=col,
                                     drop_first=drop_first, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df = df.drop(columns=[col])
            print(f"[onehot] '{col}' -> {dummies.shape[1]} columns")
        self.df = df
        return self

    def encode_nearby_city_price_m2(self, column="nearby_city",
                                    fallback=2500, drop_original=True):
        """
        Replace nearby_city with an external average asking price per m2 (EUR).
        Source: Immoweb per-city/arrondissement asking-price index (~2025),
        plus national portals for foreign cities. External figure -> no leakage.
        Unmapped cities fall back to `fallback`. Creates 'nearby_city_price_m2'.
        """
        df = self._check_loaded()
        price_per_m2 = {
            # --- Brussels / Brabant ---
            "Bruxelles": 3350, "Uccle": 3900, "Waterloo": 3200,
            "Tervuren": 3550, "Lasne": 3500, "Rhode-Saint-Genèse": 3500,
            "Overijse": 3200, "Louvain": 3100,
            # --- Wallonia ---
            "Liège": 1950, "Namur": 2400, "Charleroi": 1450, "Verviers": 1900,
            "Tournai": 1900, "Mouscron": 1750, "Mons": 1750,
            "La Louvière": 1600, "Arlon": 2050,
            # --- Flanders ---
            "Gent": 2900, "Malines": 2650, "Ostende": 2400,
            "Knokke-Heist": 5000, "Bruges": 2750, "De Panne": 2850,
            "Antwerpen": 2850, "Hasselt": 2450, "Kortrijk": 2400,
            "Waregem": 2400, "Alost": 2500, "Sint-Niklaas": 2300,
            "Turnhout": 2250, "Geel": 2250, "Dendermonde": 2150,
            "Roeselare": 2100, "Lokeren": 2050, "Genk": 2050,
            "Sint-Truiden": 2000,
            # --- Foreign (researched, ~2025 city sources) ---
            "Breda": 4600, "Eindhoven": 4500, "Maastricht": 4000,
            "Aachen": 3400, "Lille": 3400, "Reims": 2570, "Dunkerque": 2100,
        }
        df["nearby_city_price_m2"] = df[column].map(price_per_m2).fillna(fallback)
        if drop_original:
            df = df.drop(columns=[column])
        self.df = df
        print(f"'{column}' -> nearby_city_price_m2 (external €/m²)")
        return self

    def fix_binary_dtypes(self, columns):
        """Cast boolean-like columns to int once their NaNs are gone."""
        df = self._check_loaded()
        for col in columns:
            if col in df.columns and df[col].isna().sum() == 0:
                df[col] = df[col].astype(int)
        print(f"Cast to int (where NaN-free): {columns}")
        return self

    # ---------- Orchestrator ----------
    def preprocess_data(self, ordinal_specs, onehot_cols, binary_cols,
                        onehot_drop_first=False):
        """
        Run all preprocessing steps. Currently encoding only.
        Imputation and rescaling belong in the scikit-learn pipeline
        (to avoid train/test leakage) and are intentionally omitted here.
        """
        self.ordinal_encode(ordinal_specs)
        self.encode_property_type("property_type", True)
        self.onehot(onehot_cols, drop_original=True,
                    drop_first=onehot_drop_first)
        self.encode_nearby_city_price_m2("nearby_city")
        self.fix_binary_dtypes(binary_cols)
        print(">>> preprocess_data complete.")
        return self


# ====================================================================
#  MAIN  — clean first, then preprocess
# ====================================================================
if __name__ == "__main__":

    # ---- CONFIG ----
    # Columns to drop. Do NOT list columns that get encoded (their encoders
    # drop the original themselves): property_subtype, province, region,
    # nearby_city, property_type, kitchen_equipped, state_of_the_building, epc_score.
    DROP_COLS = [
        "floors_total", "city", "address", "property_url",
        "coord_swapped", "postal_code",
        "price_per_m2",   # LEAKAGE: derived from target 'price'
    ]

    ONEHOT_COLS = ["property_subtype", "province", "region"]

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

    ZERO_AREA_PAIRS = [
        ("has_garden", "garden_area_m2"),
        ("has_garage", "parking_count"),
    ]

    DROP_STATES = ("Under construction", "To demolish", "To restore")

    BINARY_COLS = ["is_nearby_city_prestigious", "has_garage", "has_garden",
                   "has_terrace", "furnished", "has_elevator"]

    # ---- PHASE 1: CLEAN ----
    cleaner = DataCleaning("data/properties_final.csv")
    cleaner.load().clean_column_names()
    cleaner.clean_data(
        zero_area_pairs=ZERO_AREA_PAIRS,
        drop_states=DROP_STATES,
        drop_cols=DROP_COLS,
        missing_threshold=0.05,
        always_keep=("price", "living_area_m2"),
        max_km=300,
    )
    df_clean = cleaner.get_data()

    # ---- PHASE 2: PREPROCESS ----
    pre = PreprocessData(df_clean)
    pre.preprocess_data(
        ordinal_specs=ORDINAL_SPECS,
        onehot_cols=ONEHOT_COLS,
        binary_cols=BINARY_COLS,
        onehot_drop_first=False,
    )
    pre.overview()
    pre.save("data/Dataframe_Clean_encoded.csv")