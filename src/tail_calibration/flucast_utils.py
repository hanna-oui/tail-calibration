import os
import pickle
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.compute as pc
import hashlib 
import shutil
import json
from typing import Optional


# ── valid filter keys per dataset ────────────────────────────────────────────
VALID_FILTER_KEYS = {
    "time_series": {"target", "location"},
    "oracle":      {"target", "location"},
    "forecasts":   {"target", "output_type", "horizon", "model_id", "location"},
}

DEFAULT_FILTERS = {
    "time_series": {"target": ["wk inc flu hosp"]},
    "oracle":      {},
    "forecasts":   {
        "target":      ["wk inc flu hosp"],
        "output_type": ["quantile"],
        "horizon":     [0],
        "model_id":    ["FluSight-ensemble"],
    },
}

def _make_cache_key(filters: dict) -> str:
    """Generate a stable MD5 hash string from a filter dict, used as the cache filename.

    Args:
        filters: Validated filter dict (output of _validate_filters).
    """
    serialized = json.dumps(filters, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()

def _validate_filters(filters: Optional[dict]) -> dict:
    """Validate and normalise the top-level filter dict.

    Each top-level key must be a dataset name in VALID_FILTER_KEYS.
    Each inner value must be a list (or will be coerced into one).
    Missing dataset keys are filled in from DEFAULT_FILTERS.

    Args:
        filters: Dict mapping dataset name -> {column: values}. Pass None to use DEFAULT_FILTERS.
    """
    if filters is None:
        return DEFAULT_FILTERS

    validated = {}
    for dataset_key, col_filters in filters.items():
        if dataset_key not in VALID_FILTER_KEYS:
            raise KeyError(
                f"Unknown dataset key '{dataset_key}'. "
                f"Valid keys: {set(VALID_FILTER_KEYS)}"
            )
        if not isinstance(col_filters, dict):
            raise TypeError(
                f"Filters for '{dataset_key}' must be a dict, "
                f"got {type(col_filters).__name__}."
            )
        validated_cols = {}
        for col, values in col_filters.items():
            if col not in VALID_FILTER_KEYS[dataset_key]:
                raise KeyError(
                    f"Unknown filter column '{col}' for dataset '{dataset_key}'. "
                    f"Valid columns: {VALID_FILTER_KEYS[dataset_key]}"
                )
            # force list
            if isinstance(values, list):
                validated_cols[col] = values
            elif isinstance(values, (str, int, float)):
                validated_cols[col] = [values]        # silently wrap scalar
            else:
                try:
                    validated_cols[col] = list(values)
                except TypeError:
                    raise TypeError(
                        f"Filter values for '{dataset_key}.{col}' must be a list "
                        f"or list-like, got {type(values).__name__}."
                    )
        validated[dataset_key] = validated_cols

    # fill in any missing dataset keys with defaults
    for key in VALID_FILTER_KEYS:
        if key not in validated:
            validated[key] = DEFAULT_FILTERS[key]

    return validated


def _apply_pandas_filters(df: pd.DataFrame, col_filters: dict) -> pd.DataFrame:
    """Filter a DataFrame by isin checks on one or more columns.

    Args:
        df: DataFrame to filter.
        col_filters: Dict mapping column name -> list of allowed values.
    """
    mask = pd.Series(True, index=df.index)
    for col, values in col_filters.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe.")
        mask &= df[col].isin(values)
    return df.loc[mask].copy()


def _build_pyarrow_filter(col_filters: dict):
    """Build a PyArrow filter expression from a col→list dict.

    Args:
        col_filters: Dict mapping column name -> list of allowed values.
    """
    expr = None
    for col, values in col_filters.items():
        if col == "horizon":
            # horizon is numeric — use isin on integers
            part = pc.field(col).isin(values)
        else:
            part = pc.field(col).isin(values)
        expr = part if expr is None else (expr & part)
    return expr


# ── main loader ───────────────────────────────────────────────────────────────

def load_flusight_data(filters: Optional[dict] = None, refresh: bool = False):
    """Load FluSight forecast, time-series, and oracle data from S3, with local pickle caching.

    Returns a tuple of (time_series, oracle, forecasts) as DataFrames.

    Args:
        filters: Optional filter dict (see _validate_filters). Defaults to DEFAULT_FILTERS.
        refresh: If True, delete any existing cache and re-fetch from S3.
    """
    validated = _validate_filters(filters)
    cache_key = _make_cache_key(validated)

    cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".flusight_data_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    if refresh and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        print("Loading existing FluSight data from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Fetching FluSight data from S3...")
    s3 = fs.S3FileSystem(anonymous=True)

    def _load():
        # ── forecasts ──────────────────────────────────────────────────────
        mo = ds.dataset(
            "cdcepi-flusight-forecast-hub/model-output/",
            filesystem=s3,
            format="parquet",
        )
        arrow_filter = _build_pyarrow_filter(validated["forecasts"])
        forecasts = mo.to_table(filter=arrow_filter).to_pandas()
        forecasts[["target_end_date", "reference_date"]] = (
            forecasts[["target_end_date", "reference_date"]].apply(pd.to_datetime)
        )

        # ── time_series → oracle_asof ──────────────────────────────────────
        with s3.open_input_file(
            "cdcepi-flusight-forecast-hub/target-data/time-series.csv"
        ) as f:
            time_series = pd.read_csv(f)

        time_series = _apply_pandas_filters(time_series, validated["time_series"])
        time_series[["target_end_date", "as_of"]] = (
            time_series[["target_end_date", "as_of"]].apply(pd.to_datetime)
        )

        # ── oracle ─────────────────────────────────────────────────────────
        with s3.open_input_file(
            "cdcepi-flusight-forecast-hub/target-data/oracle-output.csv"
        ) as f:
            oracle = pd.read_csv(f)
        if validated["oracle"]:
            oracle = _apply_pandas_filters(oracle, validated["oracle"])
        return time_series, oracle, forecasts

    data = _load()
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data



def uniquely_identified(df: pd.DataFrame, by: list[str]) -> bool:
    """Return True if the given columns uniquely identify every row in df.

    Args:
        df: DataFrame to check.
        by: List of column names forming the candidate key.
    """
    return df.duplicated(subset=by).sum() == 0

def widen(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format forecast DataFrame to wide format with one quantile column per level.

    Args:
        forecast_df: Long-format DataFrame with output_type_id and value columns.
    """
    forecasts_wide = forecast_df.pivot_table(
        index=[c for c in forecast_df.columns if c not in ["output_type", "output_type_id", "value"]],
        columns="output_type_id",
        values="value"
    ).reset_index()
    forecasts_wide.columns.name = None
    new_quantile_cols = [c for c in forecasts_wide.columns if c not in forecast_df.columns]
    forecasts_wide = forecasts_wide.rename(columns={c: f"quantile_{c}" for c in new_quantile_cols})
    return forecasts_wide

def filter_dates(oracle_df: pd.DataFrame, date_ranges: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Filter oracle data to specific season windows matched by as_of date.

    Args:
        oracle_df: Oracle DataFrame with target_end_date and as_of columns.
        date_ranges: Maps (start, end) date strings -> as_of date string.
            Each entry selects rows where target_end_date is in [start, end]
            and as_of equals the specified snapshot date.
    """
    masks = []
    for (start, end), asof in date_ranges.items():
        mask = (
            (oracle_df["target_end_date"] >= pd.Timestamp(start)) &
            (oracle_df["target_end_date"] <= pd.Timestamp(end)) &
            (oracle_df["as_of"] == pd.Timestamp(asof))
        )
        masks.append(mask)

    combined = pd.concat([oracle_df[m] for m in masks], ignore_index=True)
    return combined


def merge_and_process(forecasts: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    """Merge forecasts with oracle data and pivot to wide format.

    Args:
        forecasts: Long-format forecast DataFrame.
        oracle: Oracle DataFrame uniquely identified by target, target_end_date, location.
    """
    assert uniquely_identified(oracle, by=["target", "target_end_date", "location"])
    assert uniquely_identified(forecasts, by=["target", "target_end_date", "location", "horizon", "model_id", "output_type_id"])
    merged = forecasts.merge(oracle, on=["target", "target_end_date", "location"], how="left")
    merged_wide = widen(merged)
    return merged_wide


def build_evaluation_df(forecasts: pd.DataFrame,
                        oracle: pd.DataFrame,
                        date_ranges: dict[tuple[str, str], str],
                        id_cols = ["target", "reference_date", "location", "horizon", "model_id"]) -> pd.DataFrame:
    """Build a wide evaluation DataFrame by filtering oracle dates and merging with forecasts.

    Args:
        forecasts: Long-format forecast DataFrame.
        oracle: Oracle DataFrame.
        date_ranges: Season windows passed to filter_dates.
        id_cols: Columns that should uniquely identify each row in the result.
    """
    filtered_oracle = filter_dates(oracle, date_ranges)
    evaluation_df = merge_and_process(forecasts, filtered_oracle)
    assert uniquely_identified(evaluation_df, by=id_cols)
    return evaluation_df

def build_evaluation_dict(
    forecasts: pd.DataFrame,
    oracle: pd.DataFrame,
    date_ranges: dict[tuple[str, str], str],
    model_id: list[str] = ['FluSight-ensemble'],
) -> dict[str, pd.DataFrame]:
    """Build a dict mapping model ID -> evaluation DataFrame, one entry per requested model.

    Args:
        forecasts: Long-format forecast DataFrame.
        oracle: Oracle DataFrame.
        date_ranges: Season windows passed to filter_dates.
        model_id: List of model IDs to include in the output dict.
    """
    df = build_evaluation_df(forecasts, oracle, date_ranges)
    return {mid: df[df["model_id"] == mid].reset_index(drop=True) for mid in model_id}

date_ranges = {
    ("2023-10-14", "2024-04-27"): "2024-04-27",
    ("2024-10-14", "2025-06-21"): "2025-07-23"
}

