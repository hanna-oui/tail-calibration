import numpy as np
import pandas as pd
import warnings
from typing import Optional
from tail_calibration.types import TailCalibrationTuple, p

def assert_percentile(p: p) -> p:
    if 0 <= p <= 100:
        return p
    raise ValueError(f"Percentile {p} is out of range. Must be between 0 and 100.")

def percentile_value(df: pd.DataFrame, oracle_col: str, p: p) -> float:
    """p in [0, 100], e.g. p=90 for 90th percentile"""
    return df[oracle_col].quantile(assert_percentile(p) / 100)

def compute_cdf_value(row: pd.Series, x: float, quantile_cols: list[str], quantile_levels: list[float]) -> float:
    input_values, input_levels= np.asarray(row[quantile_cols], dtype=np.float64), np.asarray(quantile_levels, dtype=np.float64)
    values, levels = np.concatenate([[-np.inf], input_values, [np.inf]]), np.concatenate([[0.0], input_levels, [1.0]])
    assert np.all(np.diff(values) >= 0), f"Quantile crossing detected: {values}"
    assert np.all(np.diff(levels) > 0), f"Quantile levels must be increasing: {quantile_levels}"
    return float(np.interp(x, values, levels))

def parse_quantile_cols(df: pd.DataFrame, prefix: str = "quantile_") -> tuple[list[str], list[float]]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    levels = [float(c.replace(prefix, "")) for c in cols]
    cols, levels = zip(*sorted(zip(cols, levels), key=lambda x: x[1]))
    return list(cols), list(levels)

def threshold_CDF_values(df: pd.DataFrame, t: float, quantile_prefix: str = "quantile_") -> pd.Series:
    quantile_cols, quantile_levels = parse_quantile_cols(df, quantile_prefix)
    return df.apply(lambda row: compute_cdf_value(row, t, quantile_cols, quantile_levels), axis=1)

def empirical_CDF_values(df: pd.DataFrame, oracle_col: str, quantile_prefix: str = "quantile_") -> pd.Series:
    quantile_cols, quantile_levels = parse_quantile_cols(df, quantile_prefix)
    return df.apply(lambda row: compute_cdf_value(row, row[oracle_col], quantile_cols, quantile_levels), axis=1)

def left_limit_CDF_values(df: pd.DataFrame, oracle_col: str, quantile_prefix: str = "quantile_") -> pd.Series:
    quantile_cols, quantile_levels = parse_quantile_cols(df, quantile_prefix)
    return df.apply(lambda row: compute_cdf_value(row, row[oracle_col] - 1, quantile_cols, quantile_levels), axis=1)

def above_threshold(df: pd.DataFrame, oracle_col: str, threshold: float) -> pd.DataFrame:
    above = df[df[oracle_col] > threshold]
    assert len(above) > 0, "No observations above the threshold. Check the threshold value and oracle column."
    return above

def threshold_cardinality(df: pd.DataFrame, oracle_col: str, p: Optional[float] = None, t: Optional[float] = None) -> int:
    assert (p is not None) ^ (t is not None), "Must specify exactly one of p or t"
    if p is not None:
        threshold = percentile_value(df, oracle_col, assert_percentile(p))
    else:
        assert t is not None
        threshold = t
    return len(above_threshold(df, oracle_col, threshold))
    
def survival_probabilities(df: pd.DataFrame, t: float, quantile_prefix: str = "quantile_") -> pd.Series:
    quantile_cols, quantile_levels = parse_quantile_cols(df, quantile_prefix)
    return 1 - df.apply(lambda row: compute_cdf_value(row, t, quantile_cols, quantile_levels), axis=1)

def excess_PIT(
    merged_df: pd.DataFrame,  # pre-merged forecast + oracle, filtered to single model
    oracle_col: str,
    t: float,
    randomized: bool = True,
    quantile_prefix: str = "quantile_"
) -> pd.Series:
    
    exceedance_df = above_threshold(merged_df, oracle_col, t)
    F_t = threshold_CDF_values(exceedance_df, t, quantile_prefix)
    F_y = empirical_CDF_values(exceedance_df, oracle_col, quantile_prefix)

    if randomized:
        F_y_minus = left_limit_CDF_values(exceedance_df, oracle_col, quantile_prefix)
        V = np.random.uniform(0, 1, size=len(exceedance_df))

        # if F(t) = 1, set ePIT to 1 to avoid division by zero. This is a conservative choice that treats all exceedances as the most extreme possible.
        ePIT_values = np.where(
        (1 - F_t) > 0,
        ((1 - V) * (F_y_minus - F_t) + V * (F_y - F_t)) / (1 - F_t),
        1.0
        )
            
    else:
        ePIT_values = np.where(
        (1 - F_t) > 0,
        (F_y - F_t) / (1 - F_t),
        1.0
        )

    return pd.Series(ePIT_values).reset_index(drop=True)



def tail_calibration(
    merged_df: pd.DataFrame,
    id_col: str,
    oracle_col: str,
    p: p | list[p],
    randomized: bool = True,
    quantile_prefix: str = "quantile_"
) -> dict[p, TailCalibrationTuple]:

    if isinstance(p, (int, float)):
        p = [p]
    
    p = [assert_percentile(percentile) for percentile in p]

    threshold_to_ePITs = {}    
    for percentile in p:
        
        all_ePIT_values = []
        total_I_t = 0
        total_survival = 0.0
        for unit_id in merged_df[id_col].unique():
            merged_df_unit_id = merged_df[merged_df[id_col] == unit_id]

            t = percentile_value(merged_df_unit_id, oracle_col, percentile) 
            I_t = threshold_cardinality(merged_df_unit_id, oracle_col, t=t) 
            ePIT_values = excess_PIT(merged_df_unit_id, oracle_col, t, randomized, quantile_prefix) 
            S_survival = survival_probabilities(merged_df_unit_id, t, quantile_prefix=quantile_prefix).sum() 
            
            all_ePIT_values.append(ePIT_values)
            total_I_t += I_t
            total_survival += S_survival
            
        pooled_pit = pd.concat(all_ePIT_values).reset_index(drop=True)
        occ_ratio = total_I_t / total_survival
        severity_func = lambda u, pit=pooled_pit.to_numpy(), it=total_I_t: (pit <= u).sum() / it
        tail_calibration_func = lambda u, sf=severity_func, occ=occ_ratio: sf(u) * occ

        threshold_to_ePITs[percentile] = {'occurrence_ratio': occ_ratio, 
                                          'severity_func': severity_func, 
                                          'tail_calibration_func': tail_calibration_func, 
                                          'pit_values': pooled_pit, 
                                          'n_exceedances': total_I_t}
    
    return threshold_to_ePITs

def tail_calibration_by_id(
    merged_df: pd.DataFrame,
    id_col: str,
    oracle_col: str,
    p: p | list[p],
    randomized: bool = True,
    quantile_prefix: str = "quantile_"
) -> dict[str, dict[p, TailCalibrationTuple]]:

    if isinstance(p, (int, float)):
        p = [p]

    results = {}

    for unit_id in merged_df[id_col].unique():
        merged_df_unit_id = merged_df[merged_df[id_col] == unit_id]
        try:
            results[unit_id] = tail_calibration(
                merged_df_unit_id, id_col, oracle_col, p, randomized, quantile_prefix
            )
        except AssertionError:
            warnings.warn(f"Skipping '{unit_id}' — no observations above threshold.")

    return results


def evaluate_models(
    evaluation_df: pd.DataFrame,
    id_col: str,
    oracle_col: str = "observation",
    percentiles: list[float] = [75, 85, 95],
    u: np.ndarray = np.linspace(0, 1, 1000),
    quantile_prefix: str = "quantile_",
    baseline_model: str = "FluSight-baseline",
) -> pd.DataFrame:
    import scoringrules as sr

    quantile_cols = sorted(
        [c for c in evaluation_df.columns if c.startswith(quantile_prefix)],
        key=lambda c: float(c.replace(quantile_prefix, ""))
    )
    lower_cols = [c for c in quantile_cols if float(c.replace(quantile_prefix, "")) < 0.5]
    upper_cols = [c for c in quantile_cols if float(c.replace(quantile_prefix, "")) > 0.5][::-1]
    median_col = f"{quantile_prefix}0.5"
    alpha = np.array([float(c.replace(quantile_prefix, "")) * 2 for c in lower_cols])

    rows = []
    for model in evaluation_df["model_id"].unique():
        model_df = evaluation_df[evaluation_df["model_id"] == model]

        obs    = model_df[oracle_col].to_numpy()
        median = model_df[median_col].to_numpy()
        lower  = model_df[lower_cols].to_numpy()
        upper  = model_df[upper_cols].to_numpy()
        wis_scores = sr.weighted_interval_score(obs, median, lower, upper, alpha)
        row = {"model_id": model, "mean_wis": wis_scores.mean()}

        try:
            ePIT_map = tail_calibration(
                model_df, id_col, oracle_col, percentiles,
                quantile_prefix=quantile_prefix
            )
            for pct in percentiles:
                combined_values = np.array([ePIT_map[pct]['tail_calibration_func'](ui) for ui in u])
                row[f"tail_sup_{pct}"]      = np.max(np.abs(combined_values - u))
                row[f"tail_l1_{pct}"]       = np.trapezoid(np.abs(combined_values - u), u)
                row[f"n_exceedances_{pct}"] = ePIT_map[pct]['n_exceedances']

        except Exception as e:
            warnings.warn(f"Tail calibration failed for '{model}': {e}")
            for pct in percentiles:
                row[f"tail_sup_{pct}"]      = np.nan
                row[f"tail_l1_{pct}"]       = np.nan
                row[f"n_exceedances_{pct}"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    baseline_wis = result.loc[result["model_id"] == baseline_model, "mean_wis"].values[0]
    result["rwis"] = result["mean_wis"] / baseline_wis
    return result.sort_values("rwis").reset_index(drop=True)