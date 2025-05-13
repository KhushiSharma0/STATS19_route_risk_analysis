import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# Base commute risk metrics table, can be filtered using any stats19 parameter (year or casualty type)
def build_commute_risk_metrics(df, filter_col=None, filter_values=None):
    """
    Builds a route-level commute risk metrics table from the full STATS19 + routes combined dataset.

    Parameters:
    - df (pd.DataFrame): The full combined dataset.
    - filter_col (str or None): Optional column name to filter on (e.g., 'accident_year', 'casualty_type').
    - filter_values (list, int, or str): Value(s) to filter by. Accepts single value or list.

    Returns:
    - pd.DataFrame: Route-level metrics table with collisions, casualties, severity, etc.
    """
    import pandas as pd
    # Optional filtering
    if filter_col and filter_values is not None:
        if isinstance(filter_values, list):
            df = df[df[filter_col].isin(filter_values)]
        else:
            df = df[df[filter_col] == filter_values]
    # Columns needed for metrics
    cols_needed = [
        "route_id", "school_name", "borough", "length_km",
        "accident_index", "number_of_casualties", "casualty_reference"      
    ] #  "casualty_severity"
    df_metrics = df[cols_needed].copy()
    # Drop join duplicates
    df_metrics = df_metrics.drop_duplicates(subset=["route_id", "accident_index", "casualty_reference"])
    # Base route info
    base = df_metrics[["route_id", "school_name", "borough", "length_km"]].drop_duplicates()
    # Collisions
    collisions = (
        df_metrics[["route_id", "accident_index"]]
        .drop_duplicates()
        .groupby("route_id")
        .size()
        .reset_index(name="collision_count")
    )
    # Casualties
    casualties = (
        df_metrics.drop_duplicates(subset=["route_id", "accident_index"])[["route_id", "number_of_casualties"]]
        .groupby("route_id")["number_of_casualties"]
        .sum()
        .reset_index(name="casualty_count")
    )
    # # Severity
    # avg_severity = (
    #     df_metrics
    #     .groupby("route_id")["casualty_severity"]
    #     .mean()
    #     .reset_index(name="avg_casualty_severity")
    # )
    # # Serious + fatal
    # serious_fatal = (
    #     df_metrics[df_metrics["casualty_severity"].isin([1, 2])]
    #     .groupby("route_id")
    #     .size()
    #     .reset_index(name="serious_fatal_casualty_count")
    # )
    # Merge everything
    result = base
    for table in [collisions, casualties]: #  avg_severity, serious_fatal
        result = result.merge(table, on="route_id", how="left")
    # result["serious_fatal_casualty_count"] = result["serious_fatal_casualty_count"].fillna(0).astype(int)
    return result


def add_normalised_metrics(table, metric_cols, length_col="length_km"):
    """
    Adds _per_km versions of given metric columns by dividing them by route length.

    Parameters:
    - table (pd.DataFrame): DataFrame containing the base metrics.
    - metric_cols (list of str): Column names to normalise (e.g., ['collision_count', 'casualty_count']).
    - length_col (str): Name of the column that represents route length.

    Returns:
    - pd.DataFrame: Modified DataFrame with new _per_km columns.
    """
    table = table.copy()
    
    for col in metric_cols:
        per_km_col = f"{col}_per_km"
        table[per_km_col] = table[col] / table[length_col]
    
    # Handle infinities and NaNs
    per_km_cols = [f"{col}_per_km" for col in metric_cols]
    table.replace([np.inf, -np.inf], np.nan, inplace=True)
    table.dropna(subset=per_km_cols, inplace=True)

    return table

# def add_percentile_metrics(table, columns, inplace=False):
#     """
#     Adds percentile ranks (0 to 1) for selected columns, and optionally a composite percentile score and rank.

#     Parameters:
#     - table (pd.DataFrame): The base metrics table.
#     - columns (list of str): List of metric column names (e.g., collisions_per_km).
#     - add_composite (bool): Whether to compute a composite percentile from the individual ones.
#     - prefix (str): Prefix for the composite columns (e.g., 'risk' → 'risk_score', 'risk_percentile').
#     - inplace (bool): Whether to modify the original table in-place.

#     Returns:
#     - pd.DataFrame: The modified table with _percentile columns and optional composite.
#     """
#     if not inplace:
#         table = table.copy()

#     # Add individual percentile columns
#     for col in columns:
#         p_col = f"{col}_percentile"
#         if p_col not in table.columns:
#             table[p_col] = table[col].rank(pct=True)
#     return table


def add_poisson_risk_metrics(table, count_metrics, length_col="length_km", prefix="poisson"):
    """
    Adds Poisson-based per-km risk metrics for each count metric.

    For each metric, this adds:
    - <prefix>_risk_ratio_<metric>
    - <prefix>_excess_per_km_<metric>

    Parameters:
    - table (pd.DataFrame): Route-level table with count metrics and length.
    - count_metrics (list of str): Count columns to model (e.g., ['collision_count']).
    - length_col (str): Column for route length (used for offset and normalisation).
    - prefix (str): Prefix for new column names (default = 'poisson').

    Returns:
    - pd.DataFrame: Table with new per-km Poisson metrics.
    """
    table = table.copy()
    table["log_length"] = np.log(table[length_col] + 1e-8)  # offset

    for metric in count_metrics:
        # Fit Poisson GLM: count ~ offset(log(length))
        model = smf.glm(
            formula=f"{metric} ~ 1",
            data=table,
            family=sm.families.Poisson(),
            offset=table["log_length"]
        ).fit()

        # Predicted expected value per route
        expected = model.predict()

        # Add per-km metrics
        table[f"{prefix}_risk_ratio_{metric}"] = table[metric] / expected
        table[f"{prefix}_excess_per_km_{metric}"] = (table[metric] - expected) / table[length_col]

    return table.drop(columns="log_length")


def calculate_composite_risk_score(df, scaled_metrics, weights=None, score_col="composite_risk_score"):
    """
    Calculates a composite risk score using scaled metrics (0–1).

    Parameters:
    - df: DataFrame containing scaled metrics
    - scaled_metrics: list of column names (e.g., ["collision_count_per_km_scaled", ...])
    - weights: list or array of same length as scaled_metrics, or None for equal weighting
    - score_col: name of output column

    Returns:
    - DataFrame with new score column
    """
    if weights is None:
        weights = [1.0] * len(scaled_metrics)
    
    weights = np.array(weights) / np.sum(weights)  # normalize weights
    df[score_col] = df[scaled_metrics].values @ weights  # matrix-style weighted sum
    return df



