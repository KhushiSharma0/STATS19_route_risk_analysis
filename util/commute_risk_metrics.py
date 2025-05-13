import pandas as pd

# 1. singular metrics

def compute_singular_metrics(df: pd.DataFrame, route_col: str = 'route_id') -> pd.DataFrame:
    """
    Computes basic route-level raw metrics:
    - total collisions
    - total casualties
    - total serious casualties
    - total fatalities

    Parameters:
        df (pd.DataFrame): DataFrame containing STATS19 + route-mapped data
        route_col (str): Name of the route ID column (default 'route_id')

    Returns:
        pd.DataFrame: route-level summary metrics
    """
    return df.groupby(route_col).agg(
        total_collisions=('accident_index', 'nunique'),
        total_casualties=('casualty_reference', 'count'),
        total_serious=('casualty_severity', lambda x: (x == 2).sum()),
        total_fatal=('casualty_severity', lambda x: (x == 1).sum())
    ).reset_index()


# 2. Normalised length metrics

def compute_normalised_metrics(df: pd.DataFrame, route_length_col: str = 'route_length_km') -> pd.DataFrame:
    """
    Computes per-kilometre metrics to normalize for exposure:
    - collisions/km
    - casualties/km
    - serious/km
    - fatal/km
    """
    metrics = compute_singular_metrics(df)
    metrics[route_length_col] = df.groupby('route_id')[route_length_col].first().values
    
    return metrics.assign(
        collisions_per_km = metrics.total_collisions / metrics[route_length_col],
        casualties_per_km = metrics.total_casualties / metrics[route_length_col],
        serious_per_km = metrics.total_serious / metrics[route_length_col],
        fatal_per_km = metrics.total_fatal / metrics[route_length_col],
    )
