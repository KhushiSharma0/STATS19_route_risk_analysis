import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 1. Bar Chart: Top N Routes by Metric
def plot_top_routes_by_metric(df, metric, ax, top_n=10, ascending=False):
    if "avg_casualty_severity" in metric:
        ascending = True  # higher avg severity = safer
    top_df = df.sort_values(metric, ascending=ascending).head(top_n)
    sns.barplot(data=top_df, x=metric, y="route_id", ax=ax, hue='school_name')
    ax.set_title(f"Top {top_n} Routes by {metric.replace('_', ' ').title()}")
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel("Route ID")
    ax.tick_params(axis='y', labelsize=8)
    ax.get_legend().remove()



# 2. Scatter Plot: Route Length vs Risk Metric
def plot_length_vs_metric(df, metric, ax):
    sns.scatterplot(data=df, x="length_km", y=metric, hue="borough", ax=ax)
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Route Length")
    ax.set_xlabel("Length (km)")
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend(title="Borough", bbox_to_anchor=(1.05, 1), loc='upper left')



# 3. Boxplot: Metric Distribution by Borough
def plot_borough_distribution(df, metric, ax):
    sns.boxplot(data=df, x="borough", y=metric, ax=ax)
    ax.set_title(f"{metric.replace('_', ' ').title()} by Borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45)



# 4. Radar Plot: For a Single Route (used separately, not in grid)
def plot_route_radar(df, route_id):
    metrics = [
        "collision_count_per_km",
        "casualty_count_per_km",
        "avg_casualty_severity_per_km",
        "serious_fatal_casualty_count_per_km"
    ]
    route = df[df["route_id"] == route_id]
    if route.empty:
        print("Route not found.")
        return

    values = route[metrics].values.flatten().tolist()
    labels = [m.replace('_', ' ').title() for m in metrics]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Route Risk Profile: {route['route_id'].values[0]}")
    plt.tight_layout()
    plt.show()

def plot_metric_distribution(df, metric="collision_count"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import poisson

    values = df[metric].dropna()
    mu = values.mean()

    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=range(int(values.max()) + 2), stat="count", kde=False, color="skyblue", edgecolor="black")
    
    # # Overlay Poisson expected frequency
    # x_vals = np.arange(0, int(values.max()) + 1)
    # poisson_pmf = poisson.pmf(x_vals, mu) * len(values)
    # plt.plot(x_vals, poisson_pmf, marker='o', linestyle='--', color='red', label=f'Poisson($\lambda$={mu:.2f})')

    plt.title(f"Distribution of {metric.replace('_', ' ').title()} per Route")
    plt.xlabel(f"{metric.replace('_', ' ').title()}")
    plt.ylabel("Number of Routes")
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_gaussian_fit(df, metric, bins=30, stat='count', kde=False, show_fit=True):
    """
    Plots a histogram of a metric with an optional Gaussian overlay (only if stat='density').

    Parameters:
    - df: pandas DataFrame
    - metric: column name to plot
    - bins: number of bins (or custom bin edges)
    - stat: 'count', 'density', 'probability', etc.
    - kde: whether to overlay Seaborn KDE
    - show_fit: whether to overlay a Gaussian fit (only if stat='density')
    """
    values = df[metric].dropna()
    mu, std = values.mean(), values.std()

    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=bins, stat=stat, color="skyblue", edgecolor="black", kde=kde)

    if stat == 'density' and show_fit:
        x = np.linspace(values.min(), values.max(), 500)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, 'r--', label=f'Gaussian Fit\nμ={mu:.2f}, σ={std:.2f}')
        plt.legend()

    plt.title(f"Distribution of {metric.replace('_', ' ').title()}")
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel(stat.title())
    plt.tight_layout()
    plt.show()



# 5. Grid Wrapper: Multiple Subplots from One Plot Function
def plot_metrics_grid(df, metric_list, plot_func, ncols=2, figsize=(12, 6), **kwargs):
    n_plots = len(metric_list)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
    
    axes = axes.flatten() if isinstance(axes, (np.ndarray, list)) else [axes]

    for i, metric in enumerate(metric_list):
        ax = axes[i]
        plot_func(df, metric, ax=ax, **kwargs)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

def plot_metrics_loop(df, metric_list, plot_func, pause=False, **kwargs):
    """
    Plots each metric one-by-one using the given plot function.

    Parameters:
    - df: DataFrame with your metrics
    - metric_list: list of column names (metrics)
    - plot_func: a function like plot_top_routes_by_metric(df, metric, ax)
    - pause: if True, wait for user input between plots
    - kwargs: additional arguments passed to plot_func
    """
    import matplotlib.pyplot as plt

    for metric in metric_list:
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust width/height as needed
        plot_func(df, metric, ax=ax, **kwargs)
        plt.tight_layout()
        plt.show()

        if pause:
            input(f"Showing: {metric}. Press Enter to continue...")
