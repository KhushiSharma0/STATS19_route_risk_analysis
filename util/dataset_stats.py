import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summarise_and_visualise_route_school_stats(
    df: pd.DataFrame,
    route_col: str = "route_id",
    school_col: str = "school_name",
    length_col: str = "length_km",
    collision_col: str = "accident_index",
    top_n: int = 10
):
    """
    Computes and visualises route and school-level collision stats.

    Args:
        df (pd.DataFrame): Input data with route, school, length, and collision info
        route_col (str): Column containing route identifiers
        school_col (str): Column containing school names
        length_col (str): Column containing route length (in km)
        collision_col (str): Column containing collision identifiers
        top_n (int): Number of top entries to display/plot
    """

    # ----------------------------
    # ROUTE-LEVEL METRICS
    # ----------------------------
    route_stats = (
        df.groupby(route_col)
        .agg(
            school=(school_col, "first"),
            length_km=(length_col, "first"),
            collisions=(collision_col, "count")
        )
        .reset_index()
    )
    route_stats["collisions_per_km"] = route_stats["collisions"] / route_stats["length_km"]

    # Averages
    avg_route_collisions = route_stats["collisions"].mean()
    avg_collisions_per_km = route_stats["collisions_per_km"].mean()

    # Rankings
    top_routes = route_stats.nlargest(top_n, "collisions")
    top_routes_density = route_stats.nlargest(top_n, "collisions_per_km")

    # ----------------------------
    # SCHOOL-LEVEL METRICS
    # ----------------------------
    school_stats = (
        route_stats.groupby("school")
        .agg(
            total_collisions=("collisions", "sum"),
            avg_route_collisions=("collisions", "mean"),
            num_routes=("route_id", "count")
        )
        .reset_index()
    )
    avg_school_collisions = school_stats["total_collisions"].mean()
    top_schools = school_stats.nlargest(top_n, "total_collisions")

    # ----------------------------
    # TEXT OUTPUT
    # ----------------------------
    print("📊 Averages:")
    print(f"• Avg collisions per route: {avg_route_collisions:.2f}")
    print(f"• Avg collisions per km: {avg_collisions_per_km:.2f}")
    print(f"• Avg collisions per school: {avg_school_collisions:.2f}\n")

    print("🏆 Top Routes by Collision Count:")
    print(top_routes[[route_col, "school", "collisions"]].to_string(index=False))
    print("\n📏 Top Routes by Collisions per km:")
    print(top_routes_density[[route_col, "school", "collisions_per_km"]].to_string(index=False))
    print("\n🏫 Top Schools by Total Collisions:")
    print(top_schools.to_string(index=False))

    # ----------------------------
    # PLOTTING HELPERS
    # ----------------------------
    sns.set(style="whitegrid")

    def barplot(df, x, y, title, color, vline=None):
        plt.figure(figsize=(8, 4))
        sns.barplot(x=x, y=y, data=df.sort_values(x), palette=color)
        if vline:
            plt.axvline(vline, color="red", linestyle="--", label=f"Avg: {vline:.1f}")
            plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def histplot(data, title, xlabel, avg):
        plt.figure(figsize=(8, 4))
        sns.histplot(data, bins=30, color="skyblue")
        plt.axvline(avg, color="red", linestyle="--", label=f"Avg: {avg:.1f}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # PLOTS
    # ----------------------------
    histplot(route_stats["collisions"], "Distribution of Collisions per Route", "Collisions", avg_route_collisions)
    histplot(route_stats["collisions_per_km"], "Distribution of Collisions per km", "Collisions per km", avg_collisions_per_km)

    barplot(top_schools, "total_collisions", "school", "Top Schools by Total Collisions", "Blues_d", avg_school_collisions)
    barplot(top_routes, "collisions", route_col, "Top Routes by Collision Count", "Greens_d")
    barplot(top_routes_density, "collisions_per_km", route_col, "Top Routes by Collisions per km", "Reds_d")
