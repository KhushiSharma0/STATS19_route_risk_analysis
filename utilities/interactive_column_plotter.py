import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output


def get_missing_percent_series(df, year_col, col_name):
    """Returns a Series of % missing (NaN or -1) values per year for a given column."""
    if pd.api.types.is_numeric_dtype(df[col_name]):
        is_missing = df[col_name].isnull() | (df[col_name] == -1) | (df[col_name] == -1.0)
    else:
        is_missing = df[col_name].isnull()

    years = sorted(df[year_col].unique())
    year_counts = df[year_col].value_counts().sort_index()
    yearly_missing = df[is_missing].groupby(df[year_col]).size()
    percent_missing = (yearly_missing / year_counts * 100).reindex(years, fill_value=0)

    return percent_missing


def get_overall_missing_percent(df, year_col):
    """Returns a Series of overall % missing values for all columns (NaN or -1)."""
    overall_missing = {}
    for col in df.columns:
        if col == year_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            is_missing = df[col].isnull() | (df[col] == -1) | (df[col] == -1.0)
        else:
            is_missing = df[col].isnull()
        percent = (is_missing.sum() / len(df)) * 100
        overall_missing[col] = percent
    return pd.Series(overall_missing).sort_values(ascending=False)


def plot_multiple_columns(df, year_col, columns):
    """Plots % missing values per year for multiple selected columns."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, col in enumerate(columns):
        series = get_missing_percent_series(df, year_col, col)
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines+markers',
            name=col,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(symbol=i % 12, size=6),
            hovertemplate=f"<b>{col}</b><br>Year: %{{x}}<br>Missing: %{{y:.2f}}%<extra></extra>"
        ))

    fig.update_layout(
        title=f"Missing or -1 Values Over the Years ({len(columns)} columns)",
        xaxis_title="Year",
        yaxis_title="Percent Missing",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=500,
        hovermode='x unified',  # 👈 shared tooltip + vertical line
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()


def advanced_missing_explorer(df, year_col='accident_year'):
    """
    Interactive widget UI to explore % missing data over time.
    Includes filtering, select-all toggle, and multi-column plotting.
    """
    missing_summary = get_overall_missing_percent(df, year_col)

    # UI Widgets
    exclude_zero = widgets.Checkbox(value=True, description="Exclude 0% missing columns")
    slider = widgets.SelectionSlider(
        options=list(range(0, 101, 10)),
        value=10,
        description='Min % Missing:',
        style={'description_width': 'initial'}
    )
    select_all = widgets.Checkbox(value=False, description="Select/Deselect All")
    column_checks_box = widgets.VBox()
    column_scroll_area = widgets.Box([column_checks_box], layout=widgets.Layout(overflow='auto', height='300px'))
    output = widgets.Output()

    # Checkbox population
    def update_checkbox_list(*args):
        threshold = slider.value
        filtered = missing_summary.copy()
        if exclude_zero.value:
            filtered = filtered[filtered > 0]
        filtered = filtered[filtered >= threshold]
        column_checks_box.children = [widgets.Checkbox(value=select_all.value, description=col) for col in filtered.index]

    # Plotting callback
    def update_plot(change=None):
        checked = [cb.description for cb in column_checks_box.children if cb.value]
        with output:
            clear_output(wait=True)
            if checked:
                plot_multiple_columns(df, year_col, checked)

    # Toggle select all
    def toggle_all(change):
        for cb in column_checks_box.children:
            cb.value = select_all.value

    # Watch for checkbox updates
    def watch_checkboxes():
        for cb in column_checks_box.children:
            cb.observe(update_plot, names='value')

    def refresh_and_listen(*args):
        update_checkbox_list()
        watch_checkboxes()
        update_plot()

    # Initial setup
    slider.observe(refresh_and_listen, names='value')
    exclude_zero.observe(refresh_and_listen, names='value')
    select_all.observe(toggle_all, names='value')

    # Layout
    controls = widgets.VBox([
        exclude_zero,
        slider,
        select_all,
        widgets.Label("Columns:"),
        column_scroll_area
    ], layout=widgets.Layout(width='35%', padding='10px'))

    refresh_and_listen()
    display(widgets.HBox([controls, output]))
