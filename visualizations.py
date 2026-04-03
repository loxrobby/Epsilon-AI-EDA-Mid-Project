"""
Shared Plotly figure builders for EDA notebook and Streamlit.
All functions accept a cleaned DataFrame and return plotly.graph_objects.Figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE_DIR = Path(__file__).resolve().parent

# Consistent template
TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Safe


def _sample_df(df: pd.DataFrame, n: int = 8000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title="Correlation matrix (numeric features)",
        template=TEMPLATE,
        height=700,
        xaxis_tickangle=-45,
    )
    return fig


def plot_univariate_histogram(df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    ttl = title or f"Distribution of {column}"
    fig = px.histogram(df, x=column, nbins=40, title=ttl, template=TEMPLATE)
    fig.update_layout(bargap=0.05)
    return fig


def plot_univariate_bar_categorical(df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    counts = df[column].astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]
    ttl = title or f"Counts: {column}"
    fig = px.bar(counts, x=column, y="count", title=ttl, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)
    fig.update_layout(xaxis_tickangle=-35)
    return fig


def plot_q1_3d_scatter(df: pd.DataFrame) -> go.Figure:
    # 2D density plot to avoid overplotting at 30k rows.
    d = _sample_df(df, n=20000).copy()
    if "perceived_minus_actual" not in d.columns:
        d["perceived_minus_actual"] = (
            d["perceived_productivity_score"] - d["actual_productivity_score"]
        )

    d["gap"] = d["perceived_minus_actual"].astype(float)

    fig = go.Figure(
        data=go.Histogram2d(
            x=d["daily_social_media_time"].astype(float),
            y=d["gap"],
            nbinsx=45,
            nbinsy=45,
            colorscale="Viridis",
            colorbar=dict(title="count"),
            hovertemplate="Time: %{x:.2f}h<br>Gap: %{y:.2f}<br>Count: %{z}<extra></extra>",
        )
    )

    # Overlay median gap by time quartile.
    d["time_quartile"] = pd.qcut(
        d["daily_social_media_time"],
        q=4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
        duplicates="drop",
    )
    med = (
        d.groupby("time_quartile", observed=False)
        .agg(
            x=("daily_social_media_time", "mean"),
            y=("perceived_minus_actual", "median"),
        )
        .reset_index()
    )
    med["time_quartile"] = pd.Categorical(
        med["time_quartile"],
        categories=["Low", "Mid-Low", "Mid-High", "High"],
        ordered=True,
    )
    med = med.sort_values("time_quartile")

    fig.add_trace(
        go.Scatter(
            x=med["x"],
            y=med["y"],
            mode="lines+markers",
            line=dict(color="crimson", width=3),
            marker=dict(size=10),
            name="Median gap by time quartile",
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Q1: Does spending more time on social media widen the gap between how productive we feel and how productive we actually are?",
        height=640,
        xaxis_title="Daily social media time (hours)",
        yaxis_title="Perceived minus actual productivity (gap)",
        bargap=0.05,
    )
    return fig


def plot_q2_notifications_stress(df: pd.DataFrame) -> go.Figure:
    d = _sample_df(df, n=12000).copy()
    d["notif"] = d["number_of_notifications"].astype(float)
    d["stress"] = d["stress_level"].astype(float)
    d["time_bin"] = pd.qcut(
        d["daily_social_media_time"],
        q=4,
        labels=["Low time", "Mid-Low", "Mid-High", "High time"],
        duplicates="drop",
    )

    d_sorted = d.sort_values("notif")
    x = d_sorted["notif"].to_numpy(dtype=float)
    y = d_sorted["stress"].to_numpy(dtype=float)
    smoothed = lowess(y, x, frac=0.18, return_sorted=True)

    r, p = stats.spearmanr(x, y, nan_policy="omit")
    fig = px.scatter(
        d_sorted,
        x="notif",
        y="stress",
        color="time_bin",
        opacity=0.55,
        title="Q2: How do constant phone notifications impact our daily stress levels?",
        template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.add_trace(
        go.Scatter(
            x=smoothed[:, 0],
            y=smoothed[:, 1],
            mode="lines",
            line=dict(color="black", width=3),
            name="LOESS smooth",
        )
    )
    fig.update_layout(
        xaxis_title="Number of notifications (daily)",
        yaxis_title="Stress level",
        height=520,
        legend_title_text="Social media time",
    )
    fig.add_annotation(
        text=f"Spearman ρ={r:.3f}<br>p={p:.2e}",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        font=dict(size=12),
    )
    return fig


def plot_q3_platform_screen_time(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="social_platform_preference",
        y="screen_time_before_sleep",
        box=True,
        points=False,
        color="social_platform_preference",
        template=TEMPLATE,
        title="Q3: Which platform is associated with later bedtime (screen time before sleep)?",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        yaxis_title="Screen time before sleep (hours)",
        showlegend=False,
        height=520,
    )
    return fig


def plot_q4_focus_apps_burnout(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="uses_focus_apps",
        y="days_feeling_burnout_per_month",
        box=True,
        points=False,
        title="Q4: Do focus apps correlate with lower burnout (days/month)?",
        template=TEMPLATE,
        color="uses_focus_apps",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(showlegend=False, yaxis_title="Days feeling burnout / month", height=520)
    return fig


def plot_q5_age_platform_usage(df: pd.DataFrame) -> go.Figure:
    mean_pivot = df.pivot_table(
        index="age_group",
        columns="social_platform_preference",
        values="daily_social_media_time",
        aggfunc="mean",
    )
    count_pivot = df.pivot_table(
        index="age_group",
        columns="social_platform_preference",
        values="daily_social_media_time",
        aggfunc="size",
    ).fillna(0)

    # 100% normalized popularity share within each age group.
    share_pivot = count_pivot.div(
        count_pivot.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0) * 100

    # Platform on Y, age group on X for easier scanning.
    mean_t = mean_pivot.T
    share_t = share_pivot.T

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean time (hours)", "Platform popularity share (%)"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Heatmap(
            z=mean_t.values,
            x=mean_t.columns.tolist(),
            y=mean_t.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="hours"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=share_t.values,
            x=share_t.columns.tolist(),
            y=share_t.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="share %"),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Q5: Which social media platforms are most popular across different age groups, and how much time do people spend on them?",
        height=560,
    )
    return fig


def plot_q6_sleep_coffee_stress_3d(df: pd.DataFrame) -> go.Figure:
    d = _sample_df(df, n=20000).copy()
    d["coffee_bin"] = pd.qcut(
        d["coffee_consumption_per_day"],
        q=3,
        labels=["Low coffee", "Medium coffee", "High coffee"],
        duplicates="drop",
    )

    fig = px.scatter(
        d,
        x="sleep_hours",
        y="stress_level",
        facet_col="coffee_bin",
        facet_col_wrap=3,
        color="stress_level",
        color_continuous_scale="Viridis",
        opacity=0.45,
        template=TEMPLATE,
        title="Q6: How do our sleep habits and coffee consumption combine to drive up stress?",
    )
    fig.update_traces(marker=dict(size=5), showlegend=False)
    fig.update_layout(height=560)
    fig.for_each_annotation(
        lambda a: a.update(text=str(a.text).split("=")[-1].strip())
        if a.text is not None
        else None
    )
    fig.update_xaxes(title_text="Sleep hours")
    fig.update_yaxes(title_text="Stress level")
    return fig


def plot_q7_digital_wellbeing_offline(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="has_digital_wellbeing_enabled",
        y="weekly_offline_hours",
        box=True,
        points=False,
        color="has_digital_wellbeing_enabled",
        template=TEMPLATE,
        title="Q7: Does enabling Digital Wellbeing associate with more offline time?",
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(showlegend=False, yaxis_title="Weekly offline hours", height=520)
    return fig


def plot_q8_work_breaks_satisfaction(df: pd.DataFrame) -> go.Figure:
    d = _sample_df(df, n=8000).copy()
    cutoff = d["work_hours_per_day"].median()
    d["workload"] = np.where(
        d["work_hours_per_day"] >= cutoff, "High work hours", "Low work hours"
    )

    fig = px.scatter(
        d,
        x="breaks_during_work",
        y="job_satisfaction_score",
        color="workload",
        template=TEMPLATE,
        title="Q8: Breaks and satisfaction — does the effect depend on workload?",
        opacity=0.65,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_traces(marker=dict(size=7))

    for label, col in [("Low work hours", "royalblue"), ("High work hours", "darkorange")]:
        dd = d.loc[d["workload"] == label].sort_values("breaks_during_work")
        if len(dd) >= 300:
            x = dd["breaks_during_work"].to_numpy(dtype=float)
            y = dd["job_satisfaction_score"].to_numpy(dtype=float)
            sm = lowess(y, x, frac=0.25, return_sorted=True)
            fig.add_trace(
                go.Scatter(
                    x=sm[:, 0],
                    y=sm[:, 1],
                    mode="lines",
                    name=f"LOESS ({label})",
                    line=dict(color=col, width=3),
                )
            )

    fig.update_layout(
        xaxis_title="Breaks during work",
        yaxis_title="Job satisfaction score",
        legend_title_text="Workload group",
        height=540,
    )
    return fig


def plot_q9_gender_productivity(df: pd.DataFrame) -> go.Figure:
    d = _sample_df(df, n=7000).copy()
    if "perceived_minus_actual" not in d.columns:
        d["perceived_minus_actual"] = d["perceived_productivity_score"] - d["actual_productivity_score"]

    d["gap"] = d["perceived_minus_actual"].astype(float)

    xs = np.linspace(
        d["actual_productivity_score"].min(),
        d["actual_productivity_score"].max(),
        60,
    )
    x = d["actual_productivity_score"].astype(float).to_numpy()
    y = d["perceived_productivity_score"].astype(float).to_numpy()
    slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    intercept = y.mean() - slope * x.mean()
    ys = intercept + slope * xs

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Perception vs actual (scatter)", "Perceived-actual gap by gender"),
        horizontal_spacing=0.08,
    )

    for gender, gdf in d.groupby("gender"):
        fig.add_trace(
            go.Scatter(
                x=gdf["actual_productivity_score"],
                y=gdf["perceived_productivity_score"],
                mode="markers",
                name=str(gender),
                opacity=0.55,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines", name="Pooled linear fit", line=dict(color="black", width=3)
        ),
        row=1,
        col=1,
    )

    for gender, gdf in d.groupby("gender"):
        fig.add_trace(
            go.Violin(
                x=[str(gender)] * len(gdf),
                y=gdf["gap"],
                name=str(gender),
                box_visible=True,
                meanline_visible=True,
                opacity=0.75,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template=TEMPLATE,
        title="Q9: Is the perceived-vs-actual productivity bias different by gender?",
        height=580,
    )
    fig.update_xaxes(title_text="Actual productivity (scatter)", row=1, col=1)
    fig.update_yaxes(title_text="Perceived productivity (scatter)", row=1, col=1)
    fig.update_xaxes(title_text="Gender", row=1, col=2)
    fig.update_yaxes(title_text="Perceived minus actual (gap)", row=1, col=2)
    return fig


def plot_q10_job_type_habits_stress(df: pd.DataFrame) -> go.Figure:
    def _rho(g: pd.DataFrame) -> float:
        r, _ = stats.spearmanr(
            g["daily_social_media_time"].astype(float),
            g["stress_level"].astype(float),
            nan_policy="omit",
        )
        return float(r)

    base = (
        df.groupby("job_type")
        .agg(
            mean_social=("daily_social_media_time", "mean"),
            mean_stress=("stress_level", "mean"),
        )
        .reset_index()
    )
    rho_df = (
        df.groupby("job_type")
        .apply(_rho)
        .rename("rho_time_stress")
        .reset_index()
    )
    agg = base.merge(rho_df, on="job_type", how="left").sort_values(
        "mean_social", ascending=False
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Mean social media (h/day)",
            x=agg["job_type"].astype(str),
            y=agg["mean_social"],
            offsetgroup=0,
            marker_color="steelblue",
            customdata=agg["rho_time_stress"],
            hovertemplate="Job: %{x}<br>Mean time: %{y:.2f}h<br>ρ(time,stress): %{customdata:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Mean stress",
            x=agg["job_type"].astype(str),
            y=agg["mean_stress"],
            yaxis="y2",
            offsetgroup=1,
            marker_color="coral",
            customdata=agg["rho_time_stress"],
            hovertemplate="Job: %{x}<br>Mean stress: %{y:.2f}<br>ρ(time,stress): %{customdata:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Q10: Do job types differ in social-media load and associated stress?",
        template=TEMPLATE,
        xaxis=dict(title="Job type", tickangle=-30),
        yaxis=dict(title="Mean daily social media (hours)", side="left", showgrid=True),
        yaxis2=dict(title="Mean stress", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=540,
        barmode="group",
    )
    return fig


# Named map for Streamlit / notebook
QUESTION_PLOTS = {
    "Q1": plot_q1_3d_scatter,
    "Q2": plot_q2_notifications_stress,
    "Q3": plot_q3_platform_screen_time,
    "Q4": plot_q4_focus_apps_burnout,
    "Q5": plot_q5_age_platform_usage,
    "Q6": plot_q6_sleep_coffee_stress_3d,
    "Q7": plot_q7_digital_wellbeing_offline,
    "Q8": plot_q8_work_breaks_satisfaction,
    "Q9": plot_q9_gender_productivity,
    "Q10": plot_q10_job_type_habits_stress,
}
