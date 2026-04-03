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
    rho_rows: list[dict[str, object]] = []
    for jt, g in df.groupby("job_type"):
        rho_rows.append({"job_type": jt, "rho_time_stress": _rho(g)})
    rho_df = pd.DataFrame(rho_rows)
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


# --- Actionable-advice EDA (graphs + stats in notebook / Streamlit) ---

A1_TITLE = (
    "Unplugging: how do screen time before bed, caffeine, and sleep relate?"
)
A2_TITLE = "Do more weekly offline hours go with lower stress and fewer burnout days?"
A3_TITLE = "Which habits most strongly relate to the perceived–actual productivity gap?"
A4_TITLE = "Do focus-app users report higher actual productivity?"
A5_TITLE = "For long workdays (8+ h), how do breaks relate to burnout?"
A6_TITLE = "Job type vs stress & satisfaction after accounting for work hours"
A7_TITLE = "Where do notifications show the steepest climb in stress or drop in productivity?"


def plot_a1_sleep_screen_coffee(df: pd.DataFrame) -> go.Figure:
    cols = ["screen_time_before_sleep", "coffee_consumption_per_day", "sleep_hours"]
    d = df[cols].dropna()
    mat = np.zeros((3, 3))
    labels: list[str] = []
    for i, ci in enumerate(cols):
        labels.append(ci.replace("_", " "))
        for j, cj in enumerate(cols):
            r, _ = stats.spearmanr(d[ci], d[cj], nan_policy="omit")
            mat[i, j] = float(r)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Spearman correlation (pairwise)",
            "Scatter matrix (sample)",
        ),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="ρ"),
            text=np.round(mat, 3),
            texttemplate="%{z:.3f}",
            hovertemplate="%{y} vs %{x}<br>ρ=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    sm = _sample_df(d, n=4000)
    fig.add_trace(
        go.Scatter(
            x=sm["screen_time_before_sleep"],
            y=sm["sleep_hours"],
            mode="markers",
            marker=dict(size=5, opacity=0.35, color="steelblue"),
            name="Screen vs sleep",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template=TEMPLATE,
        title=A1_TITLE,
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Screen time before sleep (h)", row=1, col=2)
    fig.update_yaxes(title_text="Sleep hours", row=1, col=2)
    return fig


def plot_a2_offline_stress_burnout(df: pd.DataFrame) -> go.Figure:
    d = _sample_df(df, n=12000).copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Offline hours vs stress", "Offline hours vs burnout days"),
        horizontal_spacing=0.1,
    )
    for col, ycol, title_y in [
        (1, "stress_level", "Stress"),
        (2, "days_feeling_burnout_per_month", "Burnout days / mo"),
    ]:
        dd = d.sort_values("weekly_offline_hours")
        x = dd["weekly_offline_hours"].to_numpy(dtype=float)
        y = dd[ycol].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                opacity=0.35,
                marker=dict(size=5),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        if len(dd) >= 200:
            sm = lowess(y, x, frac=0.2, return_sorted=True)
            fig.add_trace(
                go.Scatter(
                    x=sm[:, 0],
                    y=sm[:, 1],
                    mode="lines",
                    line=dict(color="crimson", width=3),
                    name="LOESS",
                    showlegend=(col == 2),
                ),
                row=1,
                col=col,
            )
    fig.update_layout(template=TEMPLATE, title=A2_TITLE, height=480)
    fig.update_xaxes(title_text="Weekly offline hours", row=1, col=1)
    fig.update_xaxes(title_text="Weekly offline hours", row=1, col=2)
    fig.update_yaxes(title_text="Stress level", row=1, col=1)
    fig.update_yaxes(title_text="Burnout days / month", row=1, col=2)
    return fig


def plot_a3_gap_drivers(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    if "productivity_gap" not in d.columns:
        d["productivity_gap"] = (
            d["perceived_productivity_score"] - d["actual_productivity_score"]
        )
    habit_cols = [
        "daily_social_media_time",
        "number_of_notifications",
        "screen_time_before_sleep",
        "work_hours_per_day",
        "weekly_offline_hours",
        "breaks_during_work",
        "coffee_consumption_per_day",
    ]
    rows: list[tuple[str, float]] = []
    g = d["productivity_gap"].astype(float)
    for c in habit_cols:
        if c not in d.columns:
            continue
        r, _ = stats.spearmanr(d[c], g, nan_policy="omit")
        rows.append((c.replace("_", " "), float(r)))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    feat = [r[0] for r in rows]
    rho = [r[1] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=rho,
            y=feat,
            orientation="h",
            marker_color=["#c0392b" if x > 0 else "#2980b9" for x in rho],
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title=A3_TITLE,
        xaxis_title="Spearman ρ with productivity gap (perceived − actual)",
        yaxis_title="",
        height=440,
        margin=dict(l=160, r=40, t=60, b=50),
    )
    return fig


def plot_a4_focus_apps_actual_productivity(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="uses_focus_apps",
        y="actual_productivity_score",
        box=True,
        points=False,
        color="uses_focus_apps",
        template=TEMPLATE,
        title=A4_TITLE,
        color_discrete_sequence=COLOR_SEQ,
    )
    fig.update_layout(showlegend=False, height=480, yaxis_title="Actual productivity (0–10)")
    return fig


def plot_a5_breaks_burnout_long_hours(df: pd.DataFrame) -> go.Figure:
    d = df.loc[df["work_hours_per_day"].astype(float) >= 8].copy()
    d = _sample_df(d, n=min(12000, len(d)))
    d = d.sort_values("breaks_during_work")
    x = d["breaks_during_work"].to_numpy(dtype=float)
    y = d["days_feeling_burnout_per_month"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            opacity=0.35,
            marker=dict(size=6, color="slategray"),
            name="Observed",
        )
    )
    if len(d) >= 200:
        sm = lowess(y, x, frac=0.22, return_sorted=True)
        fig.add_trace(
            go.Scatter(
                x=sm[:, 0],
                y=sm[:, 1],
                mode="lines",
                line=dict(color="darkorange", width=3),
                name="LOESS",
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title=A5_TITLE,
        xaxis_title="Breaks during work (count)",
        yaxis_title="Burnout days / month",
        height=500,
    )
    return fig


def plot_a6_job_type_residualized(df: pd.DataFrame) -> go.Figure:
    wh = df["work_hours_per_day"].astype(float)
    stress = df["stress_level"].astype(float)
    sat = df["job_satisfaction_score"].astype(float)
    bs, bi = np.polyfit(wh, stress, 1)
    stress_resid = stress - (bs * wh + bi)
    bj, b0 = np.polyfit(wh, sat, 1)
    sat_resid = sat - (bj * wh + b0)
    plot_df = df.assign(stress_resid=stress_resid, sat_resid=sat_resid)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Stress (residual vs work hours)",
            "Job satisfaction (residual vs work hours)",
        ),
        horizontal_spacing=0.08,
    )
    for jt, g in plot_df.groupby("job_type"):
        jts = str(jt)
        fig.add_trace(
            go.Box(
                x=[jts] * len(g),
                y=g["stress_resid"],
                name=jts,
                boxpoints="outliers",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="Job type", row=1, col=1)
    fig.update_yaxes(title_text="Stress residual", row=1, col=1)

    for jt, g in plot_df.groupby("job_type"):
        jts = str(jt)
        fig.add_trace(
            go.Box(
                x=[jts] * len(g),
                y=g["sat_resid"],
                name=jts,
                boxpoints="outliers",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Job type", row=1, col=2)
    fig.update_yaxes(title_text="Satisfaction residual", row=1, col=2)
    fig.update_layout(
        template=TEMPLATE,
        title=A6_TITLE,
        height=520,
        boxmode="group",
    )
    return fig


def plot_a7_notification_threshold(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["notif_bin"] = pd.qcut(
        d["number_of_notifications"].astype(float),
        q=10,
        duplicates="drop",
    )
    agg = (
        d.groupby("notif_bin", observed=True)
        .agg(
            mean_notif=("number_of_notifications", "mean"),
            mean_stress=("stress_level", "mean"),
            mean_actual=("actual_productivity_score", "mean"),
            n=("number_of_notifications", "size"),
        )
        .reset_index()
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Mean stress by notification decile", "Mean actual productivity by decile"),
    )
    fig.add_trace(
        go.Scatter(
            x=agg["mean_notif"],
            y=agg["mean_stress"],
            mode="lines+markers",
            line=dict(color="coral", width=2),
            marker=dict(size=10),
            name="Stress",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=agg["mean_notif"],
            y=agg["mean_actual"],
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
            marker=dict(size=10),
            name="Actual productivity",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template=TEMPLATE,
        title=A7_TITLE,
        height=640,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Mean notifications (within decile)", row=2, col=1)
    fig.update_yaxes(title_text="Mean stress", row=1, col=1)
    fig.update_yaxes(title_text="Mean actual productivity", row=2, col=1)
    return fig


# Advice plots map (must appear immediately after A1–A7 so notebook extraction stops here)
ADVICE_PLOTS = {
    "A1": plot_a1_sleep_screen_coffee,
    "A2": plot_a2_offline_stress_burnout,
    "A3": plot_a3_gap_drivers,
    "A4": plot_a4_focus_apps_actual_productivity,
    "A5": plot_a5_breaks_burnout_long_hours,
    "A6": plot_a6_job_type_residualized,
    "A7": plot_a7_notification_threshold,
}

# Q1–Q10 maps for Streamlit (after advice block; not embedded in notebook self-contained advice cell)
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
