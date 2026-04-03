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
    fig = px.histogram(
        df,
        x=column,
        nbins=40,
        title=ttl,
        template=TEMPLATE,
        color_discrete_sequence=["#1d4ed8"],
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color="white", opacity=0.92)
    fig.update_layout(
        bargap=0.06,
        xaxis_title=str(column).replace("_", " "),
        yaxis_title="Count",
        showlegend=False,
    )
    return fig


def plot_univariate_bar_categorical(df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    counts = df[column].astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]
    ttl = title or f"Counts: {column}"
    fig = px.bar(counts, x=column, y="count", title=ttl, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)
    fig.update_layout(xaxis_tickangle=-35)
    return fig


def plot_q1_3d_scatter(df: pd.DataFrame) -> go.Figure:
    """Box plot: gap by social-media time quartile (readable vs 2D density)."""
    d = df.copy()
    if "time_quartile" not in d.columns:
        d["time_quartile"] = pd.qcut(
            d["daily_social_media_time"],
            q=4,
            labels=["Low", "Mid-Low", "Mid-High", "High"],
            duplicates="drop",
        )
    fig = px.box(
        d,
        x="time_quartile",
        y="perceived_minus_actual",
        title="Q1: Does spending more time on social media widen the gap between how productive we feel and how productive we actually are?",
        template=TEMPLATE,
        color="time_quartile",
        color_discrete_sequence=COLOR_SEQ,
        category_orders={"time_quartile": ["Low", "Mid-Low", "Mid-High", "High"]},
        points="suspectedoutliers",
    )
    fig.update_layout(
        xaxis_title="Daily social media time (quartile)",
        yaxis_title="Perceived − actual productivity",
        showlegend=False,
        height=560,
    )
    return fig


def plot_q2_notifications_stress(df: pd.DataFrame) -> go.Figure:
    """Mean stress vs mean notifications by decile (simple trend)."""
    d = df[["number_of_notifications", "stress_level"]].dropna()
    d["bin"] = pd.qcut(d["number_of_notifications"], q=10, duplicates="drop")
    agg = (
        d.groupby("bin", observed=True)
        .agg(
            mean_notif=("number_of_notifications", "mean"),
            mean_stress=("stress_level", "mean"),
        )
        .reset_index()
    )
    fig = px.line(
        agg,
        x="mean_notif",
        y="mean_stress",
        markers=True,
        title="Q2: How do constant phone notifications impact our daily stress levels?",
        template=TEMPLATE,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(
        xaxis_title="Mean notifications (decile group)",
        yaxis_title="Mean stress (1–10)",
        showlegend=False,
        height=520,
    )
    return fig


def plot_q3_platform_screen_time(df: pd.DataFrame) -> go.Figure:
    med = (
        df.groupby("social_platform_preference", observed=True)["screen_time_before_sleep"]
        .median()
        .sort_values(ascending=True)
    )
    fig = px.bar(
        x=med.values,
        y=med.index.astype(str),
        orientation="h",
        title="Q3: Which platform is associated with later bedtime (screen time before sleep)?",
        template=TEMPLATE,
        text=med.values.round(2),
    )
    fig.update_traces(marker_color="#1d4ed8")
    fig.update_layout(
        xaxis_title="Median hours before sleep",
        yaxis_title="Platform",
        height=520,
    )
    return fig


def plot_q4_focus_apps_burnout(df: pd.DataFrame) -> go.Figure:
    d = df.assign(
        focus_label=df["uses_focus_apps"].map({True: "Uses focus apps", False: "No focus apps"})
    )
    fig = px.box(
        d,
        x="focus_label",
        y="days_feeling_burnout_per_month",
        title="Q4: Do focus apps correlate with lower burnout (days/month)?",
        template=TEMPLATE,
        color="focus_label",
        color_discrete_sequence=["#94a3b8", "#2563eb"],
        points="suspectedoutliers",
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title="Days feeling burnout / month",
        height=520,
        xaxis_title="",
    )
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
    d = _sample_df(df, n=8000)
    fig = px.scatter(
        d,
        x="sleep_hours",
        y="stress_level",
        color="coffee_consumption_per_day",
        color_continuous_scale="Viridis",
        title="Q6: How do our sleep habits and coffee consumption combine to drive up stress?",
        template=TEMPLATE,
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(
        xaxis_title="Sleep hours",
        yaxis_title="Stress level (1–10)",
        coloraxis_colorbar=dict(title="Coffee / day"),
        height=560,
    )
    return fig


def plot_q7_digital_wellbeing_offline(df: pd.DataFrame) -> go.Figure:
    med = df.groupby("has_digital_wellbeing_enabled")["weekly_offline_hours"].median()
    labels = ["Digital Wellbeing OFF", "Digital Wellbeing ON"]
    fig = px.bar(
        x=labels,
        y=[float(med[False]), float(med[True])],
        title="Q7: Does enabling Digital Wellbeing associate with more offline time?",
        template=TEMPLATE,
        color=labels,
        color_discrete_sequence=["#94a3b8", "#059669"],
        text=[round(float(med[False]), 2), round(float(med[True]), 2)],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Median offline hours / week",
        showlegend=False,
        height=480,
    )
    return fig


def plot_q8_work_breaks_satisfaction(df: pd.DataFrame) -> go.Figure:
    cutoff = df["work_hours_per_day"].median()
    low_work = df.loc[df["work_hours_per_day"] < cutoff]
    high_work = df.loc[df["work_hours_per_day"] >= cutoff]

    def _binned(sub: pd.DataFrame, q: int = 10) -> pd.DataFrame:
        d = sub[["breaks_during_work", "job_satisfaction_score"]].dropna()
        d["_b"] = pd.qcut(d["breaks_during_work"], q=q, duplicates="drop")
        return (
            d.groupby("_b", observed=True)
            .agg(
                breaks_during_work=("breaks_during_work", "mean"),
                job_satisfaction_score=("job_satisfaction_score", "mean"),
            )
            .reset_index()
        )

    c_low = _binned(low_work)
    c_high = _binned(high_work)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=c_low["breaks_during_work"],
            y=c_low["job_satisfaction_score"],
            mode="lines+markers",
            name="Below median work hours",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=c_high["breaks_during_work"],
            y=c_high["job_satisfaction_score"],
            mode="lines+markers",
            name="At/above median work hours",
            line=dict(color="#ea580c", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Q8: Breaks and satisfaction — does the effect depend on workload?",
        xaxis_title="Mean breaks in bin",
        yaxis_title="Mean job satisfaction",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_q9_gender_productivity(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="gender",
        y="perceived_minus_actual",
        title="Q9: Is the perceived-vs-actual productivity bias different by gender?",
        template=TEMPLATE,
        color="gender",
        color_discrete_sequence=COLOR_SEQ,
        points="suspectedoutliers",
    )
    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Perceived − actual gap",
        showlegend=False,
        height=520,
    )
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
