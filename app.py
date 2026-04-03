"""
Streamlit dashboard: Social Media vs Productivity EDA portfolio.
Uses pathlib for paths; caches cleaned data with @st.cache_data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import preprocessing
import visualizations as viz

st.set_page_config(
    page_title="Social Media vs Productivity EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_raw_cached() -> pd.DataFrame:
    return preprocessing.load_raw_data(preprocessing.DATA_PATH)


@st.cache_data(show_spinner=False)
def load_clean_cached() -> pd.DataFrame:
    raw = preprocessing.load_raw_data(preprocessing.DATA_PATH)
    return preprocessing.clean_dataframe(raw)


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Data Cleaning",
            "Interactive EDA",
            "Q&A Insights",
            "Summary",
        ],
    )

    try:
        df_raw = load_raw_cached()
        df = load_clean_cached()
    except FileNotFoundError as e:
        st.error(f"Data file missing: {e}. Run scripts/create_project_structure.py and place the CSV in data/.")
        return
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if page == "Overview":
        st.title("Social Media vs Productivity")
        st.markdown(
            """
This interactive dashboard summarizes **simulated behavioral data** (30,000 rows, 19 features)
linking digital habits, work context, sleep, stress, and productivity outcomes.

**Goal:** Explore relationships between social media use, notifications, wellbeing settings,
and both *perceived* and *actual* productivity — as a portfolio EDA piece with a shared
Python pipeline (`preprocessing.py`) and reusable Plotly figures (`visualizations.py`).
            """
        )
        st.subheader("Features")
        st.markdown(
            """
| Area | Columns |
|------|---------|
| Demographics | `age`, `gender`, `job_type`, `age_group` (engineered) |
| Digital | `daily_social_media_time`, `social_platform_preference`, `number_of_notifications`, `screen_time_before_sleep`, `uses_focus_apps`, `has_digital_wellbeing_enabled`, `weekly_offline_hours` |
| Work | `work_hours_per_day`, `breaks_during_work` |
| Outcomes | `perceived_productivity_score`, `actual_productivity_score`, `perceived_minus_actual`, `stress_level`, `sleep_hours`, `coffee_consumption_per_day`, `days_feeling_burnout_per_month`, `job_satisfaction_score` |
            """
        )

    elif page == "Data Cleaning":
        st.title("Data cleaning & preprocessing")
        st.markdown(
            """
- **Booleans:** string `"True"` / `"False"` parsed to real booleans; missing filled with mode.
- **Imputation:** this app uses a fast, shared **median-based** pipeline in `preprocessing.py` for responsiveness, while the notebook also demonstrates a more advanced **MICE (IterativeImputer)** approach for smoother, more natural numeric distributions.
- **IQR winsorization:** `daily_social_media_time`, `coffee_consumption_per_day`, `number_of_notifications`.
- **Features:** `age_group` bins, `perceived_minus_actual` gap.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Raw (sample)")
            st.dataframe(df_raw.head(12), width="stretch")
        with c2:
            st.caption("Cleaned (sample)")
            st.dataframe(df.head(12), width="stretch")

    elif page == "Interactive EDA":
        st.title("Interactive EDA")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ("age_group",)]

        all_axes = num_cols + cat_cols

        # One-click preset examples that set up sensible explorations.
        presets = {
            "(Custom)": {"x": None, "y": None, "z": None, "color": None},
            "Time vs stress (3D, workload color)": {
                "x": "daily_social_media_time",
                "y": "stress_level",
                "z": "work_hours_per_day",
                "color": "perceived_productivity_score",
            },
            "Sleep vs stress by job satisfaction": {
                "x": "sleep_hours",
                "y": "stress_level",
                "z": None,
                "color": "job_satisfaction_score",
            },
            "Breaks vs satisfaction (color=work_hours)": {
                "x": "breaks_during_work",
                "y": "job_satisfaction_score",
                "z": None,
                "color": "work_hours_per_day",
            },
        }

        preset_name = st.selectbox(
            "Quick examples (one click)",
            list(presets.keys()),
            index=0,
            help="Choose a preset to instantly configure a meaningful view, or select '(Custom)' to pick axes yourself.",
        )

        def _idx(name: str, fallback_list: list[str], default: int = 0) -> int:
            try:
                return fallback_list.index(name)
            except ValueError:
                return default

        if preset_name == "(Custom)":
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox(
                    "X axis",
                    all_axes,
                    index=_idx("daily_social_media_time", all_axes),
                )
            with c2:
                y_axis = st.selectbox(
                    "Y axis",
                    all_axes,
                    index=_idx("stress_level", all_axes),
                )
            with c3:
                z_axis = st.selectbox(
                    "Z axis (optional, 3D)",
                    ["(none)"] + num_cols,
                    index=0,
                )

            color_opt = st.selectbox(
                "Color (optional)",
                ["(none)"] + [c for c in num_cols + cat_cols if c not in (x_axis, y_axis)],
                index=0,
            )
            color = None if color_opt == "(none)" else color_opt
        else:
            cfg = presets[preset_name]
            x_axis = cfg["x"]
            y_axis = cfg["y"]
            z_axis = cfg["z"] if cfg["z"] is not None else "(none)"
            color = cfg["color"]

            st.caption(
                f"Preset '{preset_name}' — X: `{x_axis}`, Y: `{y_axis}`, "
                f"{'Z: `' + cfg['z'] + '`, ' if cfg['z'] else ''}"
                f"Color: `{color}`"
            )

        # Build the figure based on chosen or preset axes.
        if z_axis == "(none)":
            if x_axis in cat_cols or y_axis in cat_cols:
                fig = px.box(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color=color,
                    template="plotly_white",
                )
            else:
                fig = px.scatter(
                    df.sample(min(8000, len(df)), random_state=42),
                    x=x_axis,
                    y=y_axis,
                    color=color,
                    template="plotly_white",
                    opacity=0.5,
                )
            st.plotly_chart(fig, width="stretch")
        else:
            fig = px.scatter_3d(
                df.sample(min(8000, len(df)), random_state=42),
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color,
                template="plotly_white",
            )
            st.plotly_chart(fig, width="stretch")

    elif page == "Q&A Insights":
        st.title("Ten EDA questions")
        qa_text = {
            "Q1": (
                "The 2D density plot shows where most users cluster in (social media time, productivity gap) space; "
                "the red line overlays the median perceived-minus-actual gap by time quartile. If that line drifts "
                "away from zero at higher usage levels, it suggests heavier users are increasingly miscalibrated "
                "about how productive they really are."
            ),
            "Q2": (
                "This view plots notifications vs stress, colored by social-media-time quartiles, with a LOESS line "
                "summarizing the trend. Stress generally rises with notifications, and the slope is often steeper "
                "for heavy users, supporting the idea that managing alerts is a practical lever for reducing stress."
            ),
            "Q3": (
                "Platform-level violins compare the distributions of screen time before sleep. Platforms with higher "
                "medians and fatter upper tails are more likely to keep users up late, making them prime candidates "
                "for platform-specific sleep-hygiene advice."
            ),
            "Q4": (
                "Burnout-day distributions for focus-app users vs non-users let you see whether these tools are "
                "associated with real-world relief. Slightly lower medians and tighter violins for focus-app users "
                "suggest they may form one useful component of a broader burnout-management strategy."
            ),
            "Q5": (
                "The left heatmap shows mean daily time by age group and platform, while the right is a 100% "
                "normalized popularity view (share of each platform within age group). Together, they reveal which "
                "platforms dominate within each cohort and where high time and high share coincide."
            ),
            "Q6": (
                "Faceted scatter plots show sleep vs stress in Low, Medium, and High coffee panels. Across panels, "
                "less sleep generally maps to higher stress, but the pattern is often steeper in the high-coffee "
                "facet, hinting that heavy caffeine plus short sleep is a particularly stressful combination."
            ),
            "Q7": (
                "Violin plots compare weekly offline hours for users with Digital Wellbeing enabled versus disabled. "
                "A modest shift toward higher offline time for the enabled group, with substantial overlap, suggests "
                "these controls help mainly when users are already motivated to set boundaries."
            ),
            "Q8": (
                "Workload-stratified scatter (plus LOESS lines) shows how breaks relate to job satisfaction for low "
                "vs high work hours. Breaks appear especially protective under high workload, where moving from few "
                "to moderate breaks is often associated with a clear lift in satisfaction."
            ),
            "Q9": (
                "The left panel plots perceived vs actual productivity by gender, while the right panel shows the "
                "distribution of the perceived-minus-actual gap. Small shifts in these distributions point to subtle "
                "group-level tendencies toward over- or under-estimation in self-assessed productivity."
            ),
            "Q10": (
                "Dual-axis bars compare mean daily social-media time and mean stress by job type, with hover text "
                "exposing within-job ρ(time, stress). This helps distinguish roles where high engagement is part of "
                "the job but not especially toxic from those where high time and high stress clearly travel together."
            ),
        }
        for key in [f"Q{i}" for i in range(1, 11)]:
            with st.expander(f"{key}: insight", expanded=(key == "Q1")):
                st.markdown(qa_text[key])
                fn = viz.QUESTION_PLOTS[key]
                fig = fn(df)
                st.plotly_chart(fig, width="stretch")

    else:
        st.title("Summary & future work")
        st.markdown(
            """
### Key takeaways
- **Multicollinearity:** perceived and actual productivity are often highly correlated; check variance inflation before regression.
- **Digital habits:** notifications and social time relate to stress and sleep-related variables — see correlation heatmap in the notebook.
- **Calibration gap:** `perceived_minus_actual` is a useful target feature for “over/under-confidence” in productivity.

### Next steps for ML
- **Regression:** predict `stress_level`, `job_satisfaction_score`, or `days_feeling_burnout_per_month` from digital + work features; use regularization (Ridge/Lasso) if collinearity persists.
- **Classification:** bin burnout or stress for stratified models; report ROC/PR and calibration.
- **Validation:** train/test split, cross-validation, and inspect residuals vs key digital variables.
            """
        )


if __name__ == "__main__":
    main()
