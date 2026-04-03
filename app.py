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
from scipy import stats as scipy_stats

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

# Header GIF (Giphy) — shown at top of Overview
HEADER_GIF_URL = (
    "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYnNtdThtODV0NWEwZHM2bXA4MW5nN2NvMmRybXN5b3I4eGl1dWd1OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/UiUcePrcsgS5Qp0DUj/giphy.gif"
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
            "Actionable advice",
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
        _c1, _c2, _c3 = st.columns([1, 2, 1])
        with _c2:
            st.image(HEADER_GIF_URL, width=480)
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
        st.subheader("🧾 Columns & Feature Info")
        st.markdown(
            """
| Column Name | Description |
|-------------|-------------|
| `age` | Age of the individual (18–65 years) |
| `gender` | Gender identity: Male, Female, or Other |
| `job_type` | Employment sector or status (IT, Education, Student, etc.) |
| `daily_social_media_time` | Average daily time spent on social media (hours) |
| `social_platform_preference` | Most-used social platform (Instagram, TikTok, Telegram, etc.) |
| `number_of_notifications` | Number of mobile/social notifications per day |
| `work_hours_per_day` | Average hours worked each day |
| `perceived_productivity_score` | Self-rated productivity score (scale: 0–10) |
| `actual_productivity_score` | Simulated ground-truth productivity score (scale: 0–10) |
| `stress_level` | Current stress level (scale: 1–10) |
| `sleep_hours` | Average hours of sleep per night |
| `screen_time_before_sleep` | Time spent on screens before sleeping (hours) |
| `breaks_during_work` | Number of breaks taken during work hours |
| `uses_focus_apps` | Whether the user uses digital focus apps (True/False) |
| `has_digital_wellbeing_enabled` | Whether Digital Wellbeing is activated (True/False) |
| `coffee_consumption_per_day` | Number of coffee cups consumed per day |
| `days_feeling_burnout_per_month` | Number of burnout days reported per month |
| `weekly_offline_hours` | Total hours spent offline each week (excluding sleep) |
| `job_satisfaction_score` | Satisfaction with job/life responsibilities (scale: 0–10) |
            """
        )
        st.subheader("Feature groups (quick reference)")
        st.markdown(
            """
| Area | Columns |
|------|---------|
| Demographics | `age`, `gender`, `job_type`, `age_group` (engineered) |
| Digital | `daily_social_media_time`, `social_platform_preference`, `number_of_notifications`, `screen_time_before_sleep`, `uses_focus_apps`, `has_digital_wellbeing_enabled`, `weekly_offline_hours` |
| Advice EDA | Seven plots (A1–A7) on sleep/unplugging, offline time, productivity gap drivers, focus apps, breaks under long hours, job-type residuals, notification deciles — see **Actionable advice** |
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

    elif page == "Actionable advice":
        st.title("Actionable EDA — advice-oriented questions")
        st.markdown(
            """
These seven views mirror the Cursor brief: **unplugging**, **offline time**, **productivity gap drivers**,
**focus apps**, **breaks on long days**, **job type (controlling work hours)**, and **notification thresholds**.
Correlations are **Spearman** (robust to curvature); group comparisons use **Mann–Whitney** or **Kruskal–Wallis** where noted.
            """
        )
        advice_text = {
            "A1": (
                "**Unplugging:** The heatmap summarizes how *screen time before sleep*, *coffee*, and *sleep hours* move together. "
                "Strong negative ρ between screen time and sleep supports a concrete rule: less pre-bed screen → more sleep in this dataset. "
                "Use the scatter for visual curvature; pair with cut-off experiments in real life (e.g. screen curfew + caffeine cap)."
            ),
            "A2": (
                "**Offline hours:** LOESS lines show whether more weekly offline time tracks with lower stress and fewer burnout days. "
                "If both slopes are negative, offline time is a candidate lever; magnitude (not just sign) matters for “how many hours” advice."
            ),
            "A3": (
                "**Productivity gap:** Bars rank |Spearman ρ| between the gap *(perceived − actual)* and habit variables. "
                "Larger |ρ| means a stronger monotonic link — useful for warning which distractions coincide with *miscalibration*, not causation."
            ),
            "A4": (
                "**Focus apps:** Violins compare *actual* productivity for users with vs without focus apps. "
                "Pair the plot with a Mann–Whitney test in the notebook; small median shifts need large samples — interpret as association, not proof of efficacy."
            ),
            "A5": (
                "**Long workdays (≥8 h):** Scatter + LOESS relate *breaks* to *burnout days*. "
                "A downward LOESS suggests more breaks associate with fewer burnout days among long-hour workers — useful for “minimum breaks” messaging."
            ),
            "A6": (
                "**Job type (holding hours):** Stress and satisfaction are **residualized** vs `work_hours_per_day` (linear control), then compared by `job_type`. "
                "Boxes show which industries sit above/below the stress or satisfaction expected from hours alone."
            ),
            "A7": (
                "**Notification threshold:** Decile curves show where mean stress climbs and mean *actual* productivity falls as notifications increase. "
                "Look for the steepest segment — that band is a practical “watch zone” for alert budgets (still associative, not causal)."
            ),
        }
        for key in [f"A{i}" for i in range(1, 8)]:
            with st.expander(f"{key}: insight", expanded=(key == "A1")):
                st.markdown(advice_text[key])
                fn = viz.ADVICE_PLOTS[key]
                fig = fn(df)
                st.plotly_chart(fig, width="stretch")
                if key == "A4":
                    u = df.loc[df["uses_focus_apps"], "actual_productivity_score"]
                    v = df.loc[~df["uses_focus_apps"], "actual_productivity_score"]
                    stat, p = scipy_stats.mannwhitneyu(u, v, alternative="two-sided")
                    st.caption(f"Mann–Whitney (actual productivity: focus apps True vs False): U={stat:.0f}, p={p:.2e}")

    else:
        st.title("Summary")
        st.markdown(
            """
## Project Summary: Social Media vs Productivity

---

### 1. What Was Built

| Deliverable | Role & Description |
| :---------- | :----------------- |
| **`data/social_media_vs_productivity.csv`** | 30,000 rows × 19 columns (simulated behavioral / survey-style data). |
| **`preprocessing.py`** | Shared pipeline for the Streamlit app: loads CSV, parses booleans, median imputes numerics, applies IQR winsorization on selected columns, and engineers `age_group` and `perceived_minus_actual`. |
| **`visualizations.py`** | All Plotly figures for Q1–Q10, A1–A7, plus helpers (correlation heatmap, etc.). |
| **`app.py`** | Streamlit dashboard: Overview, Data Cleaning, Interactive EDA, Q&A Insights, Actionable advice, Summary. |
| **`notebooks/eda_analysis.ipynb`** | Self-contained notebook: same-style cleaning, plots, stats, HTML exports under `output/graphs/`. |
| **`scripts/generate_notebook_selfcontained_v3.py`** | Regenerates the notebook from `visualizations.py` snippets and structured cells. |
| **Docker** | `Dockerfile`, `docker-compose.yml` (optional `docker-compose.app-only.yml`), `DOCKER_DEPLOY.md`, `.streamlit/config.toml`. |
| **Tech Stack** | Python, pandas, numpy, scipy, plotly, statsmodels (LOESS), streamlit, sklearn (MICE in notebook where used). |

---

### 2. Data & Preprocessing (High Level)

- **Dataset:** 30,000 rows, 19 columns.
- **Data Cleaning:** Boolean columns cast as True/False; missing numerics filled via median (or MICE in notebook); IQR winsorization applied to limit extreme synthetic outliers (e.g., social time, notifications, coffee).
- **Feature Engineering:** Created `perceived_minus_actual` (referred to as `productivity_gap` in advice views) and `age_group`.
- **Important Note:** Outcomes are simulated. This work serves as an EDA and storytelling exercise, not as established medical or HR facts.

---

### 3. Notebook Structure (Typical Flow)

1. **Setup:** Define paths, load CSV, inspect shape and missingness.
2. **Cleaning & Engineering:** Handle missing values, outliers, and create new features.
3. **Correlation Analysis:** Generate a correlation heatmap for numeric features.
4. **Section 4 (Q1–Q10):** Portfolio questions with plots, statistical tests (Spearman, Mann–Whitney, Kruskal–Wallis), and HTML exports.
5. **Section 5 (A1–A7):** Advice-oriented plots and statistical analysis.
6. **Section 6 (Summary):** Consolidation of themes, limitations, and optional extensions.

---

### 4. Streamlit App Structure

- **Overview:** GIF, introduction, full column table, grouped feature reference.
- **Data Cleaning:** Explanation of the app’s median pipeline versus notebook options.
- **Interactive EDA:** User-chosen axes and presets for custom exploration.
- **Q&A Insights:** Expanders for Q1–Q10 containing text and charts.
- **Actionable Advice:** Expanders for A1–A7; includes statistical captions (e.g., Mann–Whitney for focus app impact).
- **Summary:** This page — consolidated project documentation, themes, limitations, and extensions.

---

### 5. Questions and How to Read the "Answers"

#### Part A — Ten Portfolio Questions (Q1–Q10)

| ID | Question | Visualization | How to read the "answer" |
| :-- | :--------- | :------------- | :------------------------- |
| **Q1** | Does spending more time on social media widen the productivity gap? | 2D density of social time vs gap + median gap by quartile. | If median gap rises in higher time quartiles, heavier use aligns with greater miscalibration. |
| **Q2** | How do constant phone notifications impact daily stress levels? | Scatter: notifications vs stress, colored by social time; LOESS trend. | Stress tends to increase with notifications; steeper slopes for heavier social users indicate alerts act as a stress lever. |
| **Q3** | Which platform is associated with later bedtimes? | Violin: `screen_time_before_sleep` by platform. | Higher medians/upper tails indicate more late-evening screen use, highlighting sleep-hygiene priorities. |
| **Q4** | Do focus apps correlate with lower burnout? | Violin: burnout days by `uses_focus_apps`. | A lower median for focus app users suggests an association with fewer burnout days (not proof of causation). |
| **Q5** | Which platforms are most popular across age groups? | Heatmaps: mean time and share within age group. | Shows where each cohort spends time and which platforms dominate specific age bands. |
| **Q6** | How do sleep habits and coffee consumption combine to drive up stress? | Faceted scatter: sleep vs stress by coffee consumption tier. | "Short sleep + high coffee" often visualizes as a risky combination for elevated stress levels. |
| **Q7** | Does enabling Digital Wellbeing associate with more offline time? | Violin: `weekly_offline_hours` by `has_digital_wellbeing_enabled`. | An upward shift indicates tools align with more offline time, while overlap shows it is not universal. |
| **Q8** | Breaks and satisfaction — does the effect depend on workload? | Scatter: breaks vs job satisfaction, split by low/high work hours. | Breaks link to higher satisfaction, especially under high workloads (breaks act as protection). |
| **Q9** | Is the productivity bias different by gender? | Left: perceived vs actual by gender; Right: gap distribution by gender. | Small shifts indicate subtle over/under-estimation differences across distributions. |
| **Q10** | Do job types differ in social-media load and associated stress? | Grouped dual-axis bars: mean social time & stress by `job_type`. | Separates jobs where high social time is "normal" from those where time and stress rise together. |

#### Part B — Seven Advice-Oriented Analyses (A1–A7)

| ID | Question | Visualization | How to read the "answer" |
| :-- | :--------- | :------------- | :------------------------- |
| **A1** | Unplugging: how do screen time, coffee, and sleep relate? | Spearman 3×3 heatmap + scatter (screen vs sleep). | Negative screen–sleep correlation supports limiting late screens; guides cut-off style advice. |
| **A2** | Do more weekly offline hours go with lower stress and burnout? | Two panels: offline vs stress, offline vs burnout; LOESS. | Negative slopes suggest offline time as a lever; steepness informs how much offline time matters. |
| **A3** | Which habits most strongly relate to the productivity gap? | Horizontal bar chart: Spearman correlation (gap vs various habits). | Larger \|ρ\| indicates stronger monotonic association with the gap (which habits co-move with miscalibration). |
| **A4** | Do focus-app users report higher actual productivity? | Violin: actual productivity by `uses_focus_apps`. | Group difference in a snapshot; pair with Mann–Whitney — association, not proof of efficacy. |
| **A5** | For workdays ≥ 8h, how do breaks relate to burnout? | Scatter + LOESS: breaks vs burnout days. | Downward trend suggests more breaks are associated with fewer burnout days for long-hour workers. |
| **A6** | Job type vs stress & satisfaction (accounting for hours) | Linear residuals of stress/satisfaction on work hours; boxplots by job. | Shows which jobs sit above/below expected baseline stress/satisfaction, providing industry context. |
| **A7** | Where do notifications show the steepest climb in stress? | Deciles: mean stress & productivity vs mean notifications in bin. | The steepest segment indicates a "watch zone" for notification loads and alert budgets. |

---

### 6. Cross-Cutting Conclusions

- **Calibration:** The `perceived_minus_actual` metric serves as the primary lens for comparing self-assessment against simulated truth.
- **Digital Load:** Notifications and evening screen time consistently co-move with stress and sleep patterns across the visualizations.
- **Possible Levers (Hypotheses):** Offline time, structured breaks (especially on long workdays), and strict notification hygiene.
- **Heterogeneity:** Job, age, platform, and gender splits matter significantly. There is no single global rule that applies to all demographics.
- **Limitations:** The data is simulated. Correlations, LOESS, and rank tests describe patterns, but causality requires experimental design or longitudinal data.

---

### 7. Optional Next Steps

- **Qualitative Follow-up:** Conduct user interviews or surveys to add context to the behavioral data.
- **Field Trials:** Implement and measure specific interventions, such as notification batching or evening screen curfews.
- **Modeling:** If predictive models are added, ensure proper data splits, cross-validation, regularization, and calibration—especially given the inherent correlation between perceived and actual productivity.
            """
        )


if __name__ == "__main__":
    main()
