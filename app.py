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
        st.subheader("Why we use median imputation")
        st.markdown(
            """
Deleting rows with missing values would throw away too much valuable data and could bias our results if certain groups skipped questions. We also cannot simply drop the affected columns, because they contain the core outcomes we want to analyze, such as stress and productivity.

By filling in the missing values instead, we maintain our full dataset of **30,000 rows** and prevent errors in our visualizations and statistical tests. We specifically use the **median** to fill these gaps because it is robust against extreme outliers and safer for skewed survey data than the standard average.

Ultimately, this straightforward approach ensures our exploratory data analysis remains **consistent**, **fast**, and **reliable** without sacrificing important information.
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
                "**What this graph means:** It compares the gap between *felt* and *measured* productivity across four levels of daily social media time (low to high). "
                "If the boxes drift upward to the right, people who use social media more also tend to rate themselves as more productive than the data suggests. "
                "Treat it as a warning sign about self-perception, not proof that social media caused the gap."
            ),
            "Q2": (
                "**What this graph means:** Each point is a group of people who get a similar number of notifications; the height is their average stress. "
                "If the line climbs from left to right, heavier notification load goes with higher stress on average—useful when deciding whether to reduce pings. "
                "It describes a trend in the data, not a rule for every individual."
            ),
            "Q3": (
                "**What this graph means:** Longer bars are platforms where users *typically* spend more time on a screen before bed. "
                "That highlights which apps are most tied to late-night scrolling when you talk about sleep—not who is “good” or “bad.”"
            ),
            "Q4": (
                "**What this graph means:** It compares how many heavy-burnout days people report, with vs without focus apps. "
                "If the focus-app side is lower, those users report fewer bad days in this dataset. That is an association only; it does not prove the app caused the difference."
            ),
            "Q5": (
                "**What this graph means:** The left panel is average hours on each platform by age group. The right panel is each platform’s *share* of users inside that age group (each age row adds to 100%). "
                "Together they show where each generation spends time and which apps dominate that age band."
            ),
            "Q6": (
                "**What this graph means:** Every dot is one person: sleep across the bottom, stress up the side, and color shows coffee intake. "
                "You will usually see more stress toward the left (less sleep). Color shows whether heavy coffee lines up with that short-sleep, high-stress corner."
            ),
            "Q7": (
                "**What this graph means:** The taller bar is the group that reports more *offline hours per week* on average—Wellbeing off vs on. "
                "Use it to see whether turning wellbeing tools on goes with spending more real time away from screens in this data."
            ),
            "Q8": (
                "**What this graph means:** Two lines show how average job satisfaction changes as breaks increase—blue for shorter workdays, orange for longer workdays. "
                "If the orange line rises more as breaks go up, taking breaks may matter most when the workday is already long."
            ),
            "Q9": (
                "**What this graph means:** Each box is the gap between *felt* and *measured* productivity for one gender. "
                "A higher box means that group more often thinks they are more productive than the score suggests—compare the middle of each box, not the stray dots."
            ),
            "Q10": (
                "**What this graph means:** For each job, blue is average social media time and orange is average stress; hover shows how strongly time and stress move together *within* that job. "
                "Use it to spot jobs where long social time is common but stress stays moderate, versus jobs where both run high together."
            ),
        }
        for key in [f"Q{i}" for i in range(1, 11)]:
            with st.expander(f"{key}: what this graph means", expanded=(key == "Q1")):
                st.markdown(qa_text[key])
                fn = viz.QUESTION_PLOTS[key]
                fig = fn(df)
                st.plotly_chart(fig, width="stretch")

    elif page == "Actionable advice":
        st.title("Actionable EDA — advice-oriented questions")
        st.markdown(
            """
Seven practical views (sleep, unplugged time, productivity gap, focus apps, long workdays, job type, notifications).
Below, **what this graph means** is written in simple English—the notebook uses the same wording after each chart.
            """
        )
        advice_text = {
            "A1": (
                "**What this graph means:** The small grid shows how evening screen time, coffee, and sleep move together; the scatter plots screen time against sleep. "
                "If late screens and short sleep show up together, a reasonable takeaway is to try less phone before bed and to notice caffeine—patterns in simulated data, not medical advice."
            ),
            "A2": (
                "**What this graph means:** The lines show whether people who spend more hours offline also report lower stress and fewer burnout days. "
                "If the lines slope downward, more unplugged time goes with calmer outcomes on average in this dataset."
            ),
            "A3": (
                "**What this graph means:** Longer bars mark habits that line up most strongly with the gap between *felt* and *measured* productivity. "
                "Use it to see what to discuss first when someone feels productive but the numbers disagree—it shows association, not proof one habit caused the gap."
            ),
            "A4": (
                "**What this graph means:** The shapes compare *actual* productivity scores for people who use focus apps vs those who do not. "
                "If one sits higher, that group scores better on average. The statistic under the chart only says whether the gap is likely noise; it does not prove the app works for everyone."
            ),
            "A5": (
                "**What this graph means:** Only people working 8+ hours: burnout days versus number of breaks, with a trend line. "
                "If the line slopes down, more breaks go with fewer bad burnout days among long workdays—a useful nudge for breaks, not a health diagnosis."
            ),
            "A6": (
                "**What this graph means:** After a simple adjustment for how long people work, the boxes show extra stress and job satisfaction by industry. "
                "Some jobs look more stressful than their hours alone would suggest; others look happier—helpful when comparing roles, not individuals."
            ),
            "A7": (
                "**What this graph means:** The lines show average stress and average *real* productivity across notification bands. "
                "Where the slope is steepest, cutting notifications might help the most on average; it is still a population pattern, not one cutoff for every person."
            ),
        }
        for key in [f"A{i}" for i in range(1, 8)]:
            with st.expander(f"{key}: what this graph means", expanded=(key == "A1")):
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
