"""
OLS Regression Diagnostic Dashboard
=====================================
An interactive Streamlit dashboard for diagnosing heteroscedasticity
and multicollinearity in OLS regression models using statsmodels.

HOW TO RUN:
-----------
1. Install dependencies:
   pip install streamlit plotly statsmodels pandas numpy

2. Launch the app from your terminal:
   streamlit run ols_diagnostic_dashboard.py

3. The app will open in your browser at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import io

# =============================================================================
# PAGE CONFIGURATION
# st.set_page_config() MUST be the first Streamlit command in the script.
# It sets the browser tab title, icon, and default layout width.
# =============================================================================
st.set_page_config(
    page_title="OLS Diagnostic Dashboard",
    page_icon="📊",
    layout="wide",  # "wide" uses the full browser width instead of a narrow column
)

# =============================================================================
# CUSTOM CSS STYLING
# st.markdown() with unsafe_allow_html=True lets us inject raw CSS
# to override Streamlit's default styling for a more polished look.
# =============================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

    /* Main background and text */
    .stApp { background-color: #0e1117; color: #c9d1d9; }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        border: 1px solid #30363d;
        color: #8b949e;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }

    /* Header styling */
    h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #e6edf3 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING & CACHING
# @st.cache_data caches the return value so the function only runs ONCE.
# On subsequent reruns (triggered by widget interactions), Streamlit serves
# the cached DataFrame instead of regenerating it — a major performance win.
# =============================================================================
@st.cache_data
def load_sample_data():
    """Generate synthetic health/biometric data for OLS demonstration."""
    np.random.seed(42)
    n = 500

    height = np.random.normal(170, 10, n)
    weight = 0.8 * height + np.random.normal(-60, 8, n)
    bmi = weight / (height / 100) ** 2
    systolic = 90 + 0.3 * weight + 0.1 * bmi + np.random.normal(0, 8, n)
    diastolic = 55 + 0.2 * weight + np.random.normal(0, 6, n)
    age = np.random.randint(20, 70, n)
    # Target: cholesterol influenced by multiple features + heteroscedastic noise
    cholesterol = (
        80
        + 0.5 * weight
        + 0.8 * age
        + 0.3 * systolic
        + np.random.normal(0, 1, n) * (0.5 * weight)  # heteroscedastic error
    )

    return pd.DataFrame(
        {
            "Weight_kg": np.round(weight, 1),
            "Height_cm": np.round(height, 1),
            "BMI": np.round(bmi, 1),
            "Systolic_BP": np.round(systolic, 1),
            "Diastolic_BP": np.round(diastolic, 1),
            "Age": age,
            "Cholesterol": np.round(cholesterol, 1),
        }
    )


# =============================================================================
# MODEL FITTING — CACHED
# @st.cache_resource is used for non-serializable objects like fitted models.
# Unlike cache_data, it stores the actual Python object reference in memory.
# The `feature_key` param ensures re-caching when the user changes features.
# =============================================================================
@st.cache_resource
def fit_models(feature_key, target_col, _df, feature_cols):
    """
    Fit both Naive OLS and Robust HC3 models.
    _df is prefixed with underscore so Streamlit skips hashing it
    (we rely on feature_key for cache invalidation instead).
    """
    X = sm.add_constant(_df[feature_cols])
    y = _df[target_col]

    # --- Naive OLS (assumes homoscedastic errors) ---
    naive_model = sm.OLS(y, X).fit()

    # --- Robust HC3 (heteroscedasticity-consistent standard errors) ---
    # HC3 adjusts SEs to be valid even when residual variance is non-constant.
    robust_model = sm.OLS(y, X).fit(cov_type="HC3")

    return naive_model, robust_model


# =============================================================================
# VIF CALCULATION — CACHED
# VIF measures how much the variance of a coefficient is inflated due to
# collinearity with other predictors. VIF > 5 = moderate, VIF > 10 = severe.
# =============================================================================
@st.cache_data
def compute_vif(feature_key, _df, feature_cols):
    """Compute Variance Inflation Factor for each feature."""
    X = sm.add_constant(_df[feature_cols])
    vif_records = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_records.append(
            {
                "Feature": col,
                "VIF": round(variance_inflation_factor(X.values, i), 2),
            }
        )
    return pd.DataFrame(vif_records).sort_values("VIF", ascending=False)


# =============================================================================
# HETEROSCEDASTICITY TESTS — CACHED
# =============================================================================
@st.cache_data
def run_het_tests(feature_key, _df, feature_cols, target_col):
    """Run Breusch-Pagan and White's test for heteroscedasticity."""
    X = sm.add_constant(_df[feature_cols])
    y = _df[target_col]
    model = sm.OLS(y, X).fit()
    resid = model.resid
    exog = model.model.exog

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    white_stat, white_p, _, _ = het_white(resid, exog)

    return {
        "Breusch-Pagan": {"Statistic": round(bp_stat, 4), "p-value": round(bp_p, 6)},
        "White's Test": {
            "Statistic": round(white_stat, 4),
            "p-value": round(white_p, 6),
        },
    }


# =============================================================================
# PLOTLY THEME — consistent dark styling across all charts
# =============================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.8)",
    font=dict(family="JetBrains Mono, monospace", size=12, color="#c9d1d9"),
    margin=dict(l=60, r=30, t=50, b=50),
)


# =============================================================================
# MAIN APP LAYOUT
# =============================================================================
def main():
    # --- Header ---
    st.markdown("## 📊 OLS Regression Diagnostic Dashboard")
    st.markdown("*Interactive diagnostics for heteroscedasticity & multicollinearity*")
    st.divider()

    # --- Load data ---
    df = load_sample_data()

    # =================================================================
    # SIDEBAR CONTROLS
    # st.sidebar places widgets in the collapsible left panel.
    # Every widget interaction triggers a full script rerun — but
    # cached functions skip re-execution if inputs haven't changed.
    # =================================================================
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        target_col = st.selectbox(
            "🎯 Target Variable (Y)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index("Cholesterol"),
            help="The dependent variable your OLS model predicts.",
        )

        available_features = [c for c in df.columns if c != target_col]

        # st.multiselect lets the user pick multiple predictors dynamically.
        feature_cols = st.multiselect(
            "📐 Predictor Features (X)",
            options=available_features,
            default=["Weight_kg", "Height_cm", "BMI", "Systolic_BP", "Age"],
            help="Select 2+ features. VIF requires at least 2 predictors.",
        )

        st.markdown("---")
        st.markdown("### 🎨 Plot Settings")

        # Slider returns a numeric value; reruns the script on change.
        point_opacity = st.slider("Point Opacity", 0.1, 1.0, 0.5, 0.05)
        point_size = st.slider("Point Size", 2, 12, 5)
        color_scheme = st.selectbox(
            "Color Palette",
            ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo"],
        )

    # --- Guard: need at least 2 features ---
    if len(feature_cols) < 2:
        st.warning("⚠️ Please select at least **2 predictor features** to proceed.")
        st.stop()  # Halts execution without throwing an error

    # --- Fit models & compute diagnostics ---
    feature_key = "_".join(sorted(feature_cols)) + f"_{target_col}"
    naive_model, robust_model = fit_models(feature_key, target_col, df, feature_cols)
    vif_df = compute_vif(feature_key, df, feature_cols)
    het_results = run_het_tests(feature_key, df, feature_cols, target_col)

    # =================================================================
    # TOP-LEVEL METRICS ROW
    # st.columns() creates side-by-side containers. Each column is
    # independently populated, enabling a dashboard-style metric row.
    # =================================================================
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (Naive)", f"{naive_model.rsquared:.4f}")
    m2.metric("Adj. R²", f"{naive_model.rsquared_adj:.4f}")
    m3.metric("F-statistic", f"{naive_model.fvalue:.2f}")
    m4.metric("Max VIF", f"{vif_df['VIF'].max():.2f}")

    st.markdown("")

    # =================================================================
    # TABBED INTERFACE
    # st.tabs() creates a horizontal tab bar. Content inside each tab
    # is only rendered when that tab is active, keeping the UI clean.
    # =================================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Residual Diagnostics",
            "📋 Model Summaries",
            "🔢 VIF Analysis",
            "📊 Data Explorer",
        ]
    )

    # -----------------------------------------------------------------
    # TAB 1: RESIDUAL DIAGNOSTICS (Plotly scatterplots)
    # -----------------------------------------------------------------
    with tab1:
        st.markdown("### Residual Diagnostic Plots")
        st.caption(
            "Look for patterns in residuals — random scatter = good. "
            "Funneling or curves suggest heteroscedasticity or non-linearity."
        )

        residuals = naive_model.resid
        fitted = naive_model.fittedvalues
        std_resid = naive_model.get_influence().resid_studentized_internal

        # --- Two-column layout for side-by-side plots ---
        col1, col2 = st.columns(2)

        with col1:
            # Residuals vs Fitted Values
            fig1 = px.scatter(
                x=fitted,
                y=residuals,
                labels={"x": "Fitted Values", "y": "Residuals"},
                title="Residuals vs Fitted Values",
                opacity=point_opacity,
                color=np.abs(residuals),
                color_continuous_scale=color_scheme,
            )
            fig1.update_traces(marker=dict(size=point_size))
            # Add a horizontal reference line at y=0
            fig1.add_hline(y=0, line_dash="dash", line_color="#f85149", line_width=2)
            fig1.update_layout(**PLOTLY_LAYOUT)
            fig1.update_layout(coloraxis_colorbar=dict(title="|Resid|"))
            # st.plotly_chart renders an interactive Plotly figure.
            # use_container_width=True makes it responsive to the column width.
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Scale-Location Plot (√|standardized residuals| vs fitted)
            sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))
            fig2 = px.scatter(
                x=fitted,
                y=sqrt_abs_std_resid,
                labels={"x": "Fitted Values", "y": "√|Standardized Residuals|"},
                title="Scale-Location Plot",
                opacity=point_opacity,
                color=sqrt_abs_std_resid,
                color_continuous_scale=color_scheme,
            )
            fig2.update_traces(marker=dict(size=point_size))
            fig2.update_layout(**PLOTLY_LAYOUT)
            fig2.update_layout(coloraxis_colorbar=dict(title="√|Std Res|"))
            st.plotly_chart(fig2, use_container_width=True)

        # --- Second row of plots ---
        col3, col4 = st.columns(2)

        with col3:
            # Q-Q Plot for normality of residuals
            from scipy import stats as scipy_stats

            sorted_resid = np.sort(std_resid)
            theoretical_q = scipy_stats.norm.ppf(
                np.linspace(0.01, 0.99, len(sorted_resid))
            )
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=sorted_resid,
                    mode="markers",
                    marker=dict(
                        size=point_size, color="#58a6ff", opacity=point_opacity
                    ),
                    name="Residuals",
                )
            )
            # 45-degree reference line
            min_val = min(theoretical_q.min(), sorted_resid.min())
            max_val = max(theoretical_q.max(), sorted_resid.max())
            fig3.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="#f85149", dash="dash", width=2),
                    name="45° Line",
                )
            )
            fig3.update_layout(
                title="Q-Q Plot (Normality Check)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Standardized Residuals",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            # Histogram of residuals
            fig4 = px.histogram(
                x=residuals,
                nbins=35,
                labels={"x": "Residuals", "y": "Count"},
                title="Residual Distribution",
                color_discrete_sequence=["#58a6ff"],
                opacity=0.8,
            )
            fig4.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        # --- Heteroscedasticity Test Results ---
        st.markdown("### 🧪 Formal Heteroscedasticity Tests")
        st.caption(
            "If p-value < 0.05, reject H₀ of homoscedasticity → use Robust HC3 SEs."
        )
        ht1, ht2 = st.columns(2)
        for col_widget, (test_name, vals) in zip([ht1, ht2], het_results.items()):
            sig = "🔴 Significant" if vals["p-value"] < 0.05 else "🟢 Not Significant"
            col_widget.metric(
                f"{test_name}",
                f"p = {vals['p-value']:.6f}",
                delta=sig,
                delta_color="inverse" if vals["p-value"] < 0.05 else "normal",
            )

    # -----------------------------------------------------------------
    # TAB 2: MODEL SUMMARIES (toggle Naive vs Robust)
    # -----------------------------------------------------------------
    with tab2:
        st.markdown("### Model Summary Comparison")

        # st.toggle creates an on/off switch widget.
        # When toggled, Streamlit reruns and this variable flips.
        use_robust = st.toggle(
            "🔄 Show Robust HC3 Summary",
            value=False,
            help="HC3 adjusts standard errors for heteroscedasticity.",
        )

        active_model = robust_model if use_robust else naive_model
        model_label = "Robust HC3" if use_robust else "Naive OLS"

        st.markdown(f"**Currently viewing: `{model_label}`**")

        # Capture the statsmodels .summary() text output
        summary_buf = io.StringIO()
        summary_buf.write(active_model.summary().as_text())

        # st.code renders preformatted monospaced text — perfect for model summaries.
        st.code(summary_buf.getvalue(), language=None)

        # --- Coefficient comparison chart ---
        st.markdown("### Coefficient Comparison: Naive vs Robust")
        coef_df = pd.DataFrame(
            {
                "Feature": naive_model.params.index,
                "Naive_Coef": naive_model.params.values,
                "Naive_SE": naive_model.bse.values,
                "Robust_Coef": robust_model.params.values,
                "Robust_SE": robust_model.bse.values,
            }
        )
        coef_df = coef_df[coef_df["Feature"] != "const"]

        fig_coef = go.Figure()
        fig_coef.add_trace(
            go.Bar(
                name="Naive OLS",
                x=coef_df["Feature"],
                y=coef_df["Naive_Coef"],
                error_y=dict(type="data", array=coef_df["Naive_SE"].values * 1.96),
                marker_color="#58a6ff",
                opacity=0.8,
            )
        )
        fig_coef.add_trace(
            go.Bar(
                name="Robust HC3",
                x=coef_df["Feature"],
                y=coef_df["Robust_Coef"],
                error_y=dict(type="data", array=coef_df["Robust_SE"].values * 1.96),
                marker_color="#f0883e",
                opacity=0.8,
            )
        )
        fig_coef.update_layout(
            barmode="group",
            title="Coefficients with 95% CI Error Bars",
            yaxis_title="Coefficient Value",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 3: VIF ANALYSIS
    # -----------------------------------------------------------------
    with tab3:
        st.markdown("### Variance Inflation Factor (VIF) Scores")
        st.caption("VIF = 1 → no collinearity · VIF > 5 → moderate · VIF > 10 → severe")

        # --- Color-coded VIF bar chart ---
        def vif_color(val):
            if val >= 10:
                return "#f85149"  # red — severe
            elif val >= 5:
                return "#d29922"  # yellow — moderate
            else:
                return "#3fb950"  # green — acceptable

        colors = [vif_color(v) for v in vif_df["VIF"]]

        fig_vif = go.Figure(
            go.Bar(
                x=vif_df["VIF"],
                y=vif_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=vif_df["VIF"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                textfont=dict(color="#c9d1d9"),
            )
        )
        # Reference lines for thresholds
        fig_vif.add_vline(
            x=5,
            line_dash="dash",
            line_color="#d29922",
            annotation_text="Moderate (5)",
            annotation_position="top",
        )
        fig_vif.add_vline(
            x=10,
            line_dash="dash",
            line_color="#f85149",
            annotation_text="Severe (10)",
            annotation_position="top",
        )
        fig_vif.update_layout(
            title="VIF Scores by Feature",
            xaxis_title="VIF",
            yaxis_title="",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_vif, use_container_width=True)

        # --- VIF Table ---
        st.markdown("#### Detailed VIF Table")

        # st.dataframe renders an interactive, sortable table.
        st.dataframe(
            vif_df.style.applymap(
                lambda v: (
                    "color: #f85149; font-weight: bold"
                    if v >= 10
                    else "color: #d29922" if v >= 5 else "color: #3fb950"
                ),
                subset=["VIF"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # --- Correlation heatmap ---
        st.markdown("#### Feature Correlation Matrix")
        corr = df[feature_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap",
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 4: DATA EXPLORER
    # -----------------------------------------------------------------
    with tab4:
        st.markdown("### Raw Data Explorer")

        # st.expander creates a collapsible section to keep the UI tidy.
        with st.expander("📋 View Dataset", expanded=True):
            st.dataframe(df, use_container_width=True, height=400)

        st.markdown("### Feature Distribution")
        selected_feature = st.selectbox("Select feature to plot", feature_cols)

        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            fig_hist = px.histogram(
                df,
                x=selected_feature,
                nbins=40,
                title=f"Distribution of {selected_feature}",
                color_discrete_sequence=["#58a6ff"],
                marginal="box",  # adds a box plot above the histogram
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True)

        with dist_col2:
            fig_scatter = px.scatter(
                df,
                x=selected_feature,
                y=target_col,
                title=f"{selected_feature} vs {target_col}",
                opacity=point_opacity,
                color=df[selected_feature],
                color_continuous_scale=color_scheme,
                trendline="ols",  # adds an OLS trendline automatically
            )
            fig_scatter.update_traces(marker=dict(size=point_size))
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Footer ---
    st.divider()
    st.caption(
        "Built with Streamlit + Plotly + statsmodels · "
        "Diagnostics: Breusch-Pagan, White's Test, VIF, HC3 Robust SEs"
    )


# =============================================================================
# ENTRY POINT
# Python runs main() only when the script is executed directly.
# Streamlit re-executes the entire script top-to-bottom on every interaction.
# =============================================================================
if __name__ == "__main__":
    main()
