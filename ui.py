# ── MASA Hackathon 2026: R-Ignite
# Interactive Climate Risk Dashboard — Streamlit
#
# Run with:  streamlit run dashboard.py
# Requires:  pip install streamlit plotly pandas numpy scikit-learn

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="R-Ignite | Climate Risk Dashboard",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Advanced Dark Theme CSS (Fintech / Cyber Style) ─────────────────────────
st.markdown("""
<style>
    /* Global App Backgrounds */
    .stApp { background-color: #0A0F1D !important; }
    .main { background-color: #0A0F1D !important; }
    .block-container { padding-top: 1.5rem; }

    /* Hide Streamlit default header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0D1424 !important;
        border-right: 1px solid #1E293B;
    }

    /* Ensure all default text is light */
    p, span, div, h1, h2, h3, h4, h5, h6, label, li { color: #F8FAFC !important; }

    /* KPI Card Design */
    .kpi-card {
        background-color: #121A2F !important;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        text-align: left;
        position: relative;
        overflow: hidden;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00D2FF, #10B981);
    }
    .kpi-value { 
        font-size: 2.2rem; font-weight: 600; color: #F8FAFC !important; 
        margin-top: 5px; font-family: 'Inter', sans-serif;
    }
    .kpi-label { 
        font-size: 0.75rem; font-weight: 600; color: #64748B !important; 
        text-transform: uppercase; letter-spacing: 1px;
    }
    .kpi-delta { 
        font-size: 0.8rem; font-weight: 500; padding: 2px 6px; border-radius: 4px; display: inline-block;
        background: rgba(16, 185, 129, 0.15); color: #10B981 !important; 
    }
    .kpi-delta.negative { background: rgba(239, 68, 68, 0.15); color: #EF4444 !important; }

    /* Section Headers */
    .section-header {
        color: #F8FAFC !important; font-size: 1.4rem; font-weight: 600;
        margin-bottom: 1rem; border-bottom: 1px solid #1E293B; padding-bottom: 8px;
    }

    /* Insight / Policy / Financial Boxes */
    .insight-box, .policy-card, .financial-box, .recommendation-box {
        background-color: #121A2F !important; border-radius: 8px; padding: 20px;
        border: 1px solid #1E293B; margin-bottom: 15px; line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .financial-box {
        border-left: 4px solid #F59E0B; background: linear-gradient(90deg, rgba(245, 158, 11, 0.05), #121A2F);
    }
    .recommendation-box {
        border-left: 4px solid #EF4444; background: linear-gradient(90deg, rgba(239, 68, 68, 0.05), #121A2F);
    }

    .insight-box p, .policy-card p, .financial-box p, .recommendation-box p { color: #CBD5E1 !important; font-size: 0.95rem; }
    .insight-box b, .policy-card b, .financial-box b, .recommendation-box b { color: #F8FAFC !important; font-size: 1.05rem; display: block; margin-bottom: 8px; }

    .policy-card a { color: #00D2FF !important; text-decoration: none; font-weight: 500; display: inline-block; margin-top: 10px; }
    .policy-card a:hover { text-decoration: underline; color: #10B981 !important; }

    /* Streamlit Overrides */
    div[data-baseweb="select"] > div { background-color: #121A2F !important; border: 1px solid #1E293B !important; }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; }
    [data-testid="stMetricLabel"] { color: #64748B !important; }
    [data-testid="stDataFrame"] { background-color: #121A2F !important; }
</style>
""", unsafe_allow_html=True)


# ── Shared Plotly Layout Template ─────────────────────────────────────────────
def apply_fintech_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#64748B'),
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor='#1E293B', zeroline=False, title_font=dict(color='#CBD5E1')),
        yaxis=dict(showgrid=True, gridcolor='#1E293B', zeroline=False, title_font=dict(color='#CBD5E1')),
        legend=dict(orientation="h", y=-0.2, font=dict(color='#CBD5E1'))
    )
    return fig


# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("wdi_model_data.csv")
        df = df.sort_values(["COUNTRY", "year"]).reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("Error: wdi_model_data.csv not found.")
        return pd.DataFrame()


df = load_data()

COLORS = {"Indonesia": "#00D2FF", "Malaysia": "#10B981"}
INDICATOR_LABELS = {
    "average_precipitation": "Average Precipitation (mm)",
    "forest_pct": "Forest Cover (%)",
    "total_ghg_kt_include_CO2": "Total GHG Emissions (kt CO₂ eq.)",
    "fossil_fuel_twh": "Fossil Fuel Energy (TWh)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌏 R-Ignite 2026")
    st.markdown("**Climate Risk Dashboard**")
    st.markdown("*MASA Hackathon 2026*")
    st.divider()

    page = st.radio("Navigate", ["📊 Executive Summary", "🔍 Climate Risk Explorer", "🤖 Precipitation Forecast",
                                 "⚡ Scenario Analysis & Stress Test"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Data Source**")
    st.markdown("World Bank WDI (2000–2023)")
    st.divider()
    st.markdown("**Strategic Partner**")
    st.markdown("Hannover Re 🏢")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("<div class='section-header'>📊 Executive Summary</div>", unsafe_allow_html=True)
    st.markdown("### Climate Risk Assessment — Southeast Asia Reinsurance Exposure")
    st.markdown(
        "> **To:** Senior Management, Multinational Reinsurance Firm  \n> **From:** Actuarial Consulting Team  \n> **Re:** Climate-Related Risk Exposure — Indonesia & Malaysia (2000–2023)")
    st.divider()

    if not df.empty:
        for country in ["Indonesia", "Malaysia"]:
            sub = df[df["COUNTRY"] == country]
            yr00 = sub[sub["year"] == 2000].iloc[0]
            yr23 = sub[sub["year"] == 2023].iloc[0]

            st.markdown(f"#### 🌐 {country}")
            c1, c2, c3, c4 = st.columns(4)


            def kpi(col, label, val, delta, unit=""):
                arrow = "▲" if delta >= 0 else "▼"
                delta_class = "negative" if (delta >= 0 and label != "Forest Cover") or (
                            delta < 0 and label == "Forest Cover") else "positive"
                col.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label} (2023)</div>
                    <div class="kpi-value">{val}{unit}</div>
                    <div class="kpi-delta {delta_class}">{arrow} {abs(delta):.1f}% since 2000</div>
                </div>""", unsafe_allow_html=True)


            ghg_chg = (yr23["total_ghg_kt_include_CO2"] - yr00["total_ghg_kt_include_CO2"]) / abs(
                yr00["total_ghg_kt_include_CO2"]) * 100 if yr00["total_ghg_kt_include_CO2"] != 0 else 0
            ff_chg = (yr23["fossil_fuel_twh"] - yr00["fossil_fuel_twh"]) / yr00["fossil_fuel_twh"] * 100
            frst_chg = (yr23["forest_pct"] - yr00["forest_pct"]) / yr00["forest_pct"] * 100
            prec_chg = (yr23["average_precipitation"] - yr00["average_precipitation"]) / yr00[
                "average_precipitation"] * 100

            kpi(c1, "GHG Emissions (kt)", f"{yr23['total_ghg_kt_include_CO2']:,.0f}", ghg_chg)
            kpi(c2, "Fossil Fuel (TWh)", f"{yr23['fossil_fuel_twh']:,.0f}", ff_chg)
            kpi(c3, "Forest Cover", f"{yr23['forest_pct']:.1f}", frst_chg, "%")
            kpi(c4, "Precipitation", f"{yr23['average_precipitation']:,.0f}", prec_chg, " mm")
            st.markdown("")

        st.divider()

        st.markdown("<div class='section-header'>📈 Normalised Climate Indicator Trends (2000 = 100)</div>",
                    unsafe_allow_html=True)
        st.caption("All indicators rebased to 100 in year 2000 for direct comparison across countries.")

        fig = go.Figure()
        styles = {"total_ghg_kt_include_CO2": ("GHG Emissions", "solid"),
                  "fossil_fuel_twh": ("Fossil Fuel Use", "dash"), "forest_pct": ("Forest Cover", "dot"),
                  "average_precipitation": ("Precipitation", "dashdot")}
        for country in ["Indonesia", "Malaysia"]:
            sub = df[df["COUNTRY"] == country]
            base = sub[sub["year"] == 2000].iloc[0]
            for col, (label, dash) in styles.items():
                norm = sub[col] / base[col] * 100
                fig.add_trace(go.Scatter(x=sub["year"], y=norm, name=f"{country} – {label}",
                                         line=dict(color=COLORS[country], dash=dash, width=2), opacity=0.85,
                                         legendgroup=country))

        fig.add_hline(y=100, line_dash="dot", line_color="#475569", opacity=0.8)
        fig = apply_fintech_theme(fig)
        fig.update_layout(height=400, yaxis_title="Index (2000 = 100)")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- The Punchline ---
        st.markdown("<div class='section-header'>📌 Strategic Recommendation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="recommendation-box">
            <b>Actionable Mandates for Reinsurance Operations</b>
            <ul>
                <li style="margin-bottom: 8px;"><b>Pricing Adjustments (Physical Risk):</b> Increase catastrophe premium pricing in Indonesia to account for the heightened flood and landslide risks driven by a 11.5% drop in forest cover and 122% growth in fossil fuel reliance.</li>
                <li style="margin-bottom: 8px;"><b>Portfolio Management:</b> Reassess flood exposure concentrations in Malaysia (e.g., Klang Valley) given the increasing volatility in year-to-year precipitation swings.</li>
                <li><b>Capital Allocation (Transition Risk):</b> Actively reallocate capital toward climate-resilient portfolios and incentivize cedants demonstrating adherence to the Paris Agreement (1.5°C pathway) to minimize stranded asset exposure.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIMATE RISK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Climate Risk Explorer":
    st.markdown("<div class='section-header'>🔍 Climate Risk Explorer</div>", unsafe_allow_html=True)
    country = st.selectbox("Select Country", ["Indonesia", "Malaysia", "Both"])
    st.divider()

    if not df.empty:
        st.markdown("### 📈 Indicator Trends Over Time")
        cols_plot = list(INDICATOR_LABELS.keys())
        plot_df = df if country == "Both" else df[df["COUNTRY"] == country]

        fig = make_subplots(rows=2, cols=2, subplot_titles=list(INDICATOR_LABELS.values()), vertical_spacing=0.15,
                            horizontal_spacing=0.1)
        pos = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for idx, col in enumerate(cols_plot):
            r, c = pos[idx]
            if country == "Both":
                for ctry in ["Indonesia", "Malaysia"]:
                    sub = df[df["COUNTRY"] == ctry]
                    fig.add_trace(
                        go.Scatter(x=sub["year"], y=sub[col], name=f"{ctry}", line=dict(color=COLORS[ctry], width=2.5),
                                   legendgroup=ctry, showlegend=(idx == 0)), row=r, col=c)
            else:
                sub = plot_df
                fill_color = "rgba(0, 210, 255, 0.1)" if country == "Indonesia" else "rgba(16, 185, 129, 0.1)"
                fig.add_trace(go.Scatter(x=sub["year"], y=sub[col], fill="tozeroy", fillcolor=fill_color,
                                         line=dict(color=COLORS[country], width=3), name=INDICATOR_LABELS[col],
                                         showlegend=False), row=r, col=c)

        fig = apply_fintech_theme(fig)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### 🔗 Indicator Correlation Matrix")
        col_sel = st.selectbox("Country for correlation", ["Indonesia", "Malaysia"])
        sub_corr = df[df["COUNTRY"] == col_sel][list(INDICATOR_LABELS.keys())].corr()
        sub_corr.columns = [INDICATOR_LABELS[c].replace(" (", "\n(") for c in sub_corr.columns]
        sub_corr.index = sub_corr.columns

        fig_corr = px.imshow(sub_corr, text_auto=".2f", color_continuous_scale=["#0A0F1D", "#1E293B", "#00D2FF"],
                             zmin=-1, zmax=1, title=f"Correlation Matrix — {col_sel}")
        fig_corr = apply_fintech_theme(fig_corr)
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRECIPITATION FORECAST & FINANCIAL IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Precipitation Forecast":
    st.markdown("<div class='section-header'>🤖 Stage 1: Climate-Driven Forecast & Financial Proxy</div>",
                unsafe_allow_html=True)
    st.markdown(
        "**Target variable:** Annual Precipitation (mm) | **Methodology:** Multi-Variable Regression & Actuarial Proxy Calculation")
    st.divider()

    if not df.empty:
        df_climate = df.copy()
        df_climate['lag_1_precip'] = df_climate.groupby('COUNTRY')['average_precipitation'].shift(1)
        df_climate['rolling_3yr_precip'] = df_climate.groupby('COUNTRY')['average_precipitation'].transform(
            lambda x: x.shift(1).rolling(3).mean())
        df_climate = df_climate.dropna().reset_index(drop=True)

        FEATURES = ["lag_1_precip", "rolling_3yr_precip", "forest_pct", "total_ghg_kt_include_CO2"]
        TARGET = "average_precipitation"
        stage1_results = []

        BASE_EAL = {"Indonesia": 150.0, "Malaysia": 85.0}

        st.markdown("### 🎛️ Risk Sensitivity Calibration")
        elasticity_factor = st.slider(
            "EAL Sensitivity to Precipitation (Elasticity Multiplier)",
            min_value=1.0, max_value=2.5, value=1.5, step=0.1,
            help="Simulates how much a 1% deviation in precipitation affects Expected Annual Loss. Industry norm ranges from 1.2x to 2.0x."
        )
        st.divider()

        for ctry in ["Indonesia", "Malaysia"]:
            st.markdown(f"### 🌐 {ctry}")

            sub = df_climate[df_climate["COUNTRY"] == ctry].copy()
            train_data = sub[sub["year"] < 2023]
            test_data = sub[sub["year"] == 2023]

            model = LinearRegression()
            model.fit(train_data[FEATURES], train_data[TARGET])

            actual_2023 = test_data[TARGET].iloc[0]
            pred_2023 = model.predict(test_data[FEATURES])[0]
            backtest_dev = mean_absolute_percentage_error([actual_2023], [pred_2023])

            recent_3yr_avg = sub[sub["year"].isin([2021, 2022, 2023])]["average_precipitation"].mean()
            features_2024 = pd.DataFrame({
                "lag_1_precip": [actual_2023],
                "rolling_3yr_precip": [recent_3yr_avg],
                "forest_pct": [sub[sub["year"] == 2023]["forest_pct"].iloc[0]],
                "total_ghg_kt_include_CO2": [sub[sub["year"] == 2023]["total_ghg_kt_include_CO2"].iloc[0]]
            })
            pred_2024 = model.predict(features_2024)[0]

            historical_mean = train_data[TARGET].mean()
            precip_deviation_pct = ((pred_2024 - historical_mean) / historical_mean) * 100
            eal_increase_pct = precip_deviation_pct * elasticity_factor
            projected_eal = BASE_EAL[ctry] * (1 + (eal_increase_pct / 100))
            eal_delta = projected_eal - BASE_EAL[ctry]

            stage1_results.append({
                "Country": ctry,
                "2023 Actual (mm)": round(actual_2023, 2),
                "2023 Predicted (mm)": round(pred_2023, 2),
                "Backtest Dev (MAPE)": f"{backtest_dev:.1%}",
                "2024 Forecast (mm)": round(pred_2024, 2),
                "EAL Impact ($M)": round(eal_delta, 2)
            })

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("2023 Actual", f"{actual_2023:.1f} mm")
            m2.metric("2023 Backtest", f"{pred_2023:.1f} mm", delta=f"MAPE: {backtest_dev:.1%}", delta_color="inverse")
            m3.metric("Lag-1 Component", f"{model.coef_[0]:.2f}", help="Dependency on previous year's precipitation")
            m4.metric("2024 Forecast", f"{pred_2024:.1f} mm", delta=f"{pred_2024 - actual_2023:.1f} vs 2023")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["year"], y=sub[TARGET], mode="lines+markers", name="Historical Actuals",
                                     line=dict(color=COLORS[ctry], width=3), marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=[2023], y=[pred_2023], mode="markers", name=f"2023 Backtest",
                                     marker=dict(color="rgba(0,0,0,0)", size=14, symbol="circle-open",
                                                 line=dict(color="#EF4444", width=4))))
            fig.add_trace(go.Scatter(x=[2024], y=[pred_2024], mode="markers", name=f"2024 Forecast",
                                     marker=dict(size=18, color="#F59E0B", symbol="star")))

            fig = apply_fintech_theme(fig)
            fig.update_layout(height=380, yaxis_title="Annual Precipitation (mm)")
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"""
                    <div class="insight-box">
                    <b>📝 Climate-Driven Trigger Logic</b>
                    This model is an upgraded <b>Multivariate Climate Predictor</b>. It no longer relies strictly on naive time-series lags, but explicitly factors in regional <b>Forest Cover</b> and <b>GHG Emissions</b> to predict precipitation volatility. Backtest deviation is <b>{backtest_dev:.1%}</b>.
                    </div>""", unsafe_allow_html=True)
            with col_b:
                direction = "increase" if eal_delta > 0 else "decrease"
                st.markdown(f"""
                    <div class="financial-box">
                    <b>💰 Physical Risk Loss Estimator (EAL Proxy)</b>
                    Industry proxy assumes a <b>{elasticity_factor}x multiplier</b> for precipitation deviation against Expected Annual Loss (EAL).<br>
                    <span style="font-size: 0.8em; color: #94a3b8;"><i>*Calibrated proxy elasticity based on observed flood loss studies.</i></span><br><br>
                    Based on the forecasted {pred_2024:.0f} mm ({abs(precip_deviation_pct):.1f}% deviation from historical mean), 
                    projected property/flood portfolio claims in {ctry} will {direction} by <b>${abs(eal_delta):.1f} Million</b>.
                    </div>""", unsafe_allow_html=True)
            st.divider()

        st.markdown("### 📊 Stage 1 Summary Report")
        st.dataframe(pd.DataFrame(stage1_results), use_container_width=True, hide_index=True)

        st.markdown("""
        **Methodology Notes & Limitations:**
        * **Model Choice:** Linear Regression is explicitly utilized for its interpretability and transparency, which is a critical requirement in actuarial decision-making and regulatory submissions.
        * **Backtest Deviation:** Out-of-sample prediction error (MAPE) using 2023 actual data.
        * **⚠️ Spatial Granularity Limitation:** Climate hazards are highly localized. Country-wide averages do not capture micro-level exposures. *Future optimizations will integrate high-resolution Geospatial data (Lat/Long grids) to calculate Probable Maximum Loss (PML) for specific high-risk zones, such as the Klang Valley or Jakarta Metro.*
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SCENARIO ANALYSIS & STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Scenario Analysis & Stress Test":
    st.markdown("<div class='section-header'>⚡ Stage 2: Scenario Analysis & Tail Risk Stress Testing</div>",
                unsafe_allow_html=True)
    st.markdown(
        "**Capital Modeling:** Assessing the impact of transition pathways and extreme shocks on GHG trajectory and Transition Capital to 2030")
    st.markdown(
        "<span style='font-size: 0.9em; color: #94a3b8;'><i>This framework distinguishes between downside stress testing (tail risk shocks) and policy-aligned scenario analysis (transition pathways).</i></span>",
        unsafe_allow_html=True)
    st.divider()

    if not df.empty:
        country = st.selectbox("Select Country for Scenario Analysis", ["Indonesia", "Malaysia"])
        sub = df[df["COUNTRY"] == country].sort_values("year")

        st.markdown("### 🎛️ Climate Scenario Selection")
        scenario = st.selectbox(
            "Select Regulatory / Shock Pathway",
            ["Business as Usual (No Policy Intervention)",
             "Paris Agreement Aligned (1.5°C Pathway)",
             "Delayed Transition (Moderate Intervention)",
             "Tail Risk Scenario (1-in-100 Year Shock - Accelerated Degradation)"]
        )

        # 🚀 UPGRADED: True Tail Risk Parameters
        shock_active = False
        if scenario == "Business as Usual (No Policy Intervention)":
            def_ff, def_for = 0, 0
            help_text = "Baseline trajectory based on historical trends."
        elif scenario == "Paris Agreement Aligned (1.5°C Pathway)":
            def_ff, def_for = 45, 15
            help_text = "Aggressive transition simulating successful NDC targets."
        elif scenario == "Delayed Transition (Moderate Intervention)":
            def_ff, def_for = 20, 5
            help_text = "Late, uncoordinated policy responses."
        else:  # True Tail Risk Stress
            def_ff, def_for = -30, -20
            help_text = "Severe downside risk (1-in-100 year shock): Surging fossil fuel use and massive deforestation shocks."
            shock_active = True

        st.caption(help_text)

        col1, col2 = st.columns(2)
        with col1:
            ff_reduction = st.slider("Fossil Fuel Reduction by 2030 (%) [Negative = Increase]", min_value=-40,
                                     max_value=60, value=def_ff, step=5)
        with col2:
            forest_increase = st.slider("Forest Cover Increase by 2030 (%) [Negative = Deforestation]", min_value=-30,
                                        max_value=25, value=def_for, step=1)

        # 🚀 UPGRADED: The Shock Multiplier (Only visible/active during Stress)
        if shock_active:
            st.markdown("#### 🚨 Extreme Climate Shock Modifiers")
            col3, col4 = st.columns(2)
            with col3:
                shock_multiplier = st.slider(
                    "Sudden GHG Shock Multiplier (e.g., Extreme El Niño Fires)",
                    min_value=1.0, max_value=2.0, value=1.3, step=0.1,
                    help="Simulates a sudden external shock, multiplying the base stressed GHG output."
                )
            with col4:
                ghg_elasticity_factor = st.slider("Transition Risk Capital Penalty Factor", min_value=0.5,
                                                  max_value=1.5, value=0.8, step=0.1)
        else:
            shock_multiplier = 1.0  # No shock applied in transition scenarios
            ghg_elasticity_factor = st.slider(
                "Transition Capital Relief Factor",
                min_value=0.5, max_value=1.5, value=0.8, step=0.1,
                help="1% GHG change translates to this % of Capital Relief."
            )

        st.divider()

        X = sub[["fossil_fuel_twh", "forest_pct"]].values
        y = sub["total_ghg_kt_include_CO2"].values
        model = LinearRegression().fit(X, y)

        future_years = list(range(2024, 2031))
        ff_trend = LinearRegression().fit(sub["year"].values.reshape(-1, 1), sub["fossil_fuel_twh"].values)
        frt_trend = LinearRegression().fit(sub["year"].values.reshape(-1, 1), sub["forest_pct"].values)

        ff_base = ff_trend.predict(np.array(future_years).reshape(-1, 1))
        frt_base = frt_trend.predict(np.array(future_years).reshape(-1, 1))

        ghg_baseline = model.predict(np.column_stack([ff_base, frt_base]))

        ff_stressed = ff_base * (1 - np.linspace(0, ff_reduction / 100, len(future_years)))
        frt_stressed = frt_base * (1 + np.linspace(0, forest_increase / 100, len(future_years)))

        # Apply the base prediction, THEN apply the sudden shock multiplier
        ghg_stressed = model.predict(np.column_stack([ff_stressed, frt_stressed])) * shock_multiplier

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=sub["year"], y=sub["total_ghg_kt_include_CO2"], name="Historical GHG", mode="lines+markers",
                       line=dict(color=COLORS[country], width=3), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=future_years, y=ghg_baseline, name="Baseline Projection (BAU)",
                                 line=dict(color="#F59E0B", dash="dash", width=2.5)))

        # Color the stressed line based on whether it's good (green) or bad (red)
        stress_color = "#10B981" if (ff_reduction >= 0 and not shock_active) else "#EF4444"
        scenario_label = "Tail Risk Shock Scenario" if shock_active else "Transition Scenario"
        fig.add_trace(go.Scatter(x=future_years, y=ghg_stressed, name=scenario_label,
                                 line=dict(color=stress_color, dash="dot", width=3)))

        fig.add_trace(go.Scatter(
            x=future_years + future_years[::-1],
            y=list(ghg_baseline * 1.15) + list(ghg_baseline * 0.85)[::-1],
            fill="toself", fillcolor="rgba(245, 158, 11, 0.1)", line=dict(color="rgba(0,0,0,0)"),
            name="Model Risk & Volatility (±15% proxy band)"
        ))

        fig.add_vline(x=2023.5, line_dash="dot", line_color="#475569", annotation_text="Forecast →",
                      annotation_position="top right", annotation_font=dict(color="#CBD5E1"))
        fig = apply_fintech_theme(fig)
        fig.update_layout(height=450, title=f"{country} — GHG Emissions Projection to 2030",
                          yaxis_title="GHG Emissions (kt CO₂ eq.)")
        st.plotly_chart(fig, use_container_width=True)

        delta_ghg = ghg_baseline[-1] - ghg_stressed[-1]
        delta_pct = (delta_ghg / ghg_baseline[-1]) * 100 if ghg_baseline[-1] != 0 else 0

        st.markdown("### 📊 2030 Impact Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Baseline GHG 2030 (kt)", f"{ghg_baseline[-1]:,.0f}")

        delta_display = f"−{delta_ghg:,.0f} kt" if delta_ghg >= 0 else f"+{abs(delta_ghg):,.0f} kt"
        s2.metric("Projected GHG 2030 (kt)", f"{ghg_stressed[-1]:,.0f}", delta=delta_display, delta_color="inverse")

        if shock_active:
            s3.metric("Shock Increase vs BAU", f"{abs(delta_pct):.1f}%")
        else:
            s3.metric("Reduction Achieved", f"{delta_pct:.1f}%")

        s4.metric("Fossil Fuel Lever", f"{ff_reduction}% by 2030")

        st.divider()

        BASE_TRANSITION_CAPITAL = {"Indonesia": 120.0, "Malaysia": 75.0}

        capital_impact_pct = delta_pct * ghg_elasticity_factor
        capital_impact_m = BASE_TRANSITION_CAPITAL[country] * (capital_impact_pct / 100)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if shock_active:
                st.markdown(f"""
                <div class="insight-box" style="border-left: 4px solid #EF4444;">
                <b>🚨 Tail Risk Assessment (Stress Test)</b>
                Under this extreme 1-in-100 year shock scenario (incorporating a {shock_multiplier}x sudden external multiplier), {country}'s emissions trajectory surges by <b>{abs(delta_pct):.1f}%</b> against BAU.<br><br>
                This severe downside shock directly exacerbates the long-term physical risks modeled in Stage 1, while simultaneously forcing a massive capital crunch as regulatory penalties for stranded assets trigger immediately.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-box">
                <b>📝 Closing the Loop: Transition vs. Physical Risk</b>
                Under this scenario, {country}'s emissions trajectory deviates by <b>{delta_pct:.1f}%</b> against BAU.<br><br>
                Crucially, these transition pathways <b>directly influence the long-term physical risks captured in Stage 1</b>. Policy-aligned scenarios lower climate volatility, while mitigating the insurer's exposure to stranded assets.
                </div>""", unsafe_allow_html=True)

        with col_b:
            if delta_pct >= 0 and not shock_active:
                fin_title = "💰 Transition Risk Capital Relief (Proxy)"
                fin_action = f"The portfolio achieves a <b>{capital_impact_pct:.1f}% capital relief</b>. This unlocks <b>${capital_impact_m:.1f} Million</b> in regulatory transition capital."
                fin_border = "#10B981"
            else:
                fin_title = "⚠️ Transition Risk Capital Penalty (Stress Shock)"
                fin_action = f"The portfolio suffers a <b>{abs(capital_impact_pct):.1f}% capital penalty</b> due to severe deviations from NDC targets. This requires an additional <b>${abs(capital_impact_m):.1f} Million</b> to be held in regulatory capital reserves."
                fin_border = "#EF4444"

            st.markdown(f"""
            <div class="financial-box" style="border-left-color: {fin_border};">
            <b>{fin_title}</b>
            Actuarial proxy assumes a <b>{ghg_elasticity_factor}x elasticity</b>: GHG deviations directly impact internal Transition Risk Capital requirements.<br>
            <span style="font-size: 0.8em; color: #94a3b8;"><i>*Assumption based on emerging regulatory stress testing frameworks.</i></span><br><br>
            {fin_action}
            </div>""", unsafe_allow_html=True)