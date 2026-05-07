# 🌏 R-Ignite | Interactive Climate Risk Dashboard
### MASA Hackathon 2026 - Actuarial Consulting for Reinsurance

R-Ignite is an advanced analytical dashboard built to assess climate-related risk exposure for multinational reinsurance firms, specifically focusing on **Indonesia** and **Malaysia**. The platform translates complex environmental indicators (GHG emissions, forest cover, fossil fuel usage) into actionable financial insights and physical hazard forecasts.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed. The following core libraries are required:
- `streamlit`: Dashboard framework
- `plotly`: Interactive visualizations
- `pandas`: Data manipulation
- `scikit-learn`: Predictive modeling (Linear Regression)
- `numpy`: Numerical computations

### 2. Installation
Install all dependencies using the provided `requirements.txt` or via pip:
pip install streamlit plotly pandas numpy scikit-learn
```

### 3. Execution
Run the dashboard locally using the Streamlit CLI:
streamlit run dashboard.py
```

---

## 📊 Dashboard Structure
The dashboard is divided into four strategic modules:

### 1. Executive Summary
- **KPI Monitoring:** Real-time tracking of changes in GHG emissions, fossil fuel usage, and forest cover since 2000.
- **Normalized Trends:** Rebased index (2000 = 100) comparing climate trajectories across countries.
- **Strategic Recommendations:** Direct actuarial mandates for premium pricing adjustments and capital allocation.

### 2. Climate Risk Explorer
- **Visual Analytics:** Interactive multi-plot views of regional environmental indicators.
- **Correlation Matrix:** Risk heatmaps identifying statistical relationships between land-use changes and climate states.

### 3. Precipitation Forecast 
- **Predictive Modeling:** Utilizes a Multivariate Linear Regression model to forecast annual precipitation.
- **EAL Proxy:** Calculates the Expected Annual Loss (EAL) impact using an industry-standard elasticity multiplier (default: 1.5x) to estimate property/flood claims.
- **Backtesting:** Includes Mean Absolute Percentage Error (MAPE) validation against 2023 actual data.

### 4. Scenario Analysis & Stress Test 
- **Transition Pathways:** Simulates 2030 GHG trajectories under various scenarios, such as Paris Agreement Aligned vs. Business as Usual.
- **Tail Risk Stressing:** Impact modeling for 1-in-100 year extreme events, including massive deforestation or sudden emission surges.
- **Capital Impact:** Estimates transition risk capital relief or penalties based on regulatory stress-testing frameworks.

---

## 🛠️ Technical Methodology

- **Modeling Strategy:** The system utilizes Ordinary Least Squares (OLS) regression to ensure transparency and interpretability for actuarial decision-making and regulatory submissions.
- **Feature Engineering:** Incorporates lagged variables (Lag-1) and 3-year rolling averages to capture environmental "memory" and soil saturation potential.
- **Data Source:** Integrated with World Bank World Development Indicators (WDI) data spanning 2000–2023.

---

## 👥 Contributors
**Predictive Minds Team**
**Michelle Lee Xin Hui** (Team Leader) 
**Lim Zhan Xuan** (Report Editor) 
**Ng Wan Yang** (Data & Modelling) 
**Chiew Xin Yuan** (Actuarial & Risk Analysis)
