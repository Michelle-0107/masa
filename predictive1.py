import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


# --- 1. Methodology: Time-Series Feature Engineering ---
def load_precipitation_data(file_path):
    # Read the dataset directly from the file
    df = pd.read_csv(file_path)
    df = df.sort_values(["COUNTRY", "year"]).reset_index(drop=True)

    # REINSURANCE LOGIC: Rain is modeled as a univariate time-dependent process
    # lag_1_precip: Captures immediate short-term memory (T-1)
    df['lag_1_precip'] = df.groupby('COUNTRY')['average_precipitation'].shift(1)

    # rolling_3yr_precip: Captures historical trend and soil saturation potential (T-1, T-2, T-3)
    df['rolling_3yr_precip'] = df.groupby('COUNTRY')['average_precipitation'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    return df.dropna().reset_index(drop=True)


# Load data
df_climate = load_precipitation_data("wdi_model_data.csv")

# --- 2. Setup ---
COUNTRIES = ["Indonesia", "Malaysia"]
FEATURES = ["lag_1_precip", "rolling_3yr_precip"]
TARGET = "average_precipitation"

stage1_results = []

# --- 3. Modeling & Graphing ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("MASA Hackathon 2026 | Stage 1: Climate Trigger Forecast (Precipitation)",
             fontsize=16, fontweight='bold')

for i, country in enumerate(COUNTRIES):
    sub = df_climate[df_climate["COUNTRY"] == country].copy()

    # Backtest Validation: Train on data < 2023, Test on 2023 actuals
    train_data = sub[sub["year"] < 2023]
    test_data = sub[sub["year"] == 2023]

    # Fit univariate time-dependent regression
    model = LinearRegression()
    model.fit(train_data[FEATURES], train_data[TARGET])

    # 2023 Backtest calculation (One-step-ahead validation)
    actual_2023 = test_data[TARGET].iloc[0]
    pred_2023 = model.predict(test_data[FEATURES])[0]
    backtest_dev = mean_absolute_percentage_error([actual_2023], [pred_2023])

    # --- 4. Forward Projection (2024) ---
    # Prepare features for 2024 strictly using historical 2023 and 3-year rolling data
    recent_3yr_avg = sub[sub["year"].isin([2021, 2022, 2023])]["average_precipitation"].mean()
    features_2024 = pd.DataFrame({
        "lag_1_precip": [actual_2023],
        "rolling_3yr_precip": [recent_3yr_avg]
    })
    pred_2024 = model.predict(features_2024)[0]

    # Store results for the table
    stage1_results.append({
        "COUNTRY": country,
        "2023 ACTUAL": f"{actual_2023:.2f} mm",
        "2023 PREDICTED": f"{pred_2023:.2f} mm",
        "BACKTEST DEV": f"{backtest_dev:.1%}",
        "2024 FORECAST": f"{pred_2024:.2f} mm"
    })

    # --- 5. Visualization Layout ---
    ax = axes[i]
    # Blue: Historical Actuals
    ax.plot(sub["year"], sub[TARGET], 'o-', color='#1f77b4', label='Historical Actuals', linewidth=2, markersize=5)
    # Red: The 2023 Backtest point (Validation)
    ax.scatter(2023, pred_2023, color='#d62728', s=120, label=f'2023 Backtest (Dev: {backtest_dev:.1%})', zorder=5)
    # Orange: The 2024 Forecast (Forward Projection)
    ax.scatter(2024, pred_2024, color='#ff7f0e', marker='*', s=300, label=f'2024 Forecast: {pred_2024:.1f} mm',
               zorder=5)

    # Chart Styling
    ax.set_title(f"{country} Precipitation Trend", fontsize=14, pad=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Precipitation (mm)")
    ax.set_xticks(list(range(int(sub["year"].min()), 2025, 2)))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='best')

# --- 6. Professional Output (The Table) ---
print("\n" + "=" * 92)
print("  STAGE 1: CLIMATE TRIGGER FORECAST (PRECIPITATION)")
print("=" * 92)
# Aligned Header
print(
    f"{'COUNTRY':<12} | {'2023 ACTUAL':<14} | {'2023 PREDICTED':<16} | {'BACKTEST DEVIATION':<18} | {'2024 FORECAST'}")
print("-" * 92)
# Aligned Rows
for res in stage1_results:
    print(
        f"{res['COUNTRY']:<12} | {res['2023 ACTUAL']:<14} | {res['2023 PREDICTED']:<16} | {res['BACKTEST DEV']:<18} | {res['2024 FORECAST']}")
print("=" * 92)
print("METHODOLOGY NOTES:")
print("- Backtest Deviation: Out-of-sample prediction error (MAPE) using 2023 actual data.")
print("- Stochasticity: Precipitation is inherently stochastic and influenced by complex dynamics.")
print("-" * 92)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()