# ── MASA Hackathon 2026: R-Ignite
# Step 1 — Load WDI Wide Format + external CSVs, merge into one clean table

import pandas as pd
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────
COUNTRIES_CODE = ["MYS", "IDN"]
COUNTRY_NAMES  = {"MYS": "Malaysia", "IDN": "Indonesia"}
INDICATORS = {
    "WB_WDI_AG_LND_FRST_ZS"         : "forest_pct",
    "WB_WDI_EN_GHG_ALL_LU_MT_CE_AR5": "total_ghg_kt_include_CO2",
}
YEAR_START = 2000
YEAR_END   = 2023   # cap at 2023 to match WDI data

# ── PART A: WDI Wide Format ─────────────────────────────────────────────────
df = pd.read_csv("WB_WDI_WIDEF.csv", dtype=str, engine='python', on_bad_lines='skip')

filtered = df[
    (df["REF_AREA"].isin(COUNTRIES_CODE)) &
    (df["INDICATOR"].isin(INDICATORS.keys()))
].copy()

year_cols = [str(y) for y in range(YEAR_START, YEAR_END + 1)]

long = filtered.melt(
    id_vars=["REF_AREA", "INDICATOR"],
    value_vars=year_cols,
    var_name="year",
    value_name="value"
)
long["year"]      = long["year"].astype(int)
long["value"]     = pd.to_numeric(long["value"], errors="coerce")
long["indicator"] = long["INDICATOR"].map(INDICATORS)
long["COUNTRY"]   = long["REF_AREA"].map(COUNTRY_NAMES)
long = long.dropna(subset=["value"])

# Pivot WDI to wide
wdi_wide = long.pivot_table(
    index=["COUNTRY", "year"],
    columns="indicator",
    values="value"
).reset_index()

# ── PART B: Average Precipitation ──────────────────────────────────────────
precip = pd.read_csv("average-precipitation-per-year.csv")
# Column names: Entity, Code, Year, Annual precipitation
precip = precip[precip["Entity"].isin(["Indonesia", "Malaysia"])].copy()
precip = precip[(precip["Year"] >= YEAR_START) & (precip["Year"] <= YEAR_END)]
precip = precip.rename(columns={
    "Entity"              : "COUNTRY",
    "Year"                : "year",
    "Annual precipitation": "average_precipitation"
})[["COUNTRY", "year", "average_precipitation"]]

# ── PART C: Fossil Fuel Primary Energy ─────────────────────────────────────
fossil = pd.read_csv("fossil-fuel-primary-energy.csv")
# Column names: Entity, Code, Year, Fossil fuels
fossil = fossil[fossil["Entity"].isin(["Indonesia", "Malaysia"])].copy()
fossil = fossil[(fossil["Year"] >= YEAR_START) & (fossil["Year"] <= YEAR_END)]
fossil = fossil.rename(columns={
    "Entity"      : "COUNTRY",
    "Year"        : "year",
    "Fossil fuels": "fossil_fuel_twh"
})[["COUNTRY", "year", "fossil_fuel_twh"]]

# ── PART D: Merge All ───────────────────────────────────────────────────────
# Start from a complete grid of country × year so no rows are dropped
countries = ["Indonesia", "Malaysia"]
years     = list(range(YEAR_START, YEAR_END + 1))
base      = pd.DataFrame(
    [(c, y) for c in countries for y in years],
    columns=["COUNTRY", "year"]
)

merged = (
    base
    .merge(precip,   on=["COUNTRY", "year"], how="left")
    .merge(wdi_wide, on=["COUNTRY", "year"], how="left")
    .merge(fossil,   on=["COUNTRY", "year"], how="left")
)

# ── PART E: Reorder columns to match required format ───────────────────────
col_order = [
    "COUNTRY", "year",
    "average_precipitation",
    "forest_pct",
    "total_ghg_kt_include_CO2",
    "fossil_fuel_twh"
]
merged = merged[col_order].sort_values(["COUNTRY", "year"]).reset_index(drop=True)

# ── PART F: Save ────────────────────────────────────────────────────────────
merged.to_csv("wdi_model_data.csv", index=False)
print(f"Saved {len(merged)} rows → wdi_model_data.csv")
print(merged.head(10).to_string(index=False))