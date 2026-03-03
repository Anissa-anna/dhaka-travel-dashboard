import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# 1) FILE paths (EDIT THESE)
# ----------------------------
WEATHER_PATH = "dhaka_monthly_temperature_clean.xlsx"
FLIGHT_PATH  = "Flights_Clean(1).xlsx"
CRIME_PATH   = "dhaka_crimes_2025_clean.xlsx"

# ----------------------------
# Helpers
# ----------------------------
def inv_minmax(series: pd.Series) -> pd.Series:
    """Inverse min-max normalize so lower values -> higher scores (1 is best)."""
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(series), index=series.index)
    return 1 - (series - mn) / (mx - mn)

def month_name(m: int) -> str:
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return names[m-1] if 1 <= m <= 12 else str(m)

def month_filter(df: pd.DataFrame, selected_months):
    return df[df["month"].isin(selected_months)].copy()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df

def force_int_month_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df[df["month"].between(1, 12)]
    return df

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Dhaka Travel Period Recommender", layout="wide")
st.title("Dhaka Travel Period Recommender")
st.caption("Monthly recommendation using: Temperature comfort (1995–2010) • Flight price/hour (2025–2026) • Crime (2025)")

# ----------------------------
# 2) Load EXCEL files (NO upload UI)
# ----------------------------

# ===== WEATHER =====
weather_raw = pd.read_excel(WEATHER_PATH, sheet_name="Dhaka_Monthly_Temp")
weather_raw = normalize_cols(weather_raw)

# Expect (after normalize): year_month, avg_temp_c
if "year_month" not in weather_raw.columns or "avg_temp_c" not in weather_raw.columns:
    st.error(f"Weather columns available: {list(weather_raw.columns)}")
    st.stop()

weather = weather_raw.copy()
weather["year_month"] = weather["year_month"].astype(str).str.strip()

# Robust parse for many formats
dt = pd.to_datetime(weather["year_month"], errors="coerce", infer_datetime_format=True)

# fallback for YYYYMM
mask_nat = dt.isna()
if mask_nat.any():
    dt2 = pd.to_datetime(weather.loc[mask_nat, "year_month"], errors="coerce", format="%Y%m")
    dt.loc[mask_nat] = dt2

weather["year"] = dt.dt.year
weather["month"] = dt.dt.month
weather["avg_temp_f"] = (pd.to_numeric(weather["avg_temp_c"], errors="coerce") * 9/5) + 32
weather = weather.dropna(subset=["year", "month", "avg_temp_f"])
weather = force_int_month_year(weather)

# ===== FLIGHTS =====
flights = pd.read_excel(FLIGHT_PATH)
flights = normalize_cols(flights)

# Rename common variants
rename_map = {
    "duration_hrs": "duration",
    "duration_hr": "duration",
    "durationhours": "duration",
    "duration_hours": "duration",
    "flight_duration": "duration",
    "totalfare": "total_fare",
    "total_fare_bdt": "total_fare",
    "totalfare_bdt": "total_fare",
    "fare_total": "total_fare",
    "total_price": "total_fare",
    "price": "total_fare",
}
for k, v in rename_map.items():
    if k in flights.columns and v not in flights.columns:
        flights.rename(columns={k: v}, inplace=True)

# Check required fields for dashboard computations
needed = {"year", "month", "duration", "total_fare"}
missing = sorted(list(needed - set(flights.columns)))
if missing:
    st.error(f"Flight file missing columns: {missing}. Available: {list(flights.columns)}")
    st.stop()

flights = force_int_month_year(flights)

# ===== CRIME =====
crime_raw = pd.read_excel(CRIME_PATH, sheet_name="in")
crime_raw = normalize_cols(crime_raw)

# Expect (after normalize): year, month, nombre_de_crimes OR year, month, number_crime
crime = crime_raw.copy()

# Map crime column name
if "number_crime" not in crime.columns:
    if "nombre_de_crimes" in crime.columns:
        crime.rename(columns={"nombre_de_crimes": "number_crime"}, inplace=True)
    elif "number_of_crime" in crime.columns:
        crime.rename(columns={"number_of_crime": "number_crime"}, inplace=True)

if "year" not in crime.columns or "month" not in crime.columns or "number_crime" not in crime.columns:
    st.error(f"Crime columns available: {list(crime.columns)}")
    st.stop()

crime["number_crime"] = pd.to_numeric(crime["number_crime"], errors="coerce")
crime = crime.dropna(subset=["number_crime"])
crime = force_int_month_year(crime)

# ----------------------------
# 3) Sidebar filters
# ----------------------------


st.sidebar.header("Filters")

# ---- Month selection (NO RANGE ANYMORE) ----
all_months = list(range(1, 13))
month_labels = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

selected_months = st.sidebar.multiselect(
    "Select months",
    options=all_months,
    default=all_months,
    format_func=lambda x: month_labels[x]
)

if not selected_months:
    st.warning("Please select at least one month.")
    st.stop()
temp_min, temp_max = st.sidebar.slider("Comfort temperature range (°F)", 50, 110, (68, 86))

st.sidebar.subheader("Flight filters")
econ_only = st.sidebar.checkbox("Economy only", value=False)

st.sidebar.subheader("Scoring (simple)")
w_price = st.sidebar.slider("Weight: Price/hour ", 0.0, 1.0, 0.7)
w_crime = st.sidebar.slider("Weight: Crime ", 0.0, 1.0, 0.3)

w_sum = w_price + w_crime
if w_sum == 0:
    w_price, w_crime = 0.7, 0.3
else:
    w_price, w_crime = w_price / w_sum, w_crime / w_sum

# Apply month filters
weather_f = month_filter(weather, selected_months)
flights_f = month_filter(flights, selected_months)
crime_f = month_filter(crime, selected_months)

# Economy filter
if econ_only and "class" in flights_f.columns:
    flights_f = flights_f[flights_f["class"].astype(str).str.lower().eq("economy")]

# Clean numeric fields + compute price/hour
flights_f["duration"] = pd.to_numeric(flights_f["duration"], errors="coerce")
flights_f["total_fare"] = pd.to_numeric(flights_f["total_fare"], errors="coerce")
flights_f = flights_f.dropna(subset=["duration", "total_fare"])
flights_f = flights_f[flights_f["duration"] > 0]
flights_f["price_per_hour"] = flights_f["total_fare"] / flights_f["duration"]

# ----------------------------
# 4) Monthly aggregations
# ----------------------------
weather_monthly_typical = (weather_f
                           .groupby("month", as_index=False)
                           .agg(avg_temp_f=("avg_temp_f", "mean")))
weather_monthly_typical["month_name"] = weather_monthly_typical["month"].astype(int).map(month_name)

flight_monthly_typical = (flights_f
                          .groupby("month", as_index=False)
                          .agg(avg_price_per_hour=("price_per_hour", "mean")))
flight_monthly_typical["month_name"] = flight_monthly_typical["month"].astype(int).map(month_name)

crime_monthly_all = (crime_f
                     .groupby(["year", "month"], as_index=False)
                     .agg(number_crime=("number_crime", "mean")))

# Prefer year=2025 if present; otherwise fallback to whatever exists
if (crime_monthly_all["year"] == 2025).any():
    crime_monthly = crime_monthly_all[crime_monthly_all["year"] == 2025].copy()
else:
    crime_monthly = (crime_f
                     .groupby("month", as_index=False)
                     .agg(number_crime=("number_crime", "mean")))

crime_monthly["month_name"] = crime_monthly["month"].astype(int).map(month_name)

# ----------------------------
# 5) Tabs (Page 1 = Recommendation)
# ----------------------------
tab_rec, tab_overview = st.tabs(["Page 1 — Recommendation", "Page 2 — Seasonal overview"])

# ============================================================
# PAGE 1 — Recommendation (Best Month + ranking)
# ============================================================
with tab_rec:
    st.subheader("Recommendation Best Month")

    # Build combined monthly table for scoring
    combined = pd.DataFrame({"month": selected_months})
    combined["month_name"] = combined["month"].astype(int).map(month_name)

    combined = combined.merge(weather_monthly_typical[["month", "avg_temp_f"]], on="month", how="left")
    combined = combined.merge(flight_monthly_typical[["month", "avg_price_per_hour"]], on="month", how="left")
    combined = combined.merge(crime_monthly[["month", "number_crime"]], on="month", how="left")

    combined = combined.dropna(subset=["avg_temp_f", "avg_price_per_hour", "number_crime"]).copy()
    if combined.empty:
        st.error("Combined monthly table is empty after joins. This usually means crime data isn't matching months.")
        st.write("Weather months:", sorted(weather_monthly_typical["month"].dropna().unique().tolist()))
        st.write("Flight months:", sorted(flight_monthly_typical["month"].dropna().unique().tolist()))
        st.write("Crime months:", sorted(crime_monthly["month"].dropna().unique().tolist()) if not crime_monthly.empty else [])
        st.stop()

    # Temperature comfort filter (constraint)
    combined["temp_ok"] = combined["avg_temp_f"].between(temp_min, temp_max)
    eligible = combined[combined["temp_ok"]].copy()

    if eligible.empty:
        st.warning("No month matches the selected comfort temperature range. Widen the range to get a recommendation.")
        st.dataframe(combined[["month_name", "avg_temp_f"]].sort_values("avg_temp_f"), use_container_width=True)
        st.stop()

    eligible["price_score"] = inv_minmax(eligible["avg_price_per_hour"])
    eligible["crime_score"] = inv_minmax(eligible["number_crime"])

    eligible["favorable_score"] = (w_price * eligible["price_score"] + w_crime * eligible["crime_score"])

    best = eligible.sort_values("favorable_score", ascending=False).iloc[0]


    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Month", f"{int(best['month'])} ({best['month_name']})")
    c2.metric("Avg Temp (°F)", f"{best['avg_temp_f']:.1f}")
    c3.metric("Avg Price/hour (BDT)", f"{best['avg_price_per_hour']:.0f}")
    c4.metric("Crime (context)", f"{best['number_crime']:.0f}")
    c5.metric("Favorable Score", f"{best['favorable_score']:.3f}")

    st.caption(
        f"Scoring uses only Price/hour and Crime with weights → Price: {w_price:.2f}, Crime: {w_crime:.2f}. "
        "Temperature comfort range is used as an eligibility filter."
    )

    st.markdown("### Monthly ranking")
    fig_score = px.bar(
        eligible.sort_values("favorable_score", ascending=False),
        x="month_name", y="favorable_score",
        title="Favorable Score by Month ",
        labels={"month_name": "Month", "favorable_score": "Favorable score"}
    )
    st.plotly_chart(fig_score, use_container_width=True)
    st.dataframe(
        eligible.sort_values("favorable_score", ascending=False)[
            ["month", "month_name", "avg_temp_f", "avg_price_per_hour", "number_crime",
             "price_score", "crime_score", "favorable_score"]
        ],
        use_container_width=True
    )



# ============================================================
# PAGE 2 — Seasonal overview (descriptive)
# ============================================================
with tab_overview:
    st.subheader("Seasonal overview ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Average Temperature by Month")
        fig_temp = px.line(
            weather_monthly_typical.sort_values("month"),
            x="month_name",
            y="avg_temp_f",
            markers=True,
            labels={"month_name": "Month", "avg_temp_f": "Average temperature (°F)"}
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        st.markdown("### Average Flight Price per Hour by Month")
        fig_price = px.bar(
            flight_monthly_typical.sort_values("month"),
            x="month_name",
            y="avg_price_per_hour",
            labels={"month_name": "Month", "avg_price_per_hour": "Average price per hour (BDT)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")
    st.subheader("Crime context")

    if crime_monthly.empty:
        st.warning("No crime data found after cleaning.")
    else:
        avg_crime = float(crime_monthly["number_crime"].mean())
        fig_crime = px.bar(
            crime_monthly.sort_values("month"),
            x="month_name",
            y="number_crime",
            labels={"month_name": "Month", "number_crime": "Number of crimes"}
        )
        st.plotly_chart(fig_crime, use_container_width=True)
        st.caption(f"Average monthly crime (context): {avg_crime:.0f}")

        st.markdown("---")
        st.subheader("Monthly Indicators Summary (Descriptive)")

        # Create base table from selected months
        summary = pd.DataFrame({"month": selected_months})
        summary["month_name"] = summary["month"].astype(int).map(month_name)

        # Merge weather
        summary = summary.merge(
            weather_monthly_typical[["month", "avg_temp_f"]],
            on="month",
            how="left"
        )

        # Merge flight prices
        summary = summary.merge(
            flight_monthly_typical[["month", "avg_price_per_hour"]],
            on="month",
            how="left"
        )

        # Merge crime
        summary = summary.merge(
            crime_monthly[["month", "number_crime"]],
            on="month",
            how="left"
        )

        # Sort by month number
        summary = summary.sort_values("month")

        # Display table
        st.dataframe(summary, use_container_width=True)