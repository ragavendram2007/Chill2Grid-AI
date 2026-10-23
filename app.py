"""app.py - Chill2Grid AI Streamlit Application.

Interactive Virtual Power Plant (VPP) Control & Cold-Storage Flexibility Platform.
Built for the IEEE ClimateChange Global Hackathon (Green Energy Optimization through Data Analytics).
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from core.contracts import (
    CO2_GRID_EMISSION_FACTOR_KG_PER_KWH,
    CO2_GRID_EMISSION_SOURCE,
    SIMULATED_REWARD_RATE_INR_PER_KWH,
    SIMULATED_REWARD_SOURCE,
    load_facilities_from_csv,
)

# Page configuration
st.set_page_config(
    page_title="Chill2Grid AI - Cold Storage VPP",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ Chill2Grid AI — Virtual Power Plant")
st.subheader("Green Energy Optimization through Cold-Storage Thermal Inertia")
st.caption(
    "IEEE ClimateChange Global Hackathon Entry | "
    "Predictive Demand Flexibility & Closed-Loop Decarbonization Analytics"
)

# Project Executive Summary
st.markdown(
    """
    Cold-storage facilities possess massive **thermal inertia** — well-insulated rooms loaded with frozen or chilled
    goods warm up slowly. By coordinating multiple commercial cold rooms as a **Virtual Power Plant (VPP)**,
    Chill2Grid AI safely sheds peak electricity demand without risking product spoilage, accounts for post-event
    compressor rebound, and computes honest net carbon offsets using official Indian grid emission factors.
    """
)

# Key Disclaimers & Parameters
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="CEA India Grid Factor (estimate)",
        value=f"{CO2_GRID_EMISSION_FACTOR_KG_PER_KWH} kg CO₂/kWh",
        help=CO2_GRID_EMISSION_SOURCE,
    )
with col2:
    st.metric(
        label="DR Incentive Rate (simulated)",
        value=f"₹{SIMULATED_REWARD_RATE_INR_PER_KWH:.2f} / kWh",
        help=SIMULATED_REWARD_SOURCE,
    )
with col3:
    st.metric(
        label="Safety Guarantee Standard",
        value="90th-Percentile Worst-Case",
        help="Deterministic Safety Guardian with conservative prediction bounds",
    )

st.markdown("---")

# Fleet Loading & Inspection
st.header("🏢 Registered Cold-Storage Fleet (Simulated)")

csv_path = Path(__file__).resolve().parent / "data" / "facilities.csv"
try:
    facilities = load_facilities_from_csv(csv_path)
    fleet_data = []
    for f in facilities:
        fleet_data.append(
            {
                "ID": f.id,
                "Facility Name": f.name,
                "Product Type": f.product_type,
                "Rated Power (kW)": f.rated_power_kw,
                "Min Temp (°C)": f.min_temp_c,
                "Max Safe Temp (°C)": f.max_safe_temp_c,
                "Thermal Mass (kWh/°C)": f.thermal_mass_kwh_per_c,
                "Thermal Resistance (°C/kW)": f.insulation_r_deg_c_per_kw,
                "Nominal COP": f.cop_nominal,
                "Min Run (min)": f.min_on_min,
                "Min Off (min)": f.min_off_min,
            }
        )
    df_fleet = pd.DataFrame(fleet_data)

    st.dataframe(
        df_fleet,
        use_container_width=True,
        hide_index=True,
    )

    total_fleet_power = sum(f.rated_power_kw for f in facilities)
    st.info(
        f"**Fleet Overview:** {len(facilities)} facilities loaded across diverse product classes "
        f"(Frozen Food, Ice Cream, Vaccines, Fresh Produce, Seafood, Dairy). "
        f"Total aggregate nameplate refrigeration capacity: **{total_fleet_power:.1f} kW**."
    )

    # Detailed Facility Inspection
    st.subheader("🔍 Facility Thermodynamic Profile")
    selected_id = st.selectbox(
        "Select a facility to inspect thermodynamic properties:",
        options=[f.id for f in facilities],
        format_func=lambda x: f"{x} - {next(f.name for f in facilities if f.id == x)}",
    )

    selected_fac = next(f for f in facilities if f.id == selected_id)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown(f"**Facility ID:** `{selected_fac.id}`")
        st.markdown(f"**Facility Name:** {selected_fac.name}")
        st.markdown(f"**Stored Product:** {selected_fac.product_type}")
        st.markdown(f"**Refrigeration Plant COP:** {selected_fac.cop_nominal}")

    with fcol2:
        st.markdown(f"**Rated Power:** {selected_fac.rated_power_kw:.1f} kW")
        st.markdown(f"**Set-point Range:** {selected_fac.min_temp_c:.1f}°C to {selected_fac.max_safe_temp_c:.1f}°C")
        allowed_drift = selected_fac.max_safe_temp_c - selected_fac.min_temp_c
        st.markdown(f"**Max Allowable Temperature Drift:** {allowed_drift:.1f}°C")

    with fcol3:
        st.markdown(f"**Thermal Mass:** {selected_fac.thermal_mass_kwh_per_c:.1f} kWh/°C")
        st.markdown(f"**Insulation Resistance:** {selected_fac.insulation_r_deg_c_per_kw:.2f} °C/kW")
        tau_min = selected_fac.thermal_mass_kwh_per_c * selected_fac.insulation_r_deg_c_per_kw * 60.0
        st.markdown(f"**Thermal Time Constant (RC τ):** {tau_min:.1f} minutes")

except Exception as err:
    st.error(f"Error loading facility catalog: {err}")
