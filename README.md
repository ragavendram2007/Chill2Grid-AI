# Chill2Grid AI: Physics-Informed Demand Flexibility for Cold-Chain Refrigeration

> **IEEE ClimateChange Global Hackathon 2026**  
> *Fail-Closed Safety Guardian, Predictive Conformal Envelopes, and Grid Demand Response Optimization.*

---

## ❄️ Executive Overview

Refrigerated cold-storage warehouses and vaccine vaults represent massive, thermally flexible electrical loads. However, commercial demand response (DR) is blocked by product excursion liability: a single over-temperature event can destroy millions of dollars worth of biologics or food.

**Chill2Grid AI** implements a **Fail-Closed Safety Guardian** that sits between grid DR signals and physical facility controllers. The optimizer and grid operators can only propose curtailment actions; the Safety Guardian evaluates physics bounds, calibrated conformal ML risk bounds, equipment anti-cycling interlocks, and thermal debt history before granting an immutable, time-limited cryptographic lease.

---

## 🚀 Quickstart & How to Run

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
git clone https://github.com/ragavendram/chill2grid-ai.git
cd chill2grid-ai
pip install -r requirements.txt
pip install streamlit plotly
```

### 2. Run the Interactive Dashboard
Launch the full dark-themed 5-tab Streamlit dashboard:
```bash
streamlit run app/dashboard.py
```
*Access the dashboard in your browser at `http://localhost:8501`.*

### 3. Run the CLI Executive Demo
Run the simulated 35°C summer heat wave dispatch demo with one command:
```bash
python -m app.run_demo
```
*Outputs a formatted schedule table to the console and generates `outputs/part5/demo_summary.md`.*

### 4. Offline Static Dashboard
For environments without a running Python server or Streamlit runtime:
- Open `outputs/dashboard_static.html` directly in any web browser.
- Fully interactive with Plotly CDN charts and scenario toggles (25°C, 35°C, 40°C).

### 5. Run the Test Suite
Run the comprehensive 87-test verification suite (all green):
```bash
python -m pytest
```

---

## 🏛️ Project Architecture

```
chill2grid-ai/
├── app/
│   ├── dashboard.py               # 5-Tab Streamlit interactive UI
│   ├── run_demo.py                # Standalone CLI demo runner
│   └── build_html.py              # Offline static HTML dashboard generator
├── optimizer/
│   ├── plan_event.py              # Grid DR optimizer, G2 defer offsets, staggered restart
│   └── __init__.py
├── impact/
│   ├── assess.py                  # Exact twin simulation (simulate_pair), net kWh, tariff/CO2
│   └── __init__.py
├── safety/
│   ├── safety_guardian.py         # Fail-closed Safety Guardian engine (G0-G5, Key 1, Key 2)
│   ├── decision_ledger.py         # Single decision ledger generator & invariant verifier
│   ├── calibrate_guardian.py      # Approvable-subset scenario bootstrap calibration
│   ├── debt_ledger.py             # Thermal debt ledger, inter-event cooldown & 24h budget
│   ├── key1_conformal.py          # Conformal ML safety predictor
│   ├── key2_physics.py            # Calibrated physics twin & dynamic restart overshoot reserve
│   ├── audit.py                   # SHA-256 rolling cryptographic audit chain
│   ├── ood.py                     # Out-of-distribution feature distance guard
│   ├── sensor_integrity.py        # CUSUM drift detection & fault isolation
│   └── rules.py                   # Risk tier definitions (CRITICAL, HIGH, STANDARD)
├── simulation/
│   ├── thermal_model.py           # 2-capacity lumped RC thermal ODE twin & twin differential runner
│   └── demo_part2.py              # Part 2 baseline analysis runner
├── assumptions/
│   └── grid_assumptions.csv       # Hourly illustrative tariff (INR/kWh) and CO2 (g/kWh) profile
├── data/
│   ├── facilities.csv             # Fleet parameters for 6 industrial cold warehouses
│   └── safety/                    # Calib extra & red team datasets
├── outputs/
│   ├── dashboard_static.html      # Self-contained offline HTML dashboard
│   ├── part4/
│   │   ├── decision_ledger.csv.gz # Unified decision ledger across all red-team records
│   │   ├── readiness.json         # Readiness Gate checklist verification (R1–R10)
│   │   ├── rejection_funnel.csv   # 36-cell rejection distribution matrix
│   │   ├── redteam_by_tier.csv    # Tier-stratified empirical safety benchmarks
│   │   └── schedulable_flexibility.csv # Immediate vs 30m wait flexibility
│   └── part5/
│       ├── demo_summary.md        # Formatted demonstration report
│       └── impact_table.csv       # 3x3 scenario matrix (25, 35, 40°C x 30, 60, 90m)
└── tests/
    ├── test_contracts.py          # Core dataclass and validation tests (19 tests)
    ├── test_predictor.py          # ML trajectory and conformal tests (6 tests)
    ├── test_thermal_model.py      # Physics simulation and COP tests (17 tests)
    ├── test_training.py           # Model training and artifact pipeline tests (11 tests)
    ├── test_safety_guardian.py    # Guardian rules, audit chain, ledger invariant tests (30 tests)
    └── test_optimizer.py          # Optimizer bounds, knapsack, and staggering tests (4 tests)
```

---

## 🔍 The Scientific Honesty List

### 1. What is Simulated
- **Thermal dynamics:** Two-capacity lumped capacitance (air + cargo mass) ODEs with convective air-to-product coupling and ambient infiltration conductance.
- **Compressor cycle:** Variable refrigeration capacity and temperature-dependent COP degradation based on real ambient curves ($T_{\text{amb}} \in [18, 48]^\circ\text{C}$).
- **Enthalpy ingress:** Poisson-distributed door openings modeling humid air condensation and latent heat release.
- **Sensor noise:** Gaussian telemetry noise ($\sigma = 0.306^\circ\text{C}$) with Kalman filtering and CUSUM drift detection.

### 2. What is Invented
- **Reality Gap Injection:** An unmodeled $\pm 10\%$ multiplicative thermal disturbance injected into evaluation runs to rigorously stress test conformal coverage bounds.
- **Restart Latencies:** Fixed mechanical delays ($t_{\text{lag}} = 3.0$ min restart delay and $t_{\text{rev}} = 1.5$ min turnaround rise) are conservative engineering assumptions.
- **Grid Assumptions:** The 24-hour tariff profile (INR 4.0 off-peak to INR 13.0 peak) and emissions profile (560g to 850g CO2/kWh) are illustrative proxies.

### 3. What Real Data Would Be Needed for Commercial Deployment
1. **Minute-by-minute facility telemetry:** Live suction pressure, discharge pressure, compressor electrical power, and product core temperature probes.
2. **Compressor mechanical specifications:** OEM anti-short-cycle timer settings and oil return protocols directly from plant SCADA controllers.
3. **Live Grid ADR Feeds:** OpenADR 2.0b VEN client connection to Indian State Load Despatch Centre (SLDC) or DISCOM automated DR servers.
4. **14-day passive site baselining:** Undisturbed operation logging to calibrate empirical thermal capacitance and resistance parameters without synthetic gaps.

### 4. Fleet Operating Limits
- **FAC-VAX-03 (Vaccine Vault):** Classified as **CRITICAL**. Automatically locked out of all curtailment when ambient temperature reaches or exceeds 35°C (Gate 3). Excursion tolerance is 0°C.
- **FAC-ICE-02 & FAC-FZN-01 (Ice Cream & Deep Freeze):** Classified as **HIGH**. Maximum allowable single-event curtailment duration capped at 45 minutes to prevent irreversible crystallization defects.
- **First Law of Thermodynamics:** Total net energy change over settled horizons is **indistinguishable from 0 kWh**. Demand flexibility shifts peak electric stress; it does not eliminate the physical heat that entered the envelope.

---

## 📜 Compliance & Guarantees
- **87/87 Unit and Integration Tests Passing.**
- **Zero Core Contract Mutations:** `core/contracts.py` remains untouched.
- **100% Cryptographic Auditability:** Every decision cryptographically verifiable via SHA-256 rolling chain.
