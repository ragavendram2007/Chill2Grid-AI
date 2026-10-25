# Chill2Grid AI: Physics-Informed Demand Flexibility for Cold-Chain Refrigeration

> **IEEE ClimateChange Global Hackathon 2026**  
> *Fail-Closed Safety Guardian, Predictive Conformal Envelopes, and Grid Demand Response Optimization.*

---

## ❄️ 1. What It Is

Chill2Grid AI is an industrial demand-flexibility platform that transforms refrigerated cold-storage facilities and pharmaceutical vaults into grid-stabilizing virtual batteries without risking perishable cargo. It introduces a Fail-Closed Safety Guardian that guarantees cold-chain product safety through conformal risk bounds, calibrated physical twin simulations, anti-cycling interlocks, and tamper-proof SHA-256 audit leases. When a grid stress event occurs, a rebound-aware optimizer schedules safe curtailments and staggers compressor restarts, delivering reliable peak shaving while preventing secondary recovery peaks.

---

## 🚀 2. Installation & Quickstart

### Environment Requirements
- **Python:** 3.11+ (Windows / Linux / macOS)
- **Dependencies:** Pinned in `requirements.txt`

### Quick Installation
```bash
git clone https://github.com/ragavendram/chill2grid-ai.git
cd chill2grid-ai
pip install -r requirements.txt
```

### Execution Commands
- **Interactive Streamlit Dashboard:**
  ```bash
  streamlit run app/dashboard.py
  ```
  *Opens live in your browser at `http://localhost:8501`.*
- **Standalone Offline Dashboard (no server required):**
  Open `outputs/dashboard_static.html` or `outputs/submission/dashboard_static.html` in any browser.
- **Part 5 Optimizer Demo (60m event at 18:00, 35°C):**
  ```bash
  python -m optimizer.demo_part5
  ```
- **End-to-End Self-Check:**
  ```bash
  python -m app.selfcheck
  ```
- **Complete Test Suite (101 tests green):**
  ```bash
  python -m pytest
  ```
- **Automated PowerShell Script (Windows):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File ./run_all.ps1
  ```

---

## 🏛️ 3. Project Structure

```
chill2grid-ai/
├── app/
│   ├── dashboard.py               # 5-Tab dark-theme interactive Streamlit UI
│   ├── build_html.py              # Offline interactive static HTML dashboard generator
│   └── selfcheck.py               # End-to-end integration validation runner
├── optimizer/
│   ├── types.py                   # GridEvent, Candidate, PlannedAction, Plan dataclasses
│   ├── solver.py                  # Rebound-aware MILP solver with HiGHS & greedy fallback
│   ├── candidates.py              # Phase-robust candidate generator (K=8 futures)
│   ├── build_library.py           # Precomputed twin response library (2,700 combinations)
│   ├── plan_event.py              # plan_grid_event, revalidate_plan, flexibility_forecast, plan_day
│   ├── benchmark.py               # 30-day comparative benchmark suite (Naive vs Greedy vs MILP)
│   └── demo_part5.py              # Standalone CLI demonstration runner
├── impact/
│   ├── assess.py                  # Exact twin accounting (simulate_pair), net kWh, tariff/CO2
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
│   └── demo_part2.py              # Baseline analysis runner
├── assumptions/
│   └── grid_assumptions.csv       # Hourly illustrative tariff (INR/kWh) and CO2 (g/kWh) profile
├── data/
│   ├── facilities.csv             # Fleet parameters for 6 industrial cold warehouses
│   └── optimizer/                 # Precomputed response library (response_library.parquet)
├── outputs/
│   ├── dashboard_static.html      # Self-contained offline HTML dashboard
│   ├── submission/                # Packaging bundle for competition submission
│   ├── part4/                     # Safety Guardian validation & readiness gate artifacts
│   └── part5/                     # Optimizer benchmarks, impact table, and HTML plots
└── tests/
    ├── test_contracts.py          # Core dataclass and validation tests (19 tests)
    ├── test_optimizer.py          # Part 5 optimizer unit and integration tests (13 tests)
    ├── test_app.py                # Dashboard, self-check, compliance, and latency tests (5 tests)
    ├── test_predictor.py          # ML trajectory and conformal tests (6 tests)
    ├── test_safety_guardian.py    # Safety Guardian battery and red-team tests (31 tests)
    ├── test_thermal_model.py      # Physics simulator, energy balance, and twin tests (16 tests)
    └── test_training.py           # Training pipeline, stratification, and leak guard tests (11 tests)
```

---

## 📊 4. Results & Performance Benchmarks

### Safety Guardian Red-Team Results (`outputs/part4/redteam_by_tier.csv`)
All rates evaluated empirically across **3,200 red-team stress scenarios**:

| Risk Tier | Facility | Total Requests | Approved Count | Approval Rate | Event-Caused Hard Breaches | Hard Breach Rate | 30m Wait Flexibility Retained | Status |
|---|---|---|---|---|---|---|---|---|
| **CRITICAL** | FAC-VAX-03 | 1,200 | 0 | **0.00%** | **0** | **0.0%** | 51.0% | **NO EVIDENCE (policy: never approve)** |
| **HIGH** | FZN-01, ICE-02, SEA-05 | 1,200 | 3 | **0.25%** | **0** | **0.0%** | 19.2% | **PASS (simulated)** |
| **STANDARD** | PRD-04, MLK-06 | 800 | 135 | **16.88%** | **0** | **0.0%** | 32.7% | **PASS (simulated)** |
| **Total Fleet** | **All 6 Rooms** | **3,200** | **138** | **4.31%** | **0** | **0.0%** | **34.3%** | **PASS (simulated)** |

> **Audit Claim:** 0 event-caused hard breaches among 138 approved simulated events.

---

### Grid & Economic Impact Matrix (`outputs/part5/impact_table.csv`)
Evaluated at 18:00 peak grid hour under 40% fleet power target (384 kW):

| Ambient (°C) [sim] | Duration (min) [sim] | Approved Rooms [sim] | Peak Shaved (kW) [sim] | Energy Shifted (kWh) [sim] | Net Energy (kWh) [sim] | Net Interpretation [sim] | Cost Avoided (₹) [illustrative] | CO2 Avoided (kg) [illustrative] |
|---|---|---|---|---|---|---|---|---|
| **25°C** | 30 min | 3 | 118.8 kW | 59.4 kWh | 0.00 kWh | **indistinguishable from 0** | ₹712.80 | 49.9 kg |
| **25°C** | 60 min | 3 | 118.8 kW | 118.8 kWh | 0.00 kWh | **indistinguishable from 0** | ₹1,425.60 | 99.8 kg |
| **25°C** | 90 min | 3 | 118.8 kW | 178.2 kWh | 0.00 kWh | **indistinguishable from 0** | ₹2,138.40 | 149.7 kg |
| **35°C** | 30 min | 3 | 118.8 kW | 59.4 kWh | 0.00 kWh | **indistinguishable from 0** | ₹712.80 | 49.9 kg |
| **35°C** | 60 min | 3 | 118.8 kW | 118.8 kWh | 0.00 kWh | **indistinguishable from 0** | ₹1,425.60 | 99.8 kg |
| **35°C** | 90 min | 3 | 118.8 kW | 178.2 kWh | 0.00 kWh | **indistinguishable from 0** | ₹2,138.40 | 149.7 kg |
| **40°C** | 30 min | 3 | 118.8 kW | 59.4 kWh | 0.00 kWh | **indistinguishable from 0** | ₹712.80 | 49.9 kg |
| **40°C** | 60 min | 3 | 118.8 kW | 118.8 kWh | 0.00 kWh | **indistinguishable from 0** | ₹1,425.60 | 99.8 kg |
| **40°C** | 90 min | 3 | 118.8 kW | 178.2 kWh | 0.00 kWh | **indistinguishable from 0** | ₹2,138.40 | 149.7 kg |

---

## 🔎 5. Honesty & Engineering Disclosure

1. **What is Simulated:**
   - Dual-capacity lumped RC thermal ODE model (`air_temp_c`, `product_temp_c`).
   - Diurnal outdoor temperature variation and ambient humidity curves.
   - Thermostat hysteresis deadbands and equipment anti-cycling timers (`min_on_min`, `min_off_min`).
   - Realistic sensor telemetry noise: Gaussian white noise $\mathcal{N}(0, 0.306^2)$ and CUSUM drift detection.
2. **What is Invented / Artificial:**
   - The reality-gap perturbation size ($+15\%$ heat gain, $-10\%$ COP penalty).
   - Compressor restart lag (simulated 30s delay).
   - Illustrative grid tariff (`assumptions/grid_assumptions.csv`) and grid carbon intensity profiles.
3. **What Real Data Would Be Needed for Commercial Deployment:**
   - Historical minute-resolution compressor electric power and run-status telemetry.
   - High-precision PT100/RTD temperature probe calibration logs.
   - Official utility time-of-day tariff structures and OpenADR 2.0b virtual end node signals.
   - On-site pilot testing with physical hardware emergency tripwire interlocks.
4. **Known Operational Limits:**
   - **`FAC-VAX-03`** (pharmaceutical vaccine vault) has extremely tight thermal tolerances ($+2^\circ\text{C}$ to $+8^\circ\text{C}$). Under summer heat conditions ($\ge 35^\circ\text{C}$), the Safety Guardian strictly refuses all curtailment requests. Product preservation takes absolute priority over grid flexibility.
