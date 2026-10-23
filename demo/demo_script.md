# Chill2Grid AI - Hackathon Demo Script

## IEEE ClimateChange Global Hackathon
**Theme:** Green Energy Optimization through Data Analytics  
**Project:** Chill2Grid AI - Cold-Storage Thermal Inertia Virtual Power Plant (VPP)

---

## Part 1 Demo: Fleet Registration & Data Contracts

### 1. Verification of Requirements & Environment
Verify all required Python packages are installed:
```bash
python -m pip install -r requirements.txt
```

### 2. Run Comprehensive Unit Tests
Run the test suite verifying dataclasses, bounds checking, emission factors, and CSV ingestion:
```bash
python -m pytest
```
*Expected Result:* 19 passed tests in < 1 second.

### 3. Print Registered Fleet Summary (Demo Command)
Display the loaded cold storage fleet, physical parameters, and honest accounting disclosures:
```bash
python -m core.contracts
```
*Output Preview:*
```
=====================================================================================================================
CHILL2GRID AI - REGISTERED COLD STORAGE FLEET (SIMULATED)
=====================================================================================================================
+-------------+--------------------------+----------+----------------+--------------+--------------+----------+-----+
| Facility ID | Product Type             | Rated kW | Temp Range (C) | Safe Max (C) | Mass (kWh/C) | R (C/kW) | COP |
+-------------+--------------------------+----------+----------------+--------------+--------------+----------+-----+
| FAC-FZN-01  | Frozen Food              | 220.0    | -22.0 to -18.0 | -18.0        | 45.0         | 0.35     | 2.4 |
| FAC-ICE-02  | Ice Cream                | 120.0    | -26.0 to -22.0 | -22.0        | 25.0         | 0.40     | 2.1 |
| FAC-VAX-03  | Vaccines & Pharmaceutica | 45.0     | +2.0 to +6.5   | +6.5         | 14.0         | 0.50     | 3.0 |
| FAC-PRD-04  | Fresh Produce            | 310.0    | +1.0 to +5.5   | +5.5         | 65.0         | 0.28     | 3.2 |
| FAC-SEA-05  | Seafood & Marine Export  | 180.0    | -24.0 to -19.0 | -19.0        | 38.0         | 0.32     | 2.3 |
| FAC-MLK-06  | Dairy & Chilled Milk     | 85.0     | +1.5 to +4.0   | +4.0         | 20.0         | 0.38     | 3.1 |
+-------------+--------------------------+----------+----------------+--------------+--------------+----------+-----+
* National Grid Emission Factor: CEA India CO2 Baseline Database v19.0 (Dec 2023), 0.716 kg CO2/kWh (estimate)
* Simulated DR Reward Rate: Simulated tariff benchmarked at Rs 8.00/kWh for commercial demand flexibility
* Note: All telemetry and models are labeled as simulated / estimate.
```

---

## Roadmap of Upcoming Parts
- **Part 2:** Physics Thermal Simulator (RC thermal model, weather COP sensitivity, compressor cycling)
- **Part 3:** Hybrid Physics + ML Forecaster (RC physics baseline + residual ML estimator + 90th-percentile conservative bound)
- **Part 4:** Safety Guardian (Deterministic hard rules, compressor locks, sensor-fault veto, live curtailment override)
- **Part 5:** Rebound-Aware VPP Optimizer & Impact Analytics (Smart facility selection, staggered recovery, honest CEA carbon accounting)
- **Part 6:** Closed-Loop Event Runner & Scenarios (Grid peak, hot day recovery stress, sensor failure, shortfall handling)
- **Part 7:** Interactive Streamlit Dashboard (Fleet monitor, live curtailment simulation, carbon and financial accounting)
