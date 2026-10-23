# Chill2Grid AI: Cold-Storage Thermal Inertia Virtual Power Plant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()
[![Hackathon](https://img.shields.io/badge/IEEE-ClimateChange_Hackathon-orange.svg)]()

> **Theme:** Green Energy Optimization through Data Analytics  
> **Core Concept:** Cold-storage rooms warm up slowly due to insulation and stored cargo (thermal inertia). Chill2Grid AI uses AI and physics modeling to predict safe curtailment duration for refrigerated facilities, coordinates them as a Virtual Power Plant (VPP) to relieve peak grid stress, and calculates the true net energy and CO₂ impact including post-event compressor rebound.

---

## Key Innovations

1. **Hybrid Physics + ML Forecaster:** Combines a 1st-order RC thermal model with residual ML error learning, evaluated alongside physics-only and ML-only benchmarks.
2. **Conservative 90th-Percentile Safe Window:** Evaluates temperature bounds using prediction intervals to guarantee safety under worst-case drift.
3. **Baseline-Aware Flexibility:** Curtailable power is strictly bounded by real baseline draw (duty cycle × rated kW), never arbitrary nameplate capacity.
4. **Weather-Aware Compressor Efficiency:** Accounts for COP degradation at high outdoor temperatures during recovery pull-down.
5. **Rebound-Aware VPP Optimizer:** Staggers post-event compressor restarts to prevent destructive secondary grid peaks.
6. **Deterministic Safety Guardian:** Hard thermodynamic and operational rules (anti-cycling, safe product margins, sensor fault vetoes) that override all ML predictions.
7. **Honest Impact Accounting:** Net energy accounting (event savings minus recovery excess), official CEA India grid emission factor (0.716 kg CO₂/kWh, estimate), and simulated flexibility incentives.

---

## Folder Structure

```
chill2grid-ai/
├── app.py                      # Streamlit interactive VPP dashboard
├── requirements.txt            # Pinned compatible dependencies
├── README.md                   # Project documentation
├── data/
│   └── facilities.csv          # Catalog of 6 cold-storage facilities
├── core/
│   ├── __init__.py
│   └── contracts.py            # Dataclasses, units, constants, CSV loader
├── simulation/
│   └── __init__.py             # Physics RC thermal simulator
├── training/
│   └── __init__.py             # Synthetic telemetry generator & model training
├── models/
│   └── __init__.py             # Hybrid physics + ML predictor
├── vpp/
│   └── __init__.py             # Rebound-aware VPP optimizer
├── safety/
│   └── __init__.py             # Deterministic Safety Guardian
├── analytics/
│   └── __init__.py             # True net energy & CEA CO2 accounting
├── events/
│   └── __init__.py             # Closed-loop scenario runner
├── tests/
│   ├── __init__.py
│   └── test_contracts.py       # Unit tests for contracts and data validation
└── demo/
    └── demo_script.md          # Step-by-step hackathon judge demo script
```

---

## Quickstart

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest
```

### 3. Run Part 1 Fleet Demo
```bash
python -m core.contracts
```

### 4. Launch Interactive Web App
```bash
streamlit run app.py
```

---

## Data Disclaimers & Constants
- **CO₂ Grid Factor:** `0.716 kg CO₂/kWh` based on *Central Electricity Authority (CEA) CO₂ Baseline Database for the Indian Power Sector, Version 19.0 (Dec 2023)*. Labeled as an **analytical estimate**.
- **Incentive Tariff:** Simulated at `₹8.00 / kWh` saved. Labeled as **simulated**.
- All telemetry, states, and forecasts are explicitly labeled as **simulated** or **estimate**.
