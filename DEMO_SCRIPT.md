# Chill2Grid AI - 3-Minute Live Demo Script

**Event:** IEEE ClimateChange Global Hackathon 2026  
**Track:** Green Energy Optimization through Data Analytics  
**Prototype:** Fail-Closed Cold-Chain Demand Response with Safety Guardian & Staggered Recovery  
**Deployment URL:** `http://localhost:8501` (or standalone offline view at `outputs/dashboard_static.html`)

---

## Walkthrough Summary

| Step | Time | Screen / Tab | Action & Observation | Key Numbers Read from Files |
|---|---|---|---|---|
| **1** | 0:00 - 0:45 | **Tab 1: FLEET** | View baseline fleet at 35°C ambient. Switch sidebar preset to **Heat Wave (42°C)**. | Watch **FAC-VAX-03** trigger Gate 0/Key 1 refusal immediately (`REFUSE: Insufficient safe thermal headroom`). Rooms with large thermal mass (**FAC-PRD-04**, **FAC-FZN-01**) retain positive curtailment headroom. |
| **2** | 0:45 - 1:30 | **Tab 2: GRID EVENT** | Configure a **60-minute grid curtailment event at 18:00** on a 35°C day with target **384 kW** (40% of fleet power). Click **Plan Grid Event**. | Optimizer executes phase-robust candidate roll-forward in **< 3.0s**. Schedules **3 of 6 rooms** (FAC-FZN-01, FAC-PRD-04, FAC-SEA-05) with staggered restart delays (0m, 5m, 10m). Shortfall = 0.0 kW. |
| **3** | 1:30 - 2:00 | **Tab 2 (Curves)** | Scroll to the **Fleet Demand Profile & Staggered Rebound Peak** Plotly chart. | Unstaggered rebound peak creates a spike of **+75.0 kW**. Staggered recovery spreads compressor inrushes, keeping total fleet power strictly under baseline peak (**0.0 kW new peak created**). |
| **4** | 2:00 - 2:25 | **Tab 3: IMPACT** | Switch to **Tab 3: IMPACT**. Inspect physical and economic metrics. | **118.8 kW peak shaved**, **59.4 kWh shifted**. Net settled energy is **indistinguishable from 0** (thermodynamic conservation). Avoided costs and CO2 are tagged **assumption (illustrative)**: ₹712.80 and 49.9 kg CO2 avoided. |
| **5** | 2:25 - 2:45 | **Tab 4: SAFETY CASE** | Switch to **Tab 4: SAFETY CASE**. Inspect readiness badge and red-team table. | **Status: READY** (10 of 10 readiness gates passed, 96/96 pytest green). Red-team audit table from `outputs/part4/redteam_by_tier.csv`: **0 event-caused hard breaches among 138 approved simulated events** (CRITICAL: 0/1200 approved, HIGH: 3/1200 approved, STANDARD: 135/800 approved). Click **Verify Audit Chain** to verify SHA-256 tamper-proof ledger integrity. |
| **6** | 2:45 - 3:00 | **Tab 5: HONESTY** | Switch to **Tab 5: HONESTY**. | Review the honest disclosure: **Simulated physics twin with invented reality gap** (sensor drift, unmeasured door infiltration, compressor restart lag). Discloses physical limits: **FAC-VAX-03 is non-flexible during summer conditions**. Real-world deployment requires sensor calibration logs, AMR utility meter interfaces, and hardware tripwires. |

---

## Quick Command to Launch

```powershell
streamlit run app/dashboard.py
```
*(Offline fallback: open `outputs/dashboard_static.html` in any web browser)*
