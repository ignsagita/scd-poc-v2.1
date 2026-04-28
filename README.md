# Supply Chain Design Optimisation — POC v2

A Python proof-of-concept for multi-plant, multi-product supply chain network design.
Given lifecycle demand per product and receiving hub, the tool finds the optimal
allocation of production volume across factories to minimise total landed cost. It
subjects to trade-corridor rules, minimum production constraints, and strategic
coverage requirements.

All data in this repository is **fictional**. No company-specific information
is included. The model logic and file schemas are generic and applicable to any
manufacturing network.

---

## Live app

Deployed on Streamlit Community Cloud:

**[Open POC v2 →](https://scd-poc.streamlit.app)**

---

## What this solves

**How does a change in supply chain setup impact landed cost at the product level?**

The tool simulates four scenario types and compares them side by side:

| ID | Name | Logic |
|----|------|-------|
| **B1** | Nearest-Plant | Assigns each hub's demand to the cheapest allowed plant. Deterministic. |
| **B2** | Baseline Plant | All demand through one reference plant. Used as the cost-avoidance benchmark. |
| **OM** | Optimised (MILP) | Solves a mixed-integer programme to minimise total landed cost + fixed plant overhead + soft penalty for non-historical routes. |
| **GM** | Regional Mix | User specifies a target volume split by region (e.g. 30% Region_A / 70% Region_B). Cheapest plant in each region is used. |

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dummy input data
python generate_data.py

# 3. Launch the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

Streamlit Community Cloud is free and requires only a public GitHub repository.

**1. Push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/scd-poc-v2.git
git branch -M main
git push -u origin main
```

**2. Deploy**

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repository: `YOUR_USERNAME/scd-poc-v2`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **Deploy**

First deploy takes 2–3 minutes while packages install.

**Free tier notes:**
- The app sleeps after inactivity: open it a minute before any demo to wake it up
- Uploaded files are session-only (reset on reload); the `data/` folder in the repo is always the default dataset
- Memory limit: 1 GB (supposed to be well within what this model needs)

---

## Project structure

```
scd_v2/
│
├── app.py                  # Streamlit UI — 4 tabs: Data, Configuration, Results, About
├── scd_engine.py           # Computation engine — zero UI code, importable standalone
├── generate_data.py        # Generates all 8 input CSVs (schema contract)
├── requirements.txt
│
├── .streamlit/
│   └── config.toml         # Theme and upload size settings
│
└── data/                   # Input CSVs — replace with real data warehouse exports
    ├── plants.csv
    ├── products.csv
    ├── demand.csv
    ├── routes.csv
    ├── route_allowed.csv
    ├── hubs.csv
    ├── hist_flow.csv
    └── coverage_params.csv
```

`scd_engine.py` contains all computation with zero UI code. `app.py` is UI only and
imports from the engine. The engine can be tested independently or plugged into a
different UI without modification.

---

## App features

### Data tab
- Download any of the 8 input CSV templates, each with an embedded column guide explaining every field
- Upload modified files: each is validated on upload with row-level error messages
- Unuploaded files fall back to the built-in defaults automatically
- Preview the current active dataset

### Configuration tab
- **What-if demand sliders**: adjust lifecycle demand per hub by ±50% before running
- Select which scenarios to run (any combination of B1 / B2 / OM / GM)
- **B2:** single dropdown to select the baseline reference plant (applies to all products)
- **OM:** alpha slider with live explanation of what the penalty means in cost-equivalent %; regional and category coverage constraints; explicit Run button (no auto-recompute on slider touch)
- **GM:** regional percentage inputs with automatic sum validation

### Results tab
- **Scenario comparison table**: all KPIs for all scenarios in one consolidated table with cost-avoidance delta column
- **KPI metric cards**: per-scenario with hover tooltips explaining each KPI
- **Cost and regional breakdown charts** (Plotly, interactive)
- **Network summary matrix**: plant × hub heatmap showing allocated volumes; hover for product-level breakdown
- **Diff Sankey diagram**: OM vs a selected baseline, colour-coded by route status (continuing / new / dropped)
- **Alpha sensitivity analysis**: re-runs OM at 5 α values with chart and summary table
- **Alerts panel**: new-route alerts and consolidation candidates per scenario
- Full allocation and plant summary tables
- Download all results as a single ZIP or individual CSVs

### About tab
- LaTeX objective function
- Constraint reference table (C1–C8)
- Input file reference with column definitions
- Open points before real data connection
- POC v2 vs production comparison table

---

## Input files

All files in `data/` are generated by `generate_data.py` with fictional dummy values.
Column names and data types are the schema contract — replace with real data warehouse
exports using the same column structure.

| File | Content |
|------|---------|
| `plants.csv` | Plant master: ID, name, region, category (OWN/EXT), min production volume, fixed cost, capacity |
| `products.csv` | Product master: ID, family, category, base unit cost |
| `demand.csv` | Lifecycle demand per product and hub (omit rows with zero demand) |
| `routes.csv` | Landed-cost adder per (plant, hub, product category) |
| `route_allowed.csv` | Trade-corridor feasibility flag per (plant, hub) — product-agnostic |
| `hubs.csv` | Hub master: ID, name, region (display/reporting only) |
| `hist_flow.csv` | Historical flow flags per (product, plant, hub) |
| `coverage_params.csv` | Min open plants per region and category (strategic policy, configurable) |

---

## Optimisation model (OM scenario)

### Objective

```
min  Σ_{i,p,h}  Alloc[i,p,h] × LCmult[p,h] × UnitCost[i]        ← landed cost
   + Σ_p         FixedCost[p] × LifecycleYears × OpenGlobal[p]   ← plant overhead
   + α × mean(UnitCost) × Σ_{HistFlow=0}  Alloc[i,p,h]           ← non-historical penalty
```

**Landed cost** is a standard supply-chain term: the total per-unit cost of moving a
product from factory to receiving hub. It bundles transportation, customs/duties,
inventory cost-of-capital, and distribution surcharges into a single multiplier:
`LCmult = 1 + AdderPct`.

`FixedCost[p]` is the annual plant overhead (line setup, qualification, fixed staffing)
that is NOT already embedded in the per-unit landed cost. Including it prevents the
model from opening many plants for trivially small volumes.

`α` controls the preference for historical routes. At α=0.01 the penalty equals
approximately a 1% LC difference per unit — enough to prefer history when costs are
similar, not enough to block a genuinely better route.

### Constraints

| ID | Name | Type | Description |
|----|------|------|-------------|
| C1 | Demand satisfaction | Hard | All lifecycle demand per product and hub must be fully served |
| C2 | Route feasibility | Hard | No flow on forbidden trade corridors (RouteAllowed = 0) |
| C3 | Plant feasibility | Hard | Routes can only be used if the plant is open for that product |
| C4 | Allocation gating | Hard | Volume only flows on open routes (big-M link) |
| C5 | Min production | Hard | If a plant is open for a product, total volume ≥ MinProd_ifOpen |
| C6 | Hub coverage | Hard | Every (product, hub) pair with demand > 0 must have ≥ 1 supplying route |
| C7 | Regional / category coverage | Optional | Min open plants per region and category — configurable per scenario run |
| C8 | Capacity | Placeholder | Non-binding in POC v2 (set to 999,999); schema ready for activation |

---

## Example results (default dummy data)

```
KPI                            B1-Nearest  B2-Baseline  OM-Optimised  GM-RegMix
──────────────────────────────────────────────────────────────────────────────
Landed Cost (MSEK)               1532.6       1553.6        1539.5      1644.0
Fixed Plant Cost (MSEK)            48.0         16.0          22.0        54.0
Total Cost (MSEK)                1580.6       1569.6        1561.5      1698.0
Cost / Unit                      18,690       18,947        18,774      20,707
Cost Avoidance vs B2              -11.0           —           +8.1       -128.4
# Open Plants                         4            1             2           4
New-Route Alerts                      1            5             4           5
```

OM opens 2 plants and saves **8.1 MSEK vs the B2 single-plant baseline** over the
lifecycle. The Regional Mix (30% Region_A / 70% Region_B) costs more because forcing
volume into Region_A for hubs without allowed Region_A routes triggers fallbacks.

---

## Dummy data design

The fictional dataset is designed to exercise all model features:

| Entity | Description |
|--------|-------------|
| 8 plants | Spread across 4 regions (Region_A through Region_D); mix of OWN and EXT category |
| 2 products | Product_A (high-volume, 3 active hubs) and Product_B (broader coverage, 4 active hubs) |
| 7 hubs | Two hubs have zero demand for both pilot products — realistic coverage gaps |
| 8 forbidden corridors | Region_A plants cannot serve Region_C or Region_D hubs — simulates geopolitical restrictions |
| 9 historical routes | Pre-defined as a sensible "today's network"; OM may deviate and generate new-route alerts |

---

## Connecting real data

Replace the dummy CSVs in `data/` with real data warehouse exports. No code changes
are required as long as column names match the schemas in `generate_data.py`.
Each file has a docstring describing the expected source for every column.

Three things to confirm before connecting real data:

1. **Demand source** — which forecasting system to use and how to aggregate to lifecycle volume per hub
2. **MinProd_ifOpen values** — collect realistic minimum annual production volumes per plant from operations teams
3. **Historical flow data** — identify the data warehouse table and confirm the (ProductID, PlantID, Hub, HistFlow) schema

---

## Dependencies

```
pulp>=2.7
pandas>=2.0
numpy>=1.24
streamlit>=1.35
plotly>=5.20
```

Solver: CBC (bundled with PuLP). No external solver installation required.

---

## Known simplifications (POC v2 vs production)

| Dimension | POC v2 | Production |
|-----------|--------|------------|
| Products | 2 fictional products | All products across all product families |
| Plants | 8 (fictional IDs) | All active production plants |
| Time horizon | Single lifecycle aggregate | Multi-period (annual rolling) |
| Echelon | Plant → hub only | Multi-echelon with vendor and customer layers |
| Capacity | Placeholder (non-binding) | Hard constraints from industrial planning |
| Fixed plant cost | Approximate illustrative bands | Precise values from finance |
| Allocation | Continuous | Integer with per-route MOQ if needed |
| Historical flows | Dummy data; soft penalty + alert | Real data from data warehouse |
| Lead time | Out of scope | Future KPI / constraint |
