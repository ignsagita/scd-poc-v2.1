# SCD POC v2.1 — Supply Chain Design Optimisation

A Python proof-of-concept for Ericsson's Supply Chain Design (SCD) decision support tool. Given lifecycle demand per product and hub, it finds the optimal allocation of production volume across plants to minimise total landed cost — subject to geopolitical route rules, minimum production constraints, and strategic coverage requirements.

> **Status:** POC v2 — dummy data aligned to the production Snowflake schema. Ready for real-data swap-in once Snowflake tables are confirmed.

---

## Live app

Deployed on Streamlit Community Cloud — no installation needed:

**[🔗 Open SCD POC v2 →](https://scd-simulation-poc.streamlit.app)**

> Replace this link with your Streamlit Cloud URL after deployment (see [Deploying](#deploying-to-streamlit-cloud) below).

---

## What this solves

The core business question is: **how does a change in supply chain setup impact landed cost at the product level?**

The tool simulates four scenario types and compares them side by side:

| ID | Name | Logic |
|----|------|-------|
| **B1** | Nearest-Plant Baseline | Assigns each hub's demand to the cheapest allowed plant. Deterministic. |
| **B2** | Baseline Plant | All demand assigned to one user-selected reference plant. Used as the TPI cost-avoidance benchmark. |
| **OM** | Optimised (MILP) | Solves a mixed-integer linear programme to minimise total landed cost + fixed plant overhead + soft penalty for non-historical routes. |
| **GM** | Geo-Mix | User specifies a target volume split by geography (e.g. 30% CN / 70% EU). Cheapest plant within each geo is used. |

Outputs feed directly into Supply Chain Board (SCB) decisions: which plant for TPI/NPI, single-site vs multi-site risk, and cost avoidance vs the current network.

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

**Terminal-only alternative** (no Streamlit needed):

```bash
python scd_poc_v2.py          # interactive menu
python scd_poc_v2.py --all    # run all 4 scenarios with defaults
python scd_poc_v2.py --regen  # regenerate data then run
```

---

## Deploying to Streamlit Cloud

Streamlit Community Cloud is free and requires only a public GitHub repo.

**Prerequisites:** a GitHub account and a Streamlit Community Cloud account (sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub).

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

First deploy takes 2–3 minutes while packages install. After that you get a shareable public URL.

**Free tier notes:**
- The app sleeps after inactivity — open it yourself a minute before any demo to wake it up
- Uploads are session-only (reset on page reload); the `data/` files in the repo are always the default dataset
- Memory limit is 1 GB, well within what this model needs

---

## Project structure

```
scd_v2/
│
├── app.py                  # Streamlit UI (4 tabs: Data, Configuration, Results, About)
├── scd_engine.py           # Computation engine — no UI code, importable standalone
├── generate_data.py        # Generates all 8 input CSVs (schema contract for data engineering)
├── scd_poc_v2.py           # Terminal-only pipeline (alternative to the Streamlit app)
├── requirements.txt
│
├── .streamlit/
│   └── config.toml         # Theme and upload size settings
│
├── data/                   # Input CSVs — replace with real Snowflake exports
│   ├── plants.csv
│   ├── products.csv
│   ├── demand.csv
│   ├── routes.csv
│   ├── route_allowed.csv
│   ├── cg_map.csv
│   ├── hist_flow.csv
│   └── coverage_params.csv
│
└── results/                # Output CSVs and figures (terminal run only; gitignored)
```

### Key design decision

`scd_engine.py` contains all computation with zero UI code. `app.py` is UI only and imports from the engine. This means the engine can be tested independently, used from the terminal, or plugged into a different UI without modification.

---

## App features

### 📂 Data tab
- Download any of the 8 input CSV templates, each with an embedded column guide
- Upload modified files — each is validated on upload with row-level error messages (wrong column names, negative values, invalid category codes)
- Unuploaded files fall back to the built-in defaults automatically
- Preview the current active dataset in a table

### ⚙️ Configuration tab
- **What-if demand sliders** — adjust lifecycle demand per hub by ±50% before running, to stress-test the network
- Select which scenarios to run (B1 / B2 / OM / GM, any combination)
- **B2:** dropdown to select the baseline reference plant per product
- **OM:** alpha slider with a live explanation of what the penalty means in SEK/unit and LC% equivalent; geo/category coverage constraints with per-geography number inputs
- **GM:** four percentage inputs with automatic sum validation (blocks run if total ≠ 100%)
- Explicit **Run button** — no recomputation on every slider touch

### 📊 Results tab
- KPI metric cards per scenario: total cost, landed cost, fixed cost, cost per unit, weighted LC%, open plants, TPI avoidance vs B2, new-route alert count
- **Stacked bar charts** (Plotly): cost breakdown (landed vs fixed) and geo volume distribution
- **Diff Sankey diagram**: shows OM vs a user-selected baseline with colour-coded links — blue (continuing routes), green (new routes proposed by OM), red (routes dropped by OM)
- **Alpha sensitivity panel**: re-runs OM at 5 α values and plots cost, open plants, and new-route alerts as α changes
- **Alerts panel** per scenario: new-route alerts (tabular, colour-scaled by volume) and consolidation candidate alerts
- Full allocation and plant summary tables with conditional formatting
- Download all results as a single ZIP, or download individual scenario CSVs

### 📖 About tab
- LaTeX objective function
- Constraint reference table (C1–C8)
- Data lineage table (file → Snowflake source → data owner)
- Open points before production data connection
- POC vs production comparison table

---

## Input files

All files in `data/` are generated by `generate_data.py` with realistic dummy values. The column names and data types are the schema contract — the data engineer delivers the same structure from Snowflake.

| File | Description | Snowflake source |
|------|-------------|------------------|
| `plants.csv` | Plant master: SCN code, geo, category, min production, fixed cost, capacity | Plant/site master → SCN table |
| `products.csv` | Product master: product ID, area, category, unit cost (TK) | CRM cost tree / product costing |
| `demand.csv` | Lifecycle demand per product and hub | MAF + LRFP (source TBD with SuPM) |
| `routes.csv` | LC adder % and LCmult per (plant, hub, product category) | Henrik's landed cost tool |
| `route_allowed.csv` | Geopolitical feasibility flag per (plant, hub) — product-agnostic | ERIDOC BNEW-24:038086Uen (owner: Clovis Hiroshi Kawai) |
| `cg_map.csv` | Customer group → market area → hub name mapping | MA/hub master data |
| `hist_flow.csv` | Historical flow flags per (product, plant, hub) | Amrith's SCD flow description (Snowflake table TBD) |
| `coverage_params.csv` | Min open plants per geography and category | User / scenario configuration |

> **Plant IDs** use the SCN format `ERI/{country}-{site}` (e.g. `ERI/PL-TC`, `ERI/CN-ESS`) — the same identifier used in SAIDA and GFT.

---

## Optimisation model (OM scenario)

### Objective

```
min  Σ_{i,p,g}  Alloc[i,p,g] × LCmult[p,g] × UnitCost[i]       ← landed cost
   + Σ_p         FixedCost[p] × LifecycleYears × OpenGlobal[p]  ← plant overhead
   + α × mean(UnitCost) × Σ_{HistFlow=0}  Alloc[i,p,g]          ← non-historical penalty
```

- `LCmult[p,g]` = 1 + Adder[p,g]. Flow-based: same for all products on a given plant-hub route.
- `UnitCost[i]` = base TK per unit. Differs per product, not per route.
- `FixedCost[p]` = annual plant overhead not already embedded in LCmult (line setup, qualification, staffing above flow cost).
- `α` = penalty weight scalar. At α=0.01 the penalty equals ~1% LC difference per unit on non-historical routes — enough to prefer history when costs are similar, not enough to block a better route.

### Constraints

| ID | Constraint | Type | Description |
|----|-----------|------|-------------|
| C1 | Demand satisfaction | Hard | All lifecycle demand per product and hub must be fully served |
| C2 | Route feasibility | Hard | No flow on geopolitically forbidden routes (`RouteAllowed = 0`) |
| C3 | Plant feasibility | Hard | Routes can only be used if the plant is open for that product |
| C4 | Allocation gating | Hard | Volume can only flow on open routes (big-M link) |
| C5 | Min production | Hard | If a plant is open, it must produce at least `MinProd_ifOpen` units |
| C6 | CG coverage | Hard | Every hub with demand > 0 must have at least one supplying route |
| C7 | Geo/category coverage | Optional | Min open plants per geography and category — configurable per scenario |
| C8 | Capacity | Placeholder | Non-binding in POC v2 (`Cap = 999,999`); schema ready for activation |

---

## Example results (default dummy data)

```
KPI                              B1-Nearest   B2-Baseline  OM-Optimised  GM-GeoMix
─────────────────────────────────────────────────────────────────────────────────
Landed Cost (MSEK)                  1532.6        1553.6        1539.5       1596.0
Fixed Plant Cost (MSEK)               48.0          16.0          22.0         63.2
TOTAL Cost incl. Fixed (MSEK)       1580.6        1569.6        1561.5       1659.2
Cost per Unit (SEK)                 18,690        18,947        18,774       19,464
TPI Avoidance vs B2 (MSEK)           -11.0             —          +8.2        -89.6
# Open Plants                             4             1             2            5
New-Route Alerts                          1             5             4            6
```

OM opens 2 plants (TC Poland + WX Nanjing) and saves **8.2 MSEK vs the B2 single-plant baseline** over the lifecycle. The Geo-Mix scenario (30% CN / 70% EU) costs 98 MSEK more than OM because forcing CN volume into routes without direct logistics corridors (US, LatAm) triggers fallbacks and opens more plants.

---

## Pilot products

| Product ID | Description | Product Category |
|-----------|-------------|-----------------|
| `KRD901252/11` | AIR 6419 B77D | Radio — Baseband |
| `KRC4464B1B3B7` | Radio 4464 B1/B3/B7 | Radio — Baseband |

Both products share the same LC adder on any given plant-hub route (same ProductCategory). `UnitCost` differs between the two.

---

## Connecting real data

Replace the dummy CSVs in `data/` with real Snowflake exports. No code changes required as long as column names match. Each file in `generate_data.py` has a docstring describing the expected Snowflake source for each column.

Three open points that must be resolved before production data can be connected:

1. **Demand source** — confirm LRFP vs MAF (or combination) for lifecycle volume per hub. Owner: SuPM team.
2. **MinProd_ifOpen values** — collect realistic minimums from factory SMEs. The placeholder values (500–2,000 units) directly drive the consolidation alert threshold.
3. **Historical flow data** — identify the Snowflake table for `hist_flow.csv`. Owner: Amrith.

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

## Scope and known simplifications

| Dimension | POC v2 | Production (future) |
|-----------|--------|---------------------|
| Products | 2 Radio (dummy data) | ~60 products/year across Radio, RAN Compute, Site Material |
| Plants | 8 (dummy SCN codes) | All ESS/EMS plants with existing TPI flows |
| Time horizon | Single lifecycle aggregate | Multi-period (annual rolling) |
| Echelon | Plant → hub only | Multi-echelon with vendor and customer layers |
| Capacity | Placeholder (non-binding) | Hard constraints from industrial planning |
| Fixed plant cost | Approximate cost bands | Precise values from industrial finance |
| Allocation | Continuous (UI rounds to whole units) | Integer with per-route MOQ if needed |
| Historical flows | Dummy data; soft penalty + alert | Real data from Snowflake (Amrith) |
| Lead time | Out of scope | Future KPI / constraint |
