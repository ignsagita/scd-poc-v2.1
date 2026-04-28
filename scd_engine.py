"""
scd_engine.py — SCD POC v2  Computation Engine
================================================
Pure computation module. No UI, no file I/O, no terminal output.
Imported by app.py (Streamlit) and scd_poc_v2.py (terminal).

All terminology is generic — no company-specific names.

Column name conventions (aligned to generate_data.py):
  Hub      = receiving distribution hub (one-to-one with a market region)
  Geo      = Region_A | Region_B | Region_C | Region_D
  Category = OWN (company-owned) | EXT (external / contract-manufactured)
"""

from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LIFECYCLE_YEARS = 4
ALPHA_DEFAULT   = 0.01
MILP_TIME_LIMIT = 300

GEOS = ["Region_A", "Region_B", "Region_C", "Region_D"]
CATS = ["OWN", "EXT"]

# CSV schemas — used for upload validation in app.py
SCHEMAS: dict[str, dict] = {
    "plants.csv": {
        "required": ["PlantID", "PlantName", "Geo", "Category",
                     "IsPilotSite", "IsVolumeSite",
                     "MinProd_ifOpen", "FixedCost_MSEK_yr"],
        "numeric":  ["IsPilotSite", "IsVolumeSite",
                     "MinProd_ifOpen", "FixedCost_MSEK_yr"],
        "non_neg":  ["MinProd_ifOpen", "FixedCost_MSEK_yr"],
        "choices":  {
            "Geo":      ["Region_A", "Region_B", "Region_C", "Region_D"],
            "Category": ["OWN", "EXT"],
        },
        "doc": (
            "Plant master. One row per factory. "
            "Geo groups plants by geographic region (Region_A through Region_D). "
            "Category: OWN = company-operated site, EXT = external / contract manufacturer. "
            "FixedCost_MSEK_yr is the annual overhead NOT already embedded in the per-unit "
            "landed cost (line setup, qualification, fixed staffing)."
        ),
    },
    "products.csv": {
        "required": ["ProductID", "ProductFamily", "ProductCategory", "UnitCost_SEK"],
        "numeric":  ["UnitCost_SEK"],
        "non_neg":  ["UnitCost_SEK"],
        "choices":  {},
        "doc": (
            "Product master. UnitCost_SEK is the base manufacturing cost per unit "
            "(ex-works cost, from the company costing system). "
            "ProductCategory determines which landed-cost adder row applies — "
            "products in the same category share the same adder on any given route."
        ),
    },
    "demand.csv": {
        "required": ["ProductID", "Hub", "Demand"],
        "numeric":  ["Demand"],
        "non_neg":  ["Demand"],
        "choices":  {},
        "doc": (
            "Lifecycle demand per product and receiving hub. "
            "Demand = total units over the full lifecycle horizon (typically 3–5 years). "
            "Omit rows with zero demand — the engine treats missing rows as zero and "
            "excludes those product-hub pairs from all constraints. "
            "Source: long-range forecast aggregated to lifecycle horizon."
        ),
    },
    "routes.csv": {
        "required": ["PlantID", "Hub", "ProductCategory", "AdderPct", "LCmult"],
        "numeric":  ["AdderPct", "LCmult"],
        "non_neg":  ["AdderPct"],
        "choices":  {},
        "doc": (
            "Landed-cost adder per (plant, hub, product category). "
            "Landed cost = total per-unit cost of moving the product from factory to hub. "
            "It bundles transportation, customs/duties, inventory cost-of-capital, and "
            "distribution surcharges into a single multiplier. "
            "LCmult = 1 + AdderPct (e.g. AdderPct=0.18 → LCmult=1.18 → 18% added to base cost). "
            "Only include ALLOWED routes. Forbidden routes go in route_allowed.csv with 0."
        ),
    },
    "route_allowed.csv": {
        "required": ["PlantID", "Hub", "RouteAllowed"],
        "numeric":  ["RouteAllowed"],
        "non_neg":  [],
        "choices":  {},
        "doc": (
            "Trade-corridor feasibility. 1 = allowed, 0 = FORBIDDEN for ALL products. "
            "Reasons for forbidden: geopolitical restriction, tariff barrier, "
            "no established logistics path, or regulatory constraint. "
            "This flag is product-agnostic — if a plant→hub route is forbidden, "
            "it is forbidden for every product on that corridor. "
            "Assign a data owner and refresh cadence; stale data causes infeasibility."
        ),
    },
    "hubs.csv": {
        "required": ["Hub", "HubName", "HubGeo", "Region"],
        "numeric":  [],
        "non_neg":  [],
        "choices":  {},
        "doc": (
            "Hub master. One row per receiving hub. "
            "Used for display and reporting only — NOT a model constraint. "
            "Each hub maps one-to-one to a customer-facing market region."
        ),
    },
    "hist_flow.csv": {
        "required": ["ProductID", "PlantID", "Hub", "HistFlow", "HistVolShare"],
        "numeric":  ["HistFlow", "HistVolShare"],
        "non_neg":  ["HistVolShare"],
        "choices":  {},
        "doc": (
            "Historical flow flags. HistFlow=1 means this plant→hub route was used "
            "historically for this product. HistFlow=0 = new route with no precedent. "
            "Used in two ways: (1) soft penalty in the OM objective — the model prefers "
            "historical routes when cost is similar; (2) new-route alerts in the output. "
            "Source in production: historical flow data in the company data warehouse."
        ),
    },
    "coverage_params.csv": {
        "required": ["Type", "Name", "Value"],
        "numeric":  ["Value"],
        "non_neg":  ["Value"],
        "choices":  {
            "Type": ["Geo", "Cat"],
        },
        "doc": (
            "Coverage constraint defaults. "
            "Type=Geo: minimum number of globally open plants per geographic region. "
            "Type=Cat: minimum number of globally open plants per category (OWN / EXT). "
            "These are strategic policy constraints, configurable per scenario run. "
            "They are NOT hard regulatory rules."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCDData — DATA CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

class SCDData:
    """
    Loads and cross-validates all input data. Exposes fast dict lookups
    used throughout the scenario engines.

    Two construction paths:
      SCDData(data_dir)           — load from CSV files on disk
      SCDData.from_dataframes(d)  — build from dict of DataFrames (upload path)
    """

    REQUIRED_FILES = list(SCHEMAS.keys())

    def __init__(self, data_dir: Path):
        self._dir = Path(data_dir)
        missing = [f for f in self.REQUIRED_FILES
                   if not (self._dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing input files in {self._dir}/:\n  "
                + "\n  ".join(missing)
                + "\n\nRun:  python generate_data.py"
            )
        dfs = {f: pd.read_csv(self._dir / f) for f in self.REQUIRED_FILES}
        self._init_from_dfs(dfs)

    @classmethod
    def from_dataframes(cls, dfs: dict[str, pd.DataFrame]) -> "SCDData":
        """Build SCDData from a dict of DataFrames keyed by filename."""
        instance = cls.__new__(cls)
        instance._dir = Path(__file__).parent / "data"
        merged: dict[str, pd.DataFrame] = {}
        for fname in cls.REQUIRED_FILES:
            if fname in dfs:
                merged[fname] = dfs[fname]
            elif (instance._dir / fname).exists():
                merged[fname] = pd.read_csv(instance._dir / fname)
            else:
                raise FileNotFoundError(
                    f"{fname} not found in uploads or in data/ directory."
                )
        instance._init_from_dfs(merged)
        return instance

    def _init_from_dfs(self, dfs: dict[str, pd.DataFrame]):
        self.plants   = dfs["plants.csv"]
        self.products = dfs["products.csv"]
        self.demand   = dfs["demand.csv"]
        self.routes   = dfs["routes.csv"]
        self.allowed  = dfs["route_allowed.csv"]
        self.hubs     = dfs["hubs.csv"]
        self.hist     = dfs["hist_flow.csv"]
        self.coverage = dfs["coverage_params.csv"]
        self._validate()
        self._build_lookups()

    def _validate(self):
        errs = []
        known_prods  = set(self.products.ProductID)
        known_plants = set(self.plants.PlantID)
        known_hubs   = set(self.hubs.Hub)

        bad = set(self.demand.ProductID) - known_prods
        if bad:
            errs.append(f"demand.csv references unknown ProductIDs: {bad}")

        bad = set(self.routes.PlantID) - known_plants
        if bad:
            errs.append(f"routes.csv references unknown PlantIDs: {bad}")

        bad = set(self.demand.Hub) - known_hubs
        if bad:
            errs.append(f"demand.csv references unknown Hubs: {bad}")

        ra_pairs = set(zip(self.allowed.PlantID, self.allowed.Hub))
        rt_pairs = set(zip(self.routes.PlantID,  self.routes.Hub))
        missing  = rt_pairs - ra_pairs
        if missing:
            errs.append(
                f"route_allowed.csv is missing {len(missing)} (PlantID, Hub) pairs "
                "that appear in routes.csv"
            )
        if errs:
            raise ValueError("Data validation failed:\n  " + "\n  ".join(errs))

    def _build_lookups(self):
        self.plant_geo  = dict(zip(self.plants.PlantID, self.plants.Geo))
        self.plant_cat  = dict(zip(self.plants.PlantID, self.plants.Category))
        self.plant_name = dict(zip(self.plants.PlantID, self.plants.PlantName))
        self.min_prod   = dict(zip(self.plants.PlantID, self.plants.MinProd_ifOpen))
        self.fixed_cost = dict(zip(self.plants.PlantID, self.plants.FixedCost_MSEK_yr))

        self.unit_cost  = dict(zip(self.products.ProductID, self.products.UnitCost_SEK))
        self.mean_uc    = float(np.mean(list(self.unit_cost.values())))

        # demand_map: (ProductID, Hub) → Demand
        self.demand_map = {
            (r.ProductID, r.Hub): r.Demand
            for _, r in self.demand.iterrows()
        }
        # active pairs where demand > 0
        self.active_ig = [
            (r.ProductID, r.Hub)
            for _, r in self.demand.iterrows() if r.Demand > 0
        ]

        # lc_mult: (PlantID, Hub) → LCmult
        self.lc_mult = {
            (r.PlantID, r.Hub): r.LCmult
            for _, r in self.routes.iterrows()
        }

        # route_allowed: (PlantID, Hub) → 0/1
        self.route_allowed = {
            (r.PlantID, r.Hub): int(r.RouteAllowed)
            for _, r in self.allowed.iterrows()
        }

        # hist_flow: (ProductID, PlantID, Hub) → 0/1
        self.hist_flow = {
            (r.ProductID, r.PlantID, r.Hub): int(r.HistFlow)
            for _, r in self.hist.iterrows()
        }

        # coverage parameters
        self.geo_params = {
            r.Name: int(r.Value)
            for _, r in self.coverage.iterrows() if r.Type == "Geo"
        }
        self.cat_params = {
            r.Name: int(r.Value)
            for _, r in self.coverage.iterrows() if r.Type == "Cat"
        }

        self.products_list = self.products.ProductID.tolist()
        self.plants_list   = self.plants.PlantID.tolist()
        self.hubs_list     = self.hubs.Hub.tolist()

        # hub display lookups
        self.hub_name   = dict(zip(self.hubs.Hub, self.hubs.HubName))
        self.hub_region = dict(zip(self.hubs.Hub, self.hubs.Region))

        # capacity: (ProductID, PlantID) → cap
        cap_cols = [c for c in self.plants.columns if c.startswith("Cap_")]
        self.capacity: dict[tuple, float] = {}
        for _, row in self.plants.iterrows():
            for col in cap_cols:
                suffix = col.replace("Cap_", "")
                matching = [p for p in self.products_list
                            if suffix in p.replace("_", "")]
                if matching:
                    self.capacity[(matching[0], row.PlantID)] = float(row[col])

    def with_demand_adjustments(self, adjustments: dict[str, float]) -> "SCDData":
        """
        Return a deep-copied SCDData with per-hub demand adjusted by %.
        adjustments = {"Hub_1": 20.0, "Hub_3": -10.0}  (percent change)
        """
        clone = copy.deepcopy(self)
        for (i, h), v in clone.demand_map.items():
            pct = adjustments.get(h, 0.0)
            clone.demand_map[(i, h)] = max(0.0, v * (1 + pct / 100.0))
        clone.active_ig = [
            (i, h) for (i, h), v in clone.demand_map.items() if v > 0
        ]
        return clone

    def summary_dict(self) -> dict:
        return {
            "plants":        len(self.plants_list),
            "products":      self.products_list,
            "hubs":          len(self.hubs_list),
            "active_pairs":  len(self.active_ig),
            "routes_allowed": sum(self.route_allowed.values()),
            "total_demand":  sum(self.demand_map.values()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALLOCATION RECORD BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _alloc_row(product: str, plant: str, hub: str,
               qty: float, data: SCDData) -> dict:
    lc = data.lc_mult.get((plant, hub), 0.0)
    uc = data.unit_cost.get(product, 0.0)
    hf = data.hist_flow.get((product, plant, hub), 0)
    return {
        "ProductID":       product,
        "PlantID":         plant,
        "Hub":             hub,
        "Qty":             qty,
        "LCmult":          lc,
        "UnitCost_SEK":    uc,
        "LandedCost_SEK":  qty * lc * uc,
        "HistFlow":        hf,
        "IsNewRoute":      1 - hf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCENARIO ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def run_b1(data: SCDData, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    B1 — Nearest-Plant Baseline.
    For each (product, hub) with demand > 0: assign all demand to the plant
    with the lowest LCmult among allowed routes.
    Tie-break: alphabetical by PlantID.
    Returns (alloc_df, alert_messages).
    """
    rows, alerts = [], []
    for prod, hub in data.active_ig:
        demand = data.demand_map[(prod, hub)]
        candidates = sorted([
            (data.lc_mult.get((p, hub), 999.0), p)
            for p in data.plants_list
            if data.route_allowed.get((p, hub), 0) == 1
               and (p, hub) in data.lc_mult
        ])
        if not candidates:
            alerts.append(
                f"No allowed route for {prod} / {hub} — "
                f"{demand:,.0f} units unserved"
            )
            continue
        rows.append(_alloc_row(prod, candidates[0][1], hub, demand, data))
    return pd.DataFrame(rows), alerts


def run_b2(data: SCDData, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    B2 — Baseline Plant.
    Assign all demand to a single user-selected baseline plant.
    Falls back to cheapest allowed plant if baseline route is forbidden.
    Returns (alloc_df, alert_messages).
    """
    baseline = config.get("baseline_plant", {})
    rows, alerts = [], []
    for prod, hub in data.active_ig:
        demand = data.demand_map[(prod, hub)]
        plant  = baseline.get(prod, data.plants_list[0])
        if data.route_allowed.get((plant, hub), 0) == 1:
            rows.append(_alloc_row(prod, plant, hub, demand, data))
        else:
            alerts.append(
                f"{plant} has no allowed route to {hub} for {prod} "
                f"— redirected {demand:,.0f} units to cheapest allowed plant"
            )
            candidates = sorted([
                (data.lc_mult.get((p, hub), 999.0), p)
                for p in data.plants_list
                if data.route_allowed.get((p, hub), 0) == 1
            ])
            if candidates:
                rows.append(_alloc_row(prod, candidates[0][1], hub, demand, data))
    return pd.DataFrame(rows), alerts


def run_om(data: SCDData, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    OM — MILP Optimisation.
    Objective: landed cost + fixed plant overhead + α-penalty for non-historical routes.
    Returns (alloc_df, solve_metadata).
    """
    alpha    = config.get("alpha",        ALPHA_DEFAULT)
    geo_par  = config.get("geo_params",   data.geo_params)
    cat_par  = config.get("cat_params",   data.cat_params)
    use_c7   = config.get("use_coverage", True)

    P = data.plants_list
    I = data.products_list
    H = data.hubs_list
    M_big = float(sum(data.demand_map.values())) * 2

    mdl = pulp.LpProblem("SCD_OM", pulp.LpMinimize)

    Open       = pulp.LpVariable.dicts(
        "Open",  [(i,p) for i in I for p in P], cat="Binary")
    RouteOpen  = pulp.LpVariable.dicts(
        "RO",    [(i,p,h) for i in I for p in P for h in H], cat="Binary")
    Alloc      = pulp.LpVariable.dicts(
        "Alloc", [(i,p,h) for i in I for p in P for h in H], lowBound=0)
    OpenGlobal = pulp.LpVariable.dicts("OG", P, cat="Binary")

    # Objective
    obj_terms = []
    for i in I:
        uc = data.unit_cost[i]
        for p in P:
            for h in H:
                lc    = data.lc_mult.get((p, h), 0.0)
                hf    = data.hist_flow.get((i, p, h), 0)
                pen   = 0.0 if hf else alpha * data.mean_uc
                coeff = lc * uc + pen
                if coeff > 0:
                    obj_terms.append(coeff * Alloc[(i, p, h)])

    for p in P:
        fc_sek = data.fixed_cost.get(p, 0.0) * 1_000_000 * LIFECYCLE_YEARS
        if fc_sek > 0:
            obj_terms.append(fc_sek * OpenGlobal[p])

    mdl += pulp.lpSum(obj_terms), "TotalObjective"

    # C1 — Demand satisfaction
    for i, h in data.active_ig:
        mdl += (pulp.lpSum(Alloc[(i,p,h)] for p in P)
                == data.demand_map[(i, h)]), f"C1_{i}_{h}"

    # C2 — Route feasibility
    for i in I:
        for p in P:
            for h in H:
                mdl += (RouteOpen[(i,p,h)]
                        <= data.route_allowed.get((p, h), 0)), f"C2_{i}_{p}_{h}"

    # C3 — Plant feasibility
    for i in I:
        for p in P:
            for h in H:
                mdl += RouteOpen[(i,p,h)] <= Open[(i,p)], f"C3_{i}_{p}_{h}"

    # C4 — Allocation gating
    for i in I:
        for p in P:
            for h in H:
                mdl += (Alloc[(i,p,h)]
                        <= M_big * RouteOpen[(i,p,h)]), f"C4_{i}_{p}_{h}"

    # C5 — Min production if open
    for i in I:
        for p in P:
            mp = data.min_prod.get(p, 0)
            mdl += (pulp.lpSum(Alloc[(i,p,h)] for h in H)
                    >= mp * Open[(i,p)]), f"C5_{i}_{p}"

    # C6 — Hub coverage
    for i, h in data.active_ig:
        mdl += (pulp.lpSum(RouteOpen[(i,p,h)] for p in P)
                >= 1), f"C6_{i}_{h}"

    # OpenGlobal links
    for p in P:
        for i in I:
            mdl += OpenGlobal[p] >= Open[(i,p)], f"OG_lb_{p}_{i}"
        mdl += (OpenGlobal[p]
                <= pulp.lpSum(Open[(i,p)] for i in I)), f"OG_ub_{p}"

    # C7 — Geo / Category coverage (optional)
    if use_c7:
        for geo, req in geo_par.items():
            if req > 0:
                pts = [p for p in P if data.plant_geo.get(p) == geo]
                if pts:
                    mdl += (pulp.lpSum(OpenGlobal[p] for p in pts)
                            >= req), f"C7_geo_{geo}"
        for cat, req in cat_par.items():
            if req > 0:
                pts = [p for p in P if data.plant_cat.get(p) == cat]
                if pts:
                    mdl += (pulp.lpSum(OpenGlobal[p] for p in pts)
                            >= req), f"C7_cat_{cat}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_TIME_LIMIT)
    mdl.solve(solver)
    status = pulp.LpStatus[mdl.status]
    obj    = pulp.value(mdl.objective)

    if mdl.status not in (1, -2):
        raise RuntimeError(
            f"MILP returned status '{status}'.\n"
            "Possible causes: coverage constraints too tight for the given network, "
            "or a hub has no allowed routes from any plant. "
            "Check route_allowed.csv and coverage_params.csv."
        )

    rows = []
    for i in I:
        for p in P:
            for h in H:
                qty = pulp.value(Alloc[(i,p,h)]) or 0.0
                if qty > 0.01:
                    rows.append(_alloc_row(i, p, h, qty, data))

    meta = {
        "status":      status,
        "objective":   obj,
        "open_global": {p: round(pulp.value(OpenGlobal[p]) or 0) for p in P},
        "alpha":       alpha,
        "alerts":      [],
    }
    return pd.DataFrame(rows), meta


def run_gm(data: SCDData, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    GM — Geo-Mix.
    User specifies target shares per region (must sum to 100%).
    For each (product, hub, region): cheapest allowed plant in that region.
    Falls back to global cheapest if no plant in region has allowed route.
    Returns (alloc_df, alert_messages).
    """
    geo_mix = config.get("geo_mix", {g: 0 for g in GEOS})
    total   = sum(geo_mix.values())
    if abs(total - 100) > 0.1:
        raise ValueError(
            f"Geo-mix percentages must sum to 100 (got {total:.1f})."
        )

    rows, alerts = [], []
    for prod, hub in data.active_ig:
        demand = data.demand_map[(prod, hub)]
        for geo in GEOS:
            pct = geo_mix.get(geo, 0.0)
            if pct <= 0:
                continue
            target = demand * pct / 100.0
            candidates = sorted([
                (data.lc_mult.get((p, hub), 999.0), p)
                for p in data.plants_list
                if data.plant_geo.get(p) == geo
                   and data.route_allowed.get((p, hub), 0) == 1
            ])
            if candidates:
                rows.append(
                    _alloc_row(prod, candidates[0][1], hub, target, data))
            else:
                alerts.append(
                    f"No allowed {geo} plant for {prod}/{hub} "
                    f"({pct:.0f}% = {target:,.0f} units) — using cheapest global"
                )
                global_c = sorted([
                    (data.lc_mult.get((p, hub), 999.0), p)
                    for p in data.plants_list
                    if data.route_allowed.get((p, hub), 0) == 1
                ])
                if global_c:
                    rows.append(
                        _alloc_row(prod, global_c[0][1], hub, target, data))

    return pd.DataFrame(rows), alerts


# ══════════════════════════════════════════════════════════════════════════════
# 5. KPI COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(alloc_df: pd.DataFrame, data: SCDData,
                 label: str, meta: dict | None = None,
                 alerts: list[str] | None = None) -> dict:
    """Compute all KPIs from an allocation DataFrame."""
    if alloc_df.empty:
        return {"label": label, "error": "No allocation produced"}

    df = alloc_df.copy()
    df["Geo"] = df.PlantID.map(data.plant_geo)
    df["Cat"] = df.PlantID.map(data.plant_cat)

    total_vol   = df.Qty.sum()
    total_lc    = df.LandedCost_SEK.sum()
    weighted_lc = (df.Qty * df.LCmult).sum() / total_vol if total_vol else 0.0
    cost_pu     = total_lc / total_vol if total_vol else 0.0

    open_plants = sorted(df[df.Qty > 0].PlantID.unique())
    fixed_total = sum(
        data.fixed_cost.get(p, 0.0) * 1_000_000 * LIFECYCLE_YEARS
        for p in open_plants
    )

    geo_vols = df.groupby("Geo")["Qty"].sum()
    cat_vols = df.groupby("Cat")["Qty"].sum()
    geo_pct  = {g: geo_vols.get(g, 0.0) / total_vol * 100 for g in GEOS}
    cat_pct  = {c: cat_vols.get(c, 0.0) / total_vol * 100 for c in CATS}

    # Per-plant summary
    plant_rows = []
    for p in data.plants_list:
        sub     = df[df.PlantID == p]
        is_open = p in open_plants
        row = {
            "PlantID":           p,
            "PlantName":         data.plant_name.get(p, p),
            "Geo":               data.plant_geo.get(p),
            "Category":          data.plant_cat.get(p),
            "IsOpen":            int(is_open),
            "TotalQty":          round(sub.Qty.sum()),
            "LandedCost_MSEK":   sub.LandedCost_SEK.sum() / 1e6,
            "FixedCost_MSEK":    (data.fixed_cost.get(p, 0.0) * LIFECYCLE_YEARS
                                  if is_open else 0.0),
        }
        for prod in data.products_list:
            safe = prod.replace(" ", "_")
            row[f"Qty_{safe}"] = round(sub[sub.ProductID == prod].Qty.sum())
        plant_rows.append(row)

    # New-route alerts
    new_routes = df[(df.HistFlow == 0) & (df.Qty > 0)].copy()

    # Consolidation alerts
    consol = []
    for prod in data.products_list:
        total_d = sum(v for (i, h), v in data.demand_map.items() if i == prod)
        min_p   = min(data.min_prod.values()) if data.min_prod else 0
        if 0 < total_d < min_p:
            consol.append({"ProductID": prod,
                           "TotalDemand": total_d,
                           "MinProd_threshold": min_p})

    return {
        "label":             label,
        "alloc_df":          df,
        "plant_df":          pd.DataFrame(plant_rows),
        "new_routes_df":     new_routes,
        "consol_df":         pd.DataFrame(consol),
        "total_vol":         total_vol,
        "total_lc_sek":      total_lc,
        "fixed_cost_sek":    fixed_total,
        "total_cost_sek":    total_lc + fixed_total,
        "cost_per_unit_sek": cost_pu,
        "weighted_lc_pct":   weighted_lc * 100,
        "n_open_plants":     len(open_plants),
        "open_plants":       open_plants,
        "geo_pct":           geo_pct,
        "cat_pct":           cat_pct,
        "cost_by_prod":      df.groupby("ProductID")["LandedCost_SEK"].sum().to_dict(),
        "vol_by_prod":       df.groupby("ProductID")["Qty"].sum().to_dict(),
        "milp_meta":         meta,
        "run_alerts":        alerts or [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataframe(filename: str, df: pd.DataFrame) -> list[str]:
    """
    Validate an uploaded DataFrame against its schema.
    Returns a list of human-readable error strings (empty = valid).
    """
    schema = SCHEMAS.get(filename)
    if schema is None:
        return [f"Unknown file: {filename}"]

    errors: list[str] = []

    missing_cols = [c for c in schema["required"] if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors

    for col in schema.get("numeric", []):
        if col in df.columns:
            non_num = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_num:
                errors.append(
                    f"Column '{col}': {non_num} non-numeric value(s)"
                )

    for col in schema.get("non_neg", []):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            neg  = (vals < 0).sum()
            if neg:
                errors.append(
                    f"Column '{col}': {neg} negative value(s) (must be ≥ 0)"
                )

    for col, allowed in schema.get("choices", {}).items():
        if col in df.columns:
            bad = df[~df[col].astype(str).isin(
                [str(a) for a in allowed])][col].unique()
            if len(bad):
                errors.append(
                    f"Column '{col}': unexpected values {list(bad)[:5]}. "
                    f"Allowed: {allowed}"
                )

    return errors
