#!/usr/bin/env python3
"""
SCD POC v2 — Supply Chain Design Optimisation Pipeline
=======================================================
Requirements v1.0  |  Pilot: KRD901252/11  &  KRC4464B1B3B7

SCENARIOS
  B1 – Nearest-Plant Baseline   (deterministic, lowest LCmult per CG)
  B2 – Baseline Plant           (deterministic, single user-selected plant)
  OM – Optimised (MILP)         (cost + fixed-plant cost + historical penalty)
  GM – Geo-Mix                  (deterministic, user-specified geo shares)

OBJECTIVE (OM only)
  min  Σ Alloc[i,p,g] × LCmult[p,g] × UnitCost[i]          ← landed cost
     + Σ FixedCost[p] × LIFECYCLE_YEARS × OpenGlobal[p]     ← plant overhead
     + α × mean(UnitCost) × Σ Alloc[i,p,g] {HistFlow=0}    ← non-hist. penalty

USAGE
  python scd_poc_v2.py           → interactive terminal menu
  python scd_poc_v2.py --all     → run all 4 scenarios with defaults, no prompts
"""

import sys, os, textwrap, argparse
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ROOT           = Path(__file__).parent
DATA           = ROOT / "data"
RESULTS        = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LIFECYCLE_YEARS = 4          # horizon for scaling annual FixedCost → lifecycle
ALPHA_DEFAULT   = 0.01       # penalty weight (≈1% LC equivalent per non-hist. unit)
MILP_TIME_LIMIT = 300        # CBC solver time limit (seconds)

GEOS = ["CN", "EU", "LAT", "US"]
CATS = ["ESS", "EMS"]

# Colour palette for consistent figures
PRODUCT_COLORS  = {"KRD901252/11": "#1565C0", "KRC4464B1B3B7": "#2E7D32"}
SCENARIO_COLORS = {"B1": "#546E7A", "B2": "#78909C", "OM": "#1976D2", "GM": "#7B1FA2"}
GEO_COLORS      = {"CN": "#EF5350", "EU": "#42A5F5", "LAT": "#66BB6A", "US": "#FFA726"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

class SCDData:
    """
    Loads and cross-validates all 8 input CSVs.
    Raises RuntimeError (with clear diagnostics) on any missing file or
    schema violation — no silent failures.
    """

    REQUIRED_FILES = [
        "plants.csv", "products.csv", "demand.csv",
        "routes.csv", "route_allowed.csv", "cg_map.csv",
        "hist_flow.csv", "coverage_params.csv",
    ]

    def __init__(self, data_dir: Path = DATA):
        self._dir = data_dir
        self._check_files()
        self._load()
        self._validate()
        self._build_lookups()

    def _check_files(self):
        missing = [f for f in self.REQUIRED_FILES if not (self._dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"Missing input files in {self._dir}/:\n  " +
                "\n  ".join(missing) +
                "\nRun:  python generate_data.py"
            )

    def _load(self):
        d = self._dir
        self.plants   = pd.read_csv(d / "plants.csv")
        self.products = pd.read_csv(d / "products.csv")
        self.demand   = pd.read_csv(d / "demand.csv")
        self.routes   = pd.read_csv(d / "routes.csv")
        self.allowed  = pd.read_csv(d / "route_allowed.csv")
        self.cg_map   = pd.read_csv(d / "cg_map.csv")
        self.hist     = pd.read_csv(d / "hist_flow.csv")
        self.coverage = pd.read_csv(d / "coverage_params.csv")

    def _validate(self):
        errs = []
        # Products referenced in demand must exist in products.csv
        demand_prods = set(self.demand.ProductID)
        known_prods  = set(self.products.ProductID)
        bad = demand_prods - known_prods
        if bad:
            errs.append(f"demand.csv references unknown ProductIDs: {bad}")
        # Plants referenced in routes must exist in plants.csv
        known_plants = set(self.plants.PlantID)
        route_plants = set(self.routes.PlantID)
        bad = route_plants - known_plants
        if bad:
            errs.append(f"routes.csv references unknown PlantIDs: {bad}")
        # RouteAllowed must cover all (plant, CG) combinations
        ra_pairs = set(zip(self.allowed.PlantID, self.allowed.CG))
        rt_pairs = set(zip(self.routes.PlantID, self.routes.CG))
        missing_ra = rt_pairs - ra_pairs
        if missing_ra:
            errs.append(f"route_allowed.csv missing {len(missing_ra)} (plant,CG) pairs that appear in routes.csv")
        if errs:
            raise RuntimeError("Data validation failed:\n  " + "\n  ".join(errs))

    def _build_lookups(self):
        """Build fast dict lookups used throughout the engine."""
        # Plant attributes
        self.plant_geo  = dict(zip(self.plants.PlantID, self.plants.Geo))
        self.plant_cat  = dict(zip(self.plants.PlantID, self.plants.Category))
        self.plant_name = dict(zip(self.plants.PlantID, self.plants.PlantName))
        self.min_prod   = dict(zip(self.plants.PlantID, self.plants.MinProd_ifOpen))
        self.fixed_cost = dict(zip(self.plants.PlantID, self.plants.FixedCost_MSEK_yr))

        # Product attributes
        self.unit_cost  = dict(zip(self.products.ProductID, self.products.UnitCost_SEK))
        self.mean_uc    = float(np.mean(list(self.unit_cost.values())))

        # Demand: (product, CG) → volume
        self.demand_map = {
            (r.ProductID, r.CG): r.Demand
            for _, r in self.demand.iterrows()
        }
        # Active (product, CG) pairs — where demand > 0
        self.active_ig = [(r.ProductID, r.CG)
                          for _, r in self.demand.iterrows() if r.Demand > 0]

        # LC multiplier: (plant, CG) → LCmult
        self.lc_mult = {
            (r.PlantID, r.CG): r.LCmult
            for _, r in self.routes.iterrows()
        }

        # RouteAllowed: (plant, CG) → 0/1
        self.route_allowed = {
            (r.PlantID, r.CG): int(r.RouteAllowed)
            for _, r in self.allowed.iterrows()
        }

        # HistFlow: (product, plant, CG) → 0/1
        self.hist_flow = {
            (r.ProductID, r.PlantID, r.CG): int(r.HistFlow)
            for _, r in self.hist.iterrows()
        }

        # Coverage params
        self.geo_params = {
            r.Name: int(r.Value)
            for _, r in self.coverage.iterrows() if r.Type == "Geo"
        }
        self.cat_params = {
            r.Name: int(r.Value)
            for _, r in self.coverage.iterrows() if r.Type == "Cat"
        }

        # Sets
        self.products_list = self.products.ProductID.tolist()
        self.plants_list   = self.plants.PlantID.tolist()
        self.cgs_list      = self.cg_map.CG.tolist()

        # Capacity: (product_col_name, plant) → capacity
        cap_cols = [c for c in self.plants.columns if c.startswith("Cap_")]
        self.capacity = {}
        for _, row in self.plants.iterrows():
            for col in cap_cols:
                prod_suffix = col.replace("Cap_", "")
                # Match suffix to full product ID (heuristic)
                matching = [p for p in self.products_list if prod_suffix in p.replace("/","")]
                if matching:
                    self.capacity[(matching[0], row.PlantID)] = row[col]

    def summary(self):
        print(f"  Plants:   {len(self.plants_list)}")
        print(f"  Products: {len(self.products_list)}  → {self.products_list}")
        print(f"  CGs:      {len(self.cgs_list)}  ({len(self.active_ig)} active product-CG pairs)")
        print(f"  Routes:   {sum(self.route_allowed.values())} allowed, "
              f"{sum(1 for v in self.route_allowed.values() if v==0)} forbidden")
        total_demand = sum(self.demand_map.values())
        print(f"  Total lifecycle demand: {total_demand:,.0f} units")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCENARIO ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def _alloc_row(product, plant, cg, qty, data: SCDData):
    """Build one allocation record with all derived fields."""
    lc = data.lc_mult.get((plant, cg), 0.0)
    uc = data.unit_cost.get(product, 0.0)
    hf = data.hist_flow.get((product, plant, cg), 0)
    return {
        "ProductID": product,
        "PlantID":   plant,
        "CG":        cg,
        "Qty":       qty,
        "LCmult":    lc,
        "UnitCost_SEK": uc,
        "LandedCost_SEK": qty * lc * uc,
        "HistFlow":  hf,
        "IsNewRoute": 1 - hf,
    }


# ─── B1: Nearest-Plant Baseline ───────────────────────────────────────────────

def run_b1(data: SCDData, config: dict) -> pd.DataFrame:
    """
    For each (product i, CG g) with demand > 0:
      Assign all Demand[i,g] to the plant with the LOWEST LCmult[p,g]
      among plants where RouteAllowed[p,g] = 1.
      Tie-break: alphabetical by PlantID.
      Fallback: if no plant is allowed for a CG, raise an alert (flagged in output).

    History does NOT affect B1 assignments — only affects alerts.
    """
    rows, alerts = [], []
    for prod, cg in data.active_ig:
        demand = data.demand_map[(prod, cg)]
        # Collect candidate plants: allowed AND have LC data
        candidates = [
            (data.lc_mult.get((p, cg), 999), p)
            for p in data.plants_list
            if data.route_allowed.get((p, cg), 0) == 1
            and (p, cg) in data.lc_mult
        ]
        if not candidates:
            alerts.append(f"B1: No allowed route for {prod}/{cg} — demand {demand} units UNSERVED")
            continue
        candidates.sort()  # ascending LCmult, then alphabetical
        best_plant = candidates[0][1]
        rows.append(_alloc_row(prod, best_plant, cg, demand, data))

    if alerts:
        print("  [B1 alerts]")
        for a in alerts:
            print(f"    ⚠  {a}")
    return pd.DataFrame(rows)


# ─── B2: Baseline Plant ────────────────────────────────────────────────────────

def run_b2(data: SCDData, config: dict) -> pd.DataFrame:
    """
    User selects one baseline plant per product (config['baseline_plant']).
    All demand for all CGs of that product is assigned to the baseline plant.
    If RouteAllowed[baseline, CG] = 0: flagged as 'no route' in alerts;
    volume redistributed to cheapest globally allowed plant.
    """
    baseline = config.get("baseline_plant", {})
    rows, alerts = [], []

    for prod, cg in data.active_ig:
        demand  = data.demand_map[(prod, cg)]
        plant   = baseline.get(prod, data.plants_list[0])

        if data.route_allowed.get((plant, cg), 0) == 1:
            rows.append(_alloc_row(prod, plant, cg, demand, data))
        else:
            alerts.append(
                f"B2: {plant} has no allowed route to {cg} for {prod} "
                f"→ redirecting {demand} units to cheapest allowed plant"
            )
            # Fallback: cheapest allowed plant globally
            candidates = [
                (data.lc_mult.get((p, cg), 999), p)
                for p in data.plants_list
                if data.route_allowed.get((p, cg), 0) == 1
            ]
            if not candidates:
                print(f"  ⚠ B2: No allowed route at all for {prod}/{cg}. Skipping.")
                continue
            candidates.sort()
            fallback = candidates[0][1]
            rows.append(_alloc_row(prod, fallback, cg, demand, data))

    if alerts:
        print("  [B2 alerts]")
        for a in alerts:
            print(f"    ⚠  {a}")
    return pd.DataFrame(rows)


# ─── OM: MILP Optimisation ────────────────────────────────────────────────────

def run_om(data: SCDData, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Full MILP.  Objective (§3 of requirements):
      min  Σ_{i,p,g}  Alloc[i,p,g] × LCmult[p,g] × UnitCost[i]        [C1]
         + Σ_p         FixedCost[p] × LIFECYCLE_YEARS × OpenGlobal[p]  [FC]
         + α × mean(UnitCost) × Σ_{HistFlow=0}  Alloc[i,p,g]          [HP]

    Constraints (§5):
      C1  Demand satisfaction
      C2  Route feasibility (RouteAllowed)
      C3  Plant feasibility (plant must be open)
      C4  Allocation gating (big-M)
      C5  Minimum production if open (MinProd_ifOpen)
      C6  CG coverage (at least 1 route per active CG-product)
      C7  Geo/Category coverage (optional, configurable)
      C8  Capacity (placeholder, non-binding)

    Returns (alloc_df, solve_metadata_dict).
    """
    alpha    = config.get("alpha", ALPHA_DEFAULT)
    geo_par  = config.get("geo_params",  data.geo_params)
    cat_par  = config.get("cat_params",  data.cat_params)
    use_c7   = config.get("use_coverage", True)
    use_c8   = config.get("use_capacity", False)   # non-binding in POC v2

    P = data.plants_list
    I = data.products_list
    G = data.cgs_list
    M_big = float(sum(data.demand_map.values())) * 2

    # ── Model ────────────────────────────────────────────────────────────────
    mdl = pulp.LpProblem("SCD_OM", pulp.LpMinimize)

    # Decision variables
    Open       = pulp.LpVariable.dicts("Open",  [(i,p) for i in I for p in P], cat="Binary")
    RouteOpen  = pulp.LpVariable.dicts("RO",    [(i,p,g) for i in I for p in P for g in G], cat="Binary")
    Alloc      = pulp.LpVariable.dicts("Alloc", [(i,p,g) for i in I for p in P for g in G], lowBound=0)
    OpenGlobal = pulp.LpVariable.dicts("OG",    P, cat="Binary")

    # ── Objective ─────────────────────────────────────────────────────────────
    # For each Alloc variable, precompute its total cost coefficient:
    #   = LCmult × UnitCost  (landed cost per unit)
    #   + α × mean_UnitCost  (penalty if route is non-historical)
    obj_terms = []
    for i in I:
        uc = data.unit_cost[i]
        for p in P:
            for g in G:
                lc  = data.lc_mult.get((p, g), 0.0)
                hf  = data.hist_flow.get((i, p, g), 0)
                pen = 0.0 if hf else alpha * data.mean_uc
                coeff = lc * uc + pen
                if coeff > 0:
                    obj_terms.append(coeff * Alloc[(i,p,g)])

    # Fixed plant cost: FixedCost_SEK × lifecycle years per open plant
    for p in P:
        fc_sek = data.fixed_cost.get(p, 0.0) * 1_000_000 * LIFECYCLE_YEARS
        if fc_sek > 0:
            obj_terms.append(fc_sek * OpenGlobal[p])

    mdl += pulp.lpSum(obj_terms), "TotalObjective"

    # ── Constraints ──────────────────────────────────────────────────────────

    # C1 — Demand satisfaction
    for i, g in data.active_ig:
        dem = data.demand_map[(i, g)]
        mdl += pulp.lpSum(Alloc[(i,p,g)] for p in P) == dem, f"C1_{i}_{g}"

    # C2 — Route feasibility (RouteAllowed)
    for i in I:
        for p in P:
            for g in G:
                allowed = data.route_allowed.get((p, g), 0)
                mdl += RouteOpen[(i,p,g)] <= allowed, f"C2_{i}_{p}_{g}"

    # C3 — Plant feasibility (plant must be open to use route)
    for i in I:
        for p in P:
            for g in G:
                mdl += RouteOpen[(i,p,g)] <= Open[(i,p)], f"C3_{i}_{p}_{g}"

    # C4 — Allocation gating (big-M: volume only on open routes)
    for i in I:
        for p in P:
            for g in G:
                mdl += Alloc[(i,p,g)] <= M_big * RouteOpen[(i,p,g)], f"C4_{i}_{p}_{g}"

    # C5 — Minimum production if plant is open
    for i in I:
        for p in P:
            mp = data.min_prod.get(p, 0)
            mdl += (pulp.lpSum(Alloc[(i,p,g)] for g in G)
                    >= mp * Open[(i,p)]), f"C5_{i}_{p}"

    # C6 — CG coverage: at least 1 route per active (product, CG)
    for i, g in data.active_ig:
        mdl += pulp.lpSum(RouteOpen[(i,p,g)] for p in P) >= 1, f"C6_{i}_{g}"

    # OpenGlobal links: OG[p] = 1 iff Open[i,p]=1 for any i
    for p in P:
        for i in I:
            mdl += OpenGlobal[p] >= Open[(i,p)], f"OG_lb_{p}_{i}"
        mdl += OpenGlobal[p] <= pulp.lpSum(Open[(i,p)] for i in I), f"OG_ub_{p}"

    # C7 — Geo / Category coverage (optional)
    if use_c7:
        for geo, req in geo_par.items():
            if req > 0:
                plants_geo = [p for p in P if data.plant_geo.get(p) == geo]
                mdl += pulp.lpSum(OpenGlobal[p] for p in plants_geo) >= req, f"C7_geo_{geo}"
        for cat, req in cat_par.items():
            if req > 0:
                plants_cat = [p for p in P if data.plant_cat.get(p) == cat]
                mdl += pulp.lpSum(OpenGlobal[p] for p in plants_cat) >= req, f"C7_cat_{cat}"

    # C8 — Capacity (placeholder — non-binding in POC v2)
    if use_c8:
        for i in I:
            for p in P:
                cap = data.capacity.get((i, p), 999_999)
                mdl += pulp.lpSum(Alloc[(i,p,g)] for g in G) <= cap, f"C8_{i}_{p}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=MILP_TIME_LIMIT)
    mdl.solve(solver)
    status = pulp.LpStatus[mdl.status]
    obj    = pulp.value(mdl.objective)

    if mdl.status not in (1, -2):  # 1=Optimal, -2=Not solved (timeout with feasible)
        raise RuntimeError(
            f"MILP solve failed with status '{status}'. "
            "Check feasibility: demand may exceed allowed route capacity, "
            "or coverage constraints may be too tight."
        )

    print(f"  Status  : {status}")
    print(f"  Objective: {obj:,.0f} SEK  (incl. fixed costs + penalty)")

    # Extract allocation
    rows = []
    for i in I:
        for p in P:
            for g in G:
                qty = pulp.value(Alloc[(i,p,g)]) or 0.0
                if qty > 0.01:
                    rows.append(_alloc_row(i, p, g, qty, data))

    meta = {
        "status":      status,
        "objective":   obj,
        "open_global": {p: round(pulp.value(OpenGlobal[p]) or 0) for p in P},
        "open_by_prod":{
            i: [p for p in P if round(pulp.value(Open[(i,p)]) or 0) == 1]
            for i in I
        },
        "alpha":      alpha,
    }
    return pd.DataFrame(rows), meta


# ─── GM: Geo-Mix ──────────────────────────────────────────────────────────────

def run_gm(data: SCDData, config: dict) -> pd.DataFrame:
    """
    User specifies geo volume shares: CN% / EU% / LAT% / US% (must sum to 100).
    For each (product i, CG g, geo):
      target_vol = Demand[i,g] × GeoMix[geo] / 100
      Find plant p* = argmin LCmult[p,g]  s.t.  Geo[p]=geo  AND  RouteAllowed[p,g]=1
      Alloc[i, p*, g] += target_vol
    If no allowed plant in a geo for a given CG:
      Volume redistributed to cheapest globally allowed plant.  Alert generated.

    FixedCost and penalty are NOT applied (deterministic, no optimisation).
    """
    geo_mix = config.get("geo_mix", {"CN": 30, "EU": 70, "LAT": 0, "US": 0})
    total_pct = sum(geo_mix.values())
    if abs(total_pct - 100) > 0.01:
        raise ValueError(f"Geo-mix percentages must sum to 100. Got {total_pct}.")

    rows, alerts = [], []

    for prod, cg in data.active_ig:
        demand = data.demand_map[(prod, cg)]
        allocated = 0.0

        for geo in GEOS:
            pct = geo_mix.get(geo, 0)
            if pct <= 0:
                continue
            target_vol = demand * pct / 100.0

            # Cheapest allowed plant in this geo for this CG
            candidates = [
                (data.lc_mult.get((p, cg), 999), p)
                for p in data.plants_list
                if data.plant_geo.get(p) == geo
                and data.route_allowed.get((p, cg), 0) == 1
            ]

            if candidates:
                candidates.sort()
                best_plant = candidates[0][1]
                rows.append(_alloc_row(prod, best_plant, cg, target_vol, data))
                allocated += target_vol
            else:
                alerts.append(
                    f"GM: No allowed {geo} plant for {prod}/{cg} "
                    f"({pct}% = {target_vol:.0f} units) → fallback to cheapest global"
                )
                # Global fallback
                global_cands = [
                    (data.lc_mult.get((p, cg), 999), p)
                    for p in data.plants_list
                    if data.route_allowed.get((p, cg), 0) == 1
                ]
                if global_cands:
                    global_cands.sort()
                    fbp = global_cands[0][1]
                    rows.append(_alloc_row(prod, fbp, cg, target_vol, data))
                    allocated += target_vol
                else:
                    print(f"  ⚠ GM: No route at all for {prod}/{cg}. Demand unserved.")

    if alerts:
        print("  [GM alerts]")
        for a in alerts:
            print(f"    ⚠  {a}")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KPI ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(alloc_df: pd.DataFrame, data: SCDData,
                 config: dict, label: str, meta: dict = None) -> dict:
    """
    Compute all KPIs from §8 of requirements.
    Returns a dict with scalar KPIs, per-plant summary, and alert lists.
    """
    if alloc_df.empty:
        return {"label": label, "error": "No allocation data"}

    df = alloc_df.copy()
    df["Geo"] = df.PlantID.map(data.plant_geo)
    df["Cat"] = df.PlantID.map(data.plant_cat)

    total_vol    = df.Qty.sum()
    total_lc     = df.LandedCost_SEK.sum()
    weighted_lc  = (df.Qty * df.LCmult).sum() / total_vol if total_vol else 0
    cost_per_unit = total_lc / total_vol if total_vol else 0

    # Open plants (from allocation, not MILP meta — consistent across all scenarios)
    open_plants = df[df.Qty > 0].PlantID.unique().tolist()
    n_open      = len(open_plants)

    # Fixed cost (lifecycle, for open plants)
    fixed_cost_total = sum(
        data.fixed_cost.get(p, 0.0) * 1_000_000 * LIFECYCLE_YEARS
        for p in open_plants
    )
    total_cost_incl_fixed = total_lc + fixed_cost_total

    # Geo & category distribution
    geo_vols = df.groupby("Geo")["Qty"].sum()
    cat_vols = df.groupby("Cat")["Qty"].sum()
    geo_pct  = {g: geo_vols.get(g, 0) / total_vol * 100 for g in GEOS}
    cat_pct  = {c: cat_vols.get(c, 0) / total_vol * 100 for c in CATS}

    # Per-plant summary
    plant_rows = []
    for p in data.plants_list:
        sub = df[df.PlantID == p]
        is_open = p in open_plants
        pr = {
            "PlantID":       p,
            "PlantName":     data.plant_name.get(p, p),
            "Geo":           data.plant_geo.get(p),
            "Category":      data.plant_cat.get(p),
            "IsOpen":        int(is_open),
            "TotalQty":      sub.Qty.sum(),
            "LandedCost_SEK": sub.LandedCost_SEK.sum(),
            "FixedCost_SEK": data.fixed_cost.get(p, 0) * 1_000_000 * LIFECYCLE_YEARS if is_open else 0,
            "UtilPct":       0.0,  # non-binding; placeholder
        }
        for prod in data.products_list:
            pr[f"Qty_{prod.replace('/','_')}"] = sub[sub.ProductID==prod].Qty.sum()
        plant_rows.append(pr)
    plant_df = pd.DataFrame(plant_rows)

    # Dual/multi-plant flag — per (product, CG)
    dual_flags = []
    for prod in data.products_list:
        for cg in data.cgs_list:
            sub = df[(df.ProductID==prod) & (df.CG==cg) & (df.Qty>0)]
            n_plants = sub.PlantID.nunique()
            if n_plants > 1:
                dual_flags.append({
                    "ProductID": prod, "CG": cg,
                    "N_Plants": n_plants,
                    "Plants": ", ".join(sub.PlantID.unique()),
                })
    dual_df = pd.DataFrame(dual_flags)

    # New-route alerts (HistFlow=0 AND Alloc>0)
    new_routes = df[(df.HistFlow==0) & (df.Qty>0)][
        ["ProductID","PlantID","CG","Qty","LCmult"]
    ].copy()
    new_routes["AlertType"] = "new_scd_flow"

    # Consolidation alerts (total demand < MinProd for a product)
    consol_alerts = []
    for prod in data.products_list:
        total_demand = sum(
            v for (i, g), v in data.demand_map.items() if i == prod
        )
        min_prods = list(data.min_prod.values())
        min_overall = min(min_prods) if min_prods else 0
        if total_demand > 0 and total_demand < min_overall:
            consol_alerts.append({
                "ProductID": prod,
                "TotalDemand": total_demand,
                "MinProd_threshold": min_overall,
                "AlertType": "consolidation_candidate",
            })
    consol_df = pd.DataFrame(consol_alerts)

    # Cost by product
    cost_by_prod = df.groupby("ProductID")["LandedCost_SEK"].sum().to_dict()
    vol_by_prod  = df.groupby("ProductID")["Qty"].sum().to_dict()

    return {
        "label":                label,
        "alloc_df":             df,
        "plant_df":             plant_df,
        "dual_df":              dual_df,
        "new_routes_df":        new_routes,
        "consol_df":            consol_df,
        # Scalar KPIs
        "total_vol":            total_vol,
        "total_lc_sek":         total_lc,
        "fixed_cost_sek":       fixed_cost_total,
        "total_cost_sek":       total_cost_incl_fixed,
        "cost_per_unit_sek":    cost_per_unit,
        "weighted_lc_pct":      weighted_lc * 100,
        "n_open_plants":        n_open,
        "open_plants":          open_plants,
        "geo_pct":              geo_pct,
        "cat_pct":              cat_pct,
        "cost_by_prod":         cost_by_prod,
        "vol_by_prod":          vol_by_prod,
        "milp_meta":            meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

def export_scenario(kpis: dict):
    lbl = kpis["label"].replace(" ","_")

    # alloc_{scenario}.csv
    kpis["alloc_df"].to_csv(RESULTS / f"alloc_{lbl}.csv", index=False)

    # plant_summary_{scenario}.csv
    kpis["plant_df"].to_csv(RESULTS / f"plant_summary_{lbl}.csv", index=False)

    # alerts_{scenario}.csv — consolidation + new-route combined
    alert_frames = []
    if not kpis["new_routes_df"].empty:
        nr = kpis["new_routes_df"].copy()
        nr["Detail"] = nr.apply(
            lambda r: f"{r.Qty:.0f} units on new route (LCmult={r.LCmult})", axis=1)
        alert_frames.append(nr[["AlertType","ProductID","PlantID","CG","Detail"]])
    if not kpis["consol_df"].empty:
        cs = kpis["consol_df"].copy()
        cs["PlantID"] = "ALL"
        cs["CG"]      = "ALL"
        cs["Detail"]  = cs.apply(
            lambda r: f"Total demand {r.TotalDemand} < MinProd {r.MinProd_threshold}", axis=1)
        alert_frames.append(cs[["AlertType","ProductID","PlantID","CG","Detail"]])
    alerts_df = pd.concat(alert_frames) if alert_frames else pd.DataFrame(
        columns=["AlertType","ProductID","PlantID","CG","Detail"])
    alerts_df.to_csv(RESULTS / f"alerts_{lbl}.csv", index=False)

    print(f"  → alloc_{lbl}.csv | plant_summary | alerts  ({len(kpis['alloc_df'])} alloc rows)")


def export_kpi_comparison(all_kpis: list):
    rows = []
    for k in all_kpis:
        r = {
            "Scenario":           k["label"],
            "TotalLandedCost_MSEK": k["total_lc_sek"] / 1e6,
            "FixedCost_MSEK":     k["fixed_cost_sek"] / 1e6,
            "TotalCost_MSEK":     k["total_cost_sek"] / 1e6,
            "CostPerUnit_SEK":    k["cost_per_unit_sek"],
            "WeightedLC_Pct":     k["weighted_lc_pct"],
            "N_OpenPlants":       k["n_open_plants"],
            "NewRouteAlerts":     len(k["new_routes_df"]),
        }
        for g in GEOS:
            r[f"Pct_{g}"] = k["geo_pct"].get(g, 0)
        for c in CATS:
            r[f"Pct_{c}"] = k["cat_pct"].get(c, 0)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "kpi_comparison.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(all_kpis: list, data: SCDData):
    W = 84
    print(f"\n{'═'*W}")
    print("  SCD POC v2 — SCENARIO COMPARISON")
    print(f"{'═'*W}")

    labels = [k["label"] for k in all_kpis]
    cw = max(18, 84 // (len(labels) + 1))

    def row(lbl, vals, fmt=None):
        vstr = "".join(f"{v:>{cw}}" for v in vals)
        print(f"  {lbl:<34}{vstr}")

    # Header
    print(f"  {'KPI':<34}" + "".join(f"{l:>{cw}}" for l in labels))
    print(f"  {'─'*34}" + "─"*(cw*len(labels)))

    b2_cost = next((k["total_cost_sek"] for k in all_kpis if "B2" in k["label"]), None)
    b1_cost = next((k["total_cost_sek"] for k in all_kpis if "B1" in k["label"]), None)

    row("Landed Cost (MSEK)",
        [f"{k['total_lc_sek']/1e6:.1f}" for k in all_kpis])
    row("Fixed Plant Cost (MSEK)",
        [f"{k['fixed_cost_sek']/1e6:.1f}" for k in all_kpis])
    row("TOTAL Cost incl. Fixed (MSEK)",
        [f"{k['total_cost_sek']/1e6:.1f}" for k in all_kpis])
    row("Cost per Unit (SEK)",
        [f"{k['cost_per_unit_sek']:,.0f}" for k in all_kpis])
    row("Weighted LC %",
        [f"{k['weighted_lc_pct']:.2f}%" for k in all_kpis])
    row("TPI Avoidance vs B2 (MSEK)",
        ["—" if b2_cost is None or "B2" in k["label"]
         else f"{(b2_cost-k['total_cost_sek'])/1e6:+.1f}"
         for k in all_kpis])
    row("# Open Plants",
        [str(k["n_open_plants"]) for k in all_kpis])
    row("New-Route Alerts",
        [str(len(k["new_routes_df"])) for k in all_kpis])

    print(f"\n  {'── Cost by Product (MSEK) ──':<34}" + "─"*(cw*len(all_kpis)))
    for prod in data.products_list:
        row(f"  {prod}",
            [f"{k['cost_by_prod'].get(prod,0)/1e6:.1f}" for k in all_kpis])

    print(f"\n  {'── Volume by Product (k units) ──':<34}" + "─"*(cw*len(all_kpis)))
    for prod in data.products_list:
        row(f"  {prod}",
            [f"{k['vol_by_prod'].get(prod,0)/1e3:.1f}k" for k in all_kpis])

    print(f"\n  {'── Open Plants by Geo ──':<34}" + "─"*(cw*len(all_kpis)))
    for geo in GEOS:
        vals = []
        for k in all_kpis:
            plants_geo = [p for p in k["open_plants"] if data.plant_geo.get(p)==geo]
            vals.append(f"{len(plants_geo)}  ({k['geo_pct'].get(geo,0):.0f}% vol)")
        row(f"  {geo}", vals)

    print(f"\n  {'── Open Plants by Category ──':<34}" + "─"*(cw*len(all_kpis)))
    for cat in CATS:
        vals = []
        for k in all_kpis:
            plants_cat = [p for p in k["open_plants"] if data.plant_cat.get(p)==cat]
            vals.append(f"{len(plants_cat)}  ({k['cat_pct'].get(cat,0):.0f}% vol)")
        row(f"  {cat}", vals)

    # MILP details
    om_kpis = next((k for k in all_kpis if "OM" in k["label"] and k.get("milp_meta")), None)
    if om_kpis:
        meta = om_kpis["milp_meta"]
        print(f"\n  {'── OM MILP Details ──'}")
        print(f"  Status       : {meta['status']}")
        print(f"  Objective    : {meta['objective']:,.0f} SEK")
        print(f"  Alpha (α)    : {meta['alpha']} (≈{meta['alpha']*100:.0f}% LC penalty per non-hist. unit)")
        print(f"  Open globally: {[p for p in data.plants_list if meta['open_global'].get(p,0)>0.5]}")
        if not om_kpis["new_routes_df"].empty:
            print(f"\n  New-route alerts (OM): {len(om_kpis['new_routes_df'])} flows with no historical precedent")
            for _, r in om_kpis["new_routes_df"].iterrows():
                print(f"    ⚠ {r.ProductID} | {r.PlantID} → {r.CG}  ({r.Qty:.0f} units)")

    print(f"\n{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1_comparison(all_kpis: list, data: SCDData):
    """3-panel: total cost breakdown | cost by product | geo distribution."""
    labels = [k["label"] for k in all_kpis]
    x = np.arange(len(labels))
    colors = [SCENARIO_COLORS.get(l.split("-")[0].strip(), "#607D8B") for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "SCD POC v2 — Scenario Comparison\n"
        "KRD901252/11 (AIR 6419 B77D)  &  KRC 4464 B1B3B7 | Lifecycle Horizon",
        fontsize=12, fontweight="bold")

    # Panel A: Stacked cost (landed + fixed)
    ax = axes[0]
    lc_vals  = [k["total_lc_sek"]/1e6 for k in all_kpis]
    fc_vals  = [k["fixed_cost_sek"]/1e6 for k in all_kpis]
    b1 = ax.bar(x, lc_vals, color=colors, width=0.5, edgecolor="white", label="Landed cost")
    b2_bar = ax.bar(x, fc_vals, bottom=lc_vals, color="#CFD8DC", width=0.5,
                    edgecolor="white", hatch="//", label="Fixed plant cost")
    for i, (lc, fc) in enumerate(zip(lc_vals, fc_vals)):
        ax.text(i, lc+fc+2, f"{lc+fc:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Total Cost (MSEK)", fontweight="bold")
    ax.set_ylabel("MSEK")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.legend(fontsize=8)

    # Panel B: Cost by product (stacked)
    ax = axes[1]
    bottom = np.zeros(len(all_kpis))
    for prod in data.products_list:
        vals = np.array([k["cost_by_prod"].get(prod,0)/1e6 for k in all_kpis])
        short = prod.replace("/","_")[:12]
        ax.bar(x, vals, bottom=bottom, label=short,
               color=PRODUCT_COLORS.get(prod,"#888"), width=0.5, edgecolor="white")
        bottom += vals
    ax.set_title("Landed Cost by Product (MSEK)", fontweight="bold")
    ax.set_ylabel("MSEK")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.legend(fontsize=8)

    # Panel C: Geo volume distribution (stacked %)
    ax = axes[2]
    bottom = np.zeros(len(all_kpis))
    for geo in GEOS:
        vals = np.array([k["geo_pct"].get(geo,0) for k in all_kpis])
        ax.bar(x, vals, bottom=bottom, label=geo,
               color=GEO_COLORS.get(geo,"#888"), width=0.5, edgecolor="white")
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(i, bottom[i]+v/2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_title("Volume by Geography (%)", fontweight="bold")
    ax.set_ylabel("% of Lifecycle Volume")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 115)

    plt.tight_layout()
    path = RESULTS / "fig1_scenario_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path.name}")


def plot_fig2_network(om_kpis: dict, data: SCDData):
    """OM allocation heatmap + plant summary + constraint check."""
    alloc_df = om_kpis["alloc_df"]
    meta     = om_kpis["milp_meta"] or {}
    open_g   = meta.get("open_global", {})

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    fig.suptitle("OM — Optimised Network Design  |  Plant × Hub Allocation",
                 fontsize=13, fontweight="bold")

    active_cgs = sorted(alloc_df.CG.unique())

    # A: Allocation heatmap (spans left 2 columns, both rows)
    ax1 = fig.add_subplot(gs[:, :2])
    pivot = (alloc_df.groupby(["PlantID","CG"])["Qty"].sum()
             .unstack(fill_value=0)
             .reindex(data.plants_list, fill_value=0)
             .reindex(columns=active_cgs, fill_value=0))

    # Annotation: product breakdown inside each cell
    annot = np.empty(pivot.shape, dtype=object)
    for ri, p in enumerate(data.plants_list):
        for ci, cg in enumerate(active_cgs):
            sub = alloc_df[(alloc_df.PlantID==p)&(alloc_df.CG==cg)]
            total = sub.Qty.sum()
            if total < 1:
                annot[ri,ci] = ""
            else:
                parts = [f"{r.ProductID.split('/')[0]}:{r.Qty/1000:.1f}k"
                         for _,r in sub.iterrows()]
                annot[ri,ci] = "\n".join(parts)

    cmap = LinearSegmentedColormap.from_list("scd", ["#FFFFFF","#BBDEFB","#0D47A1"])
    sns.heatmap(pivot/1000, ax=ax1, cmap=cmap, linewidths=0.6, linecolor="#E0E0E0",
                cbar_kws={"label":"Volume (k units)","shrink":0.7},
                annot=annot, fmt="", annot_kws={"size":8,"va":"center"})

    ylabels = []
    for p in data.plants_list:
        is_open = open_g.get(p, 0) > 0.5
        marker  = " ●" if is_open else "  ○"
        ylabels.append(f"{p}  [{data.plant_geo.get(p)}|{data.plant_cat.get(p)}]{marker}")
    ax1.set_yticklabels(ylabels, rotation=0, fontsize=8.5)
    ax1.set_xlabel("Hub (Customer Group)", fontsize=10)
    ax1.set_ylabel("")
    ax1.set_title("Allocation by Plant × Hub  (k units, per product)\n● open plant  ○ closed",
                  fontsize=10, fontweight="bold")

    # B: Plant volume bar (right top)
    ax2 = fig.add_subplot(gs[0,2])
    ax2.set_facecolor("#FFFFFF")
    ax2.spines[["top","right"]].set_visible(False)
    vol_pp = alloc_df.groupby(["PlantID","ProductID"])["Qty"].sum().unstack(fill_value=0)
    for prod in data.products_list:
        if prod not in vol_pp.columns:
            vol_pp[prod] = 0
    yp = np.arange(len(data.plants_list))
    bottom = np.zeros(len(data.plants_list))
    for prod in data.products_list:
        vals = np.array([vol_pp.get(prod, pd.Series()).get(p, 0)/1000
                         for p in data.plants_list])
        ax2.barh(yp, vals, left=bottom, height=0.6,
                 color=PRODUCT_COLORS.get(prod,"#888"),
                 label=prod.split("/")[0], edgecolor="white")
        bottom += vals
    ax2.set_yticks(yp)
    ylabs2 = [f"{p.split('/')[-1]} ({data.plant_geo.get(p)})" for p in data.plants_list]
    ax2.set_yticklabels(ylabs2, fontsize=8)
    for i, p in enumerate(data.plants_list):
        if open_g.get(p, 0) > 0.5:
            ax2.get_yticklabels()[i].set_color("#1565C0")
            ax2.get_yticklabels()[i].set_fontweight("bold")
    ax2.set_xlabel("Volume (k units)")
    ax2.set_title("Volume by Plant & Product\n(blue labels = open)", fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right")

    # C: Coverage constraint check (right bottom)
    ax3 = fig.add_subplot(gs[1,2])
    ax3.set_facecolor("#FFFFFF")
    ax3.spines[["top","right"]].set_visible(False)
    checks = [
        ("CN ≥1","Geo","CN",1), ("EU ≥1","Geo","EU",1),
        ("LAT ≥0","Geo","LAT",0), ("US ≥0","Geo","US",0),
        ("ESS ≥1","Cat","ESS",1), ("EMS ≥1","Cat","EMS",1),
    ]
    labs, actuals, oks = [], [], []
    open_pl = [p for p in data.plants_list if open_g.get(p,0) > 0.5]
    for lbl, typ, name, req in checks:
        if typ == "Geo":
            actual = sum(1 for p in open_pl if data.plant_geo.get(p)==name)
        else:
            actual = sum(1 for p in open_pl if data.plant_cat.get(p)==name)
        labs.append(lbl)
        actuals.append(actual)
        oks.append(actual >= req)
    xc = np.arange(len(labs))
    bcolors = ["#4CAF50" if o else "#F44336" for o in oks]
    ax3.bar(xc, actuals, color=bcolors, width=0.55, edgecolor="white")
    ax3.axhline(1, color="#333", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("# Open Plants")
    ax3.set_title("Coverage Constraint Check\n(green=satisfied)", fontweight="bold")
    for i, (a, o) in enumerate(zip(actuals, oks)):
        ax3.text(i, a + 0.05, f"{'✓' if o else '✗'} {a}",
                 ha="center", va="bottom", fontsize=11,
                 color="#1B5E20" if o else "#B71C1C", fontweight="bold")

    fig.savefig(RESULTS / "fig2_om_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → fig2_om_network.png")


def plot_fig3_demand_satisfaction(all_kpis: list, data: SCDData):
    """Per-scenario demand vs served volume check for each product."""
    n_scen = len(all_kpis)
    n_prod = len(data.products_list)
    fig, axes = plt.subplots(n_prod, n_scen, figsize=(5*n_scen, 4*n_prod),
                             squeeze=False)
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("Demand Satisfaction Check — All Scenarios & Products",
                 fontsize=12, fontweight="bold")

    for pi, prod in enumerate(data.products_list):
        demand_pos = data.demand[data.demand.ProductID == prod]
        active_cgs = demand_pos[demand_pos.Demand > 0].CG.tolist()
        xp = np.arange(len(active_cgs))
        w  = 0.36

        for si, kpis in enumerate(all_kpis):
            ax = axes[pi][si]
            ax.set_facecolor("#FFFFFF")
            ax.spines[["top","right"]].set_visible(False)

            d_vals = [data.demand_map.get((prod,cg),0)/1000 for cg in active_cgs]
            a_vals = []
            for cg in active_cgs:
                sub = kpis["alloc_df"]
                qty = sub[(sub.ProductID==prod)&(sub.CG==cg)].Qty.sum()
                a_vals.append(qty/1000)

            ax.bar(xp-w/2, d_vals, w, label="Demand",
                   color="#90CAF9", edgecolor="white")
            ax.bar(xp+w/2, a_vals, w, label="Allocated",
                   color=PRODUCT_COLORS.get(prod,"#888"),
                   edgecolor="white", alpha=0.85)

            # Check: are all demands met?
            max_err = max(abs(d-a) for d,a in zip(d_vals,a_vals)) if d_vals else 0
            status  = "✓ 100% served" if max_err < 0.01 else f"⚠ max gap {max_err:.1f}k"
            ax.set_title(f"{kpis['label']}\n{prod[:14]}", fontsize=9, fontweight="bold",
                         color=PRODUCT_COLORS.get(prod,"#555"))
            ax.set_xlabel(status, fontsize=8,
                          color="#2E7D32" if "✓" in status else "#C62828")
            ax.set_xticks(xp)
            ax.set_xticklabels(active_cgs, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Volume (k units)", fontsize=8)
            if si == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS / "fig3_demand_check.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → fig3_demand_check.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TERMINAL UI
# ══════════════════════════════════════════════════════════════════════════════

def _prompt(msg: str, default=None, cast=str, choices=None):
    """Prompt user for input with a default and optional validation."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {msg}{default_str}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗ Expected {cast.__name__}, got '{raw}'. Try again.")
            continue
        if choices and val not in choices:
            print(f"  ✗ Must be one of {choices}. Try again.")
            continue
        return val


def _prompt_baseline_plant(data: SCDData) -> dict:
    print("\n  Available plants:")
    for i, p in enumerate(data.plants_list):
        print(f"    {i+1}. {p}  ({data.plant_name[p]}, {data.plant_geo[p]}, {data.plant_cat[p]})")
    baseline = {}
    for prod in data.products_list:
        idx = _prompt(f"Baseline plant index for {prod}",
                      default=3, cast=int)  # default = ERI/PL-TC (index 3)
        idx = max(1, min(idx, len(data.plants_list)))
        baseline[prod] = data.plants_list[idx-1]
        print(f"    → {baseline[prod]}")
    return baseline


def _prompt_geo_mix() -> dict:
    print("\n  Enter geo-mix percentages (must sum to 100).")
    while True:
        cn  = _prompt("CN %",  default=30, cast=float)
        eu  = _prompt("EU %",  default=70, cast=float)
        lat = _prompt("LAT %", default=0,  cast=float)
        us  = _prompt("US %",  default=0,  cast=float)
        total = cn + eu + lat + us
        if abs(total - 100) < 0.01:
            return {"CN": cn, "EU": eu, "LAT": lat, "US": us}
        print(f"  ✗ Sum = {total:.1f}. Must equal 100. Try again.")


def _prompt_om_params(data: SCDData) -> dict:
    print("\n  Configure OM parameters (press Enter for defaults).")
    alpha    = _prompt("α (penalty weight, 0=no penalty)", default=ALPHA_DEFAULT, cast=float)
    use_cov  = _prompt("Enforce geo/category coverage? (y/n)", default="y").lower() == "y"
    geo_par  = deepcopy(data.geo_params)
    cat_par  = deepcopy(data.cat_params)
    if use_cov:
        print("  Coverage minimums (press Enter to keep defaults):")
        for geo in GEOS:
            geo_par[geo] = _prompt(f"  Min {geo} plants", default=geo_par.get(geo,0), cast=int)
        for cat in CATS:
            cat_par[cat] = _prompt(f"  Min {cat} plants", default=cat_par.get(cat,0), cast=int)
    return {
        "alpha":        alpha,
        "use_coverage": use_cov,
        "geo_params":   geo_par,
        "cat_params":   cat_par,
    }


SCENARIO_MENU = {
    "1": "B1",
    "2": "B2",
    "3": "OM",
    "4": "GM",
    "5": "ALL",
}

def terminal_menu(data: SCDData) -> list[dict]:
    """
    Interactive terminal menu.
    Returns a list of scenario config dicts to run.
    """
    W = 60
    print(f"\n{'╔'+'═'*(W-2)+'╗'}")
    print(f"{'║':1}{'SCD POC v2 — Supply Chain Design Optimisation':^{W-2}}{'║':1}")
    print(f"{'╚'+'═'*(W-2)+'╝'}")
    print()
    print("  Network:  8 plants  |  7 CGs  |  2 pilot products")
    print("  Horizon:  lifecycle (~4 years)")
    print()
    data.summary()
    print()
    print("  ─── Select Scenario ─────────────────────────────────")
    print("    1  B1 — Nearest-Plant Baseline")
    print("    2  B2 — Baseline Plant")
    print("    3  OM — MILP Optimised")
    print("    4  GM — Geo-Mix")
    print("    5  ALL — Run all 4 scenarios and compare")
    print("    0  Exit")
    print()

    choice = _prompt("Choice", default="5", choices=["0","1","2","3","4","5"])
    if choice == "0":
        print("  Exiting.")
        sys.exit(0)

    scenarios_to_run = list(SCENARIO_MENU[choice].split()) if choice != "5" else ["B1","B2","OM","GM"]

    configs = []
    for scen in scenarios_to_run:
        cfg = {"scenario": scen}

        if scen == "B2":
            print(f"\n{'─'*50}")
            print("  B2 — Baseline Plant Configuration")
            cfg["baseline_plant"] = _prompt_baseline_plant(data)

        elif scen == "OM":
            print(f"\n{'─'*50}")
            print("  OM — MILP Configuration")
            cfg.update(_prompt_om_params(data))

        elif scen == "GM":
            print(f"\n{'─'*50}")
            print("  GM — Geo-Mix Configuration")
            cfg["geo_mix"] = _prompt_geo_mix()

        # For B1: defaults only (no user params needed)
        if scen == "B2" and "baseline_plant" not in cfg:
            cfg["baseline_plant"] = {p: "ERI/PL-TC" for p in data.products_list}

        configs.append(cfg)

    return configs


def apply_defaults(configs: list, data: SCDData) -> list:
    """Apply default parameters for non-interactive (--all) mode."""
    for cfg in configs:
        if cfg["scenario"] == "B2" and "baseline_plant" not in cfg:
            cfg["baseline_plant"] = {p: "ERI/PL-TC" for p in data.products_list}
        if cfg["scenario"] == "OM":
            cfg.setdefault("alpha",        ALPHA_DEFAULT)
            cfg.setdefault("use_coverage", True)
            cfg.setdefault("geo_params",   data.geo_params)
            cfg.setdefault("cat_params",   data.cat_params)
        if cfg["scenario"] == "GM":
            cfg.setdefault("geo_mix", {"CN": 30, "EU": 70, "LAT": 0, "US": 0})
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "B1": lambda d, c: (run_b1(d, c), None),
    "B2": lambda d, c: (run_b2(d, c), None),
    "OM": lambda d, c: run_om(d, c),
    "GM": lambda d, c: (run_gm(d, c), None),
}

LABELS = {
    "B1": "B1-Nearest",
    "B2": "B2-Baseline",
    "OM": "OM-Optimised",
    "GM": "GM-GeoMix",
}


def main():
    parser = argparse.ArgumentParser(description="SCD POC v2")
    parser.add_argument("--all",   action="store_true", help="Run all 4 scenarios with defaults")
    parser.add_argument("--regen", action="store_true", help="Regenerate dummy data before running")
    args = parser.parse_args()

    # ── Regenerate data if requested or missing ────────────────────────────
    if args.regen or not (DATA / "plants.csv").exists():
        print("Generating dummy data...")
        import generate_data
        generate_data.main()

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading input data...")
    try:
        data = SCDData()
    except RuntimeError as e:
        print(f"\n✗ {e}")
        sys.exit(1)

    # ── Get scenario configs ───────────────────────────────────────────────
    if args.all:
        print("\n[Non-interactive mode — running all 4 scenarios with defaults]\n")
        configs = [{"scenario": s} for s in ["B1","B2","OM","GM"]]
        configs = apply_defaults(configs, data)
    else:
        configs = terminal_menu(data)
        configs = apply_defaults(configs, data)

    # ── Run scenarios ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    all_kpis = []
    for cfg in configs:
        scen = cfg["scenario"]
        lbl  = LABELS.get(scen, scen)
        print(f"\n  ► Running {lbl}...")
        try:
            alloc_df, meta = RUNNERS[scen](data, cfg)
        except Exception as e:
            print(f"  ✗ {scen} failed: {e}")
            raise

        kpis = compute_kpis(alloc_df, data, cfg, lbl, meta)
        all_kpis.append(kpis)

        print(f"    Total landed cost : {kpis['total_lc_sek']/1e6:.1f} MSEK")
        print(f"    Fixed plant cost  : {kpis['fixed_cost_sek']/1e6:.1f} MSEK")
        print(f"    Total incl. fixed : {kpis['total_cost_sek']/1e6:.1f} MSEK")
        print(f"    # Open plants     : {kpis['n_open_plants']}")
        print(f"    New-route alerts  : {len(kpis['new_routes_df'])}")
        export_scenario(kpis)

    # ── Reports ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print_report(all_kpis, data)

    print("  Exporting KPI comparison table...")
    export_kpi_comparison(all_kpis)
    print(f"  → kpi_comparison.csv")

    print("\n  Generating figures...")
    plot_fig1_comparison(all_kpis, data)
    om_kpis = next((k for k in all_kpis if "OM" in k["label"] and k.get("milp_meta")), None)
    if om_kpis:
        plot_fig2_network(om_kpis, data)
    plot_fig3_demand_satisfaction(all_kpis, data)

    print(f"\n  All outputs in: {RESULTS}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
