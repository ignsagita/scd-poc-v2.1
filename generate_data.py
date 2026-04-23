#!/usr/bin/env python3
"""
SCD POC v2 — Dummy Data Generator
===================================
Generates all input CSVs with realistic dummy data aligned to the
production Snowflake schema (Requirements v1.0, §6).

PURPOSE FOR DATA ENGINEERS:
  Each CSV generated here defines the exact table/column contract
  that must be delivered from Snowflake before real data can be
  connected.  Column names, data types, and allowed values are
  intentional — do not rename without updating scd_poc_v2.py.

PILOT PRODUCTS:
  KRD901252/11  (AIR 6419 B77D)
  KRC4464B1B3B7 (Radio 4464 B1/B3/B7)

PLANT IDs use ERI/{country}-{site} SCN format (same as SAIDA / GFT).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["KRD901252/11", "KRC4464B1B3B7"]
CGS      = ["CG_1","CG_2","CG_3","CG_4","CG_5","CG_6","CG_7"]

LIFECYCLE_YEARS = 4  # horizon used to scale annual FixedCost → lifecycle cost

# ─── PLANT ATTRIBUTES ────────────────────────────────────────────────────────
#   PlantID        : SCN code (primary key in SAIDA / GFT)
#   PlantName      : human-readable factory name
#   Geo            : CN | EU | LAT | US
#   Category       : ESS | EMS
#   IsNPISite      : 1 = approved for NPI/pilot production
#   IsVolumeSite   : 1 = regular high-volume site
#   MinProd_ifOpen : min lifecycle volume if plant is open for a product
#                    (= economic MOQ per plant; SME input needed for real values)
#   FixedCost_MSEK_yr : annual plant overhead NOT already in LC adder
#                       (line setup, qualification, staffing above flow cost)
#                       Source: industrial finance (approx. bands for POC)
#   Cap_KRD, Cap_KRC : eco-fulfilment capacity per product (lifecycle units)
#                      999999 = non-binding placeholder until real data arrives
# ─────────────────────────────────────────────────────────────────────────────

PLANT_ROWS = [
    # PlantID        Name              Geo  Cat  NPI  Vol  MinProd  FC_yr  Cap_KRD  Cap_KRC
    ("ERI/CN-ESS",  "ENC_Nanjing",    "CN","ESS", 1,   1,  2000,   2.0, 999999, 999999),
    ("ERI/CN-EMS",  "WX_Nanjing",     "CN","EMS", 0,   1,  1500,   1.5, 999999, 999999),
    ("ERI/PL-TC",   "TC_Poland",      "EU","ESS", 1,   1,  1000,   4.0, 999999, 999999),
    ("ERI/SE-ESH",  "ESH_Sweden",     "EU","EMS", 1,   0,   800,   3.5, 999999, 999999),
    ("ERI/HU-EMS",  "Flex_Hungary",   "EU","EMS", 0,   1,  1200,   3.8, 999999, 999999),
    ("ERI/US-EFA",  "EFA_USA",        "US","ESS", 0,   1,  1500,   3.5, 999999, 999999),
    ("ERI/US-JABIL","Jabil_USA",      "US","EMS", 0,   1,  1200,   3.0, 999999, 999999),
    ("ERI/MX-GUAD", "GUAD_Mexico",    "LAT","ESS",0,   1,  1500,   2.5, 999999, 999999),
]

PLANTS_COLS = ["PlantID","PlantName","Geo","Category","IsNPISite","IsVolumeSite",
               "MinProd_ifOpen","FixedCost_MSEK_yr","Cap_KRD901252","Cap_KRC4464"]


# ─── LC ADDER MATRIX ─────────────────────────────────────────────────────────
#   Adder[p,g,ProductCategory] = fraction added on top of UnitCost
#   LCmult = 1 + Adder
#   Both pilot products are Baseband → same adder on any given route.
#   None = geopolitically FORBIDDEN route → RouteAllowed=0
#
#   Design rules (approximate; real values come from Henrik's LC tool):
#     Same region    : lowest adder (~0.05-0.10)
#     Cross-region   : moderate (~0.15-0.22)
#     Forbidden corridors (ERIDOC BNEW-24:038086Uen):
#       CN → US (CG_5):  tariff/geopolitical barrier
#       CN → LAT (CG_4): no established logistics corridor
#       LAT → SEAO (CG_3): no direct corridor
#       LAT → MNEA (CG_6, CG_7): no direct corridor
# ─────────────────────────────────────────────────────────────────────────────
#                   CG_1   CG_2   CG_3   CG_4   CG_5   CG_6   CG_7
#                  EMEA/SE EMEA/DB SEAO   LatAm  US     MNEA   MNEA/JP
ADDER = {
    "ERI/CN-ESS":  [0.18,  0.20,  0.08,  None,  None,  0.06,  0.07],
    "ERI/CN-EMS":  [0.19,  0.21,  0.09,  None,  None,  0.07,  0.08],
    "ERI/PL-TC":   [0.05,  0.10,  0.18,  0.22,  0.20,  0.12,  0.15],
    "ERI/SE-ESH":  [0.06,  0.11,  0.19,  0.23,  0.21,  0.13,  0.16],
    "ERI/HU-EMS":  [0.07,  0.10,  0.19,  0.22,  0.21,  0.12,  0.15],
    "ERI/US-EFA":  [0.20,  0.22,  0.22,  0.12,  0.08,  0.20,  0.22],
    "ERI/US-JABIL":[0.21,  0.23,  0.23,  0.13,  0.09,  0.21,  0.23],
    "ERI/MX-GUAD": [0.22,  None,  None,  0.10,  0.12,  None,  None],
}


def gen_plants():
    df = pd.DataFrame(PLANT_ROWS, columns=PLANTS_COLS)
    df.to_csv(DATA / "plants.csv", index=False)
    print(f"  plants.csv          — {len(df)} rows")
    return df


def gen_products():
    """
    products.csv
    ─────────────────────────────────────────────────────
    ProductID       : parent product code (KRCxxx / KRDxxx level)
    ProductArea     : Radio | RAN Compute | Site Material
    ProductCategory : AAS | Baseband | Classical Macro
                      (determines which LC adder row applies in the LC tool)
    UnitCost_SEK    : base TK per unit (manufacturing cost, from CRM cost tree)
    Description     : human-readable label
    """
    rows = [
        ("KRD901252/11",  "Radio", "Baseband", 18000, "AIR 6419 B77D"),
        ("KRC4464B1B3B7", "Radio", "Baseband", 17506, "Radio 4464 B1/B3/B7"),
    ]
    df = pd.DataFrame(rows, columns=["ProductID","ProductArea","ProductCategory",
                                     "UnitCost_SEK","Description"])
    df.to_csv(DATA / "products.csv", index=False)
    print(f"  products.csv        — {len(df)} rows")
    return df


def gen_demand():
    """
    demand.csv
    ─────────────────────────────────────────────────────
    ProductID : links to products.csv
    CG        : customer group / hub (links to cg_map.csv)
    Demand    : lifecycle volume (3–5 year total) per product-hub

    Source: MAF + LRFP (final source TBD with SuPM team).
    Rows with Demand=0 are OMITTED — the engine treats missing rows as zero
    and excludes those CG-product pairs from all constraints (C1, C6).

    CG_2 and CG_7 have no demand for either pilot product in this dummy set —
    this reflects the realistic pattern that not every product covers every MA.
    """
    rows = [
        # KRD901252/11 — focused on EMEA, SEAO, US
        ("KRD901252/11",  "CG_1", 30000),
        ("KRD901252/11",  "CG_3",  5000),
        ("KRD901252/11",  "CG_5",  2000),
        # KRC4464B1B3B7 — broader coverage incl. LatAm, MNEA
        ("KRC4464B1B3B7", "CG_1", 40500),
        ("KRC4464B1B3B7", "CG_3",  3600),
        ("KRC4464B1B3B7", "CG_4",   450),
        ("KRC4464B1B3B7", "CG_6",   450),
    ]
    df = pd.DataFrame(rows, columns=["ProductID","CG","Demand"])
    df.to_csv(DATA / "demand.csv", index=False)
    print(f"  demand.csv          — {len(df)} rows  (zero-demand CGs omitted)")
    return df


def gen_routes():
    """
    routes.csv
    ─────────────────────────────────────────────────────
    PlantID         : SCN plant code
    CG              : customer group / hub
    ProductCategory : AAS | Baseband | Classical Macro
                      The LC adder is per (plant, hub, ProductCategory).
                      Both pilot products are Baseband — they share the same
                      adder on any given route. Include this column so the
                      schema supports future product categories without change.
    AdderPct        : LC adder as decimal (e.g. 0.18 = 18%)
    LCmult          : 1 + AdderPct (convenience column; redundant but useful)

    Source: Henrik's landed cost tool / CRM cost tree.
    Only ALLOWED routes appear here. Forbidden routes → route_allowed.csv.
    """
    rows = []
    plants_list = [r[0] for r in PLANT_ROWS]
    for plant in plants_list:
        for j, cg in enumerate(CGS):
            adder = ADDER[plant][j]
            if adder is not None:
                rows.append({
                    "PlantID": plant,
                    "CG": cg,
                    "ProductCategory": "Baseband",
                    "AdderPct": round(adder, 4),
                    "LCmult": round(1 + adder, 4),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "routes.csv", index=False)
    print(f"  routes.csv          — {len(df)} rows  (allowed routes only)")
    return df


def gen_route_allowed():
    """
    route_allowed.csv
    ─────────────────────────────────────────────────────
    PlantID      : SCN plant code
    CG           : customer group / hub
    RouteAllowed : 1 = geopolitically allowed for all products
                   0 = FORBIDDEN for all products (ERIDOC BNEW-24:038086Uen)

    IMPORTANT: RouteAllowed is product-agnostic. If a plant-hub route is
    forbidden, it is forbidden for every product on that route.

    Data owner: Clovis Hiroshi Kawai.
    Changes infrequently but is a HARD model constraint — stale data causes
    infeasibility or incorrect recommendations.

    Forbidden corridors in this dummy set:
      CN → US (CG_5)   : tariff / geopolitical barrier
      CN → LAT (CG_4)  : no established logistics corridor
      LAT → SEAO (CG_3): no direct corridor
      LAT → MNEA (CG_6/7): no direct corridor
      LAT → EMEA/DB (CG_2): no corridor from Mexico/LatAm to Dubai
    """
    rows = []
    plants_list = [r[0] for r in PLANT_ROWS]
    for plant in plants_list:
        for j, cg in enumerate(CGS):
            adder = ADDER[plant][j]
            rows.append({
                "PlantID": plant,
                "CG": cg,
                "RouteAllowed": 0 if adder is None else 1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "route_allowed.csv", index=False)
    allowed = df.RouteAllowed.sum()
    forbidden = (df.RouteAllowed == 0).sum()
    print(f"  route_allowed.csv   — {len(df)} rows  ({allowed} allowed, {forbidden} forbidden)")
    return df


def gen_cg_map():
    """
    cg_map.csv
    ─────────────────────────────────────────────────────
    CG          : customer group ID
    MarketArea  : Ericsson market area name
    HubName     : supply hub name
    HubGeo      : geo region of the hub (used for reporting only)

    Used for display/reporting only — NOT a constraint parameter.

    NOTE on naming: CG_3 hub is 'Nanjing', CG_6/CG_7 hub is 'China'.
    Both are physically in China but they are DISTINCT hubs in Ericsson's
    hub network. Do not conflate them.
    """
    rows = [
        ("CG_1","EMEA",         "Sweden_SE",    "EU"),
        ("CG_2","EMEA",         "Dubai_DB",     "MNEA"),
        ("CG_3","MOAI_SEAO",    "Nanjing_NJ",   "CN"),
        ("CG_4","MACS_ex_US",   "LatAm_Multi",  "LAT"),
        ("CG_5","MACS_US",      "Dallas_EFA",   "US"),
        ("CG_6","MNEA_ex_JP",   "China_CN",     "CN"),
        ("CG_7","MNEA_JP",      "China_CN",     "CN"),
    ]
    df = pd.DataFrame(rows, columns=["CG","MarketArea","HubName","HubGeo"])
    df.to_csv(DATA / "cg_map.csv", index=False)
    print(f"  cg_map.csv          — {len(df)} rows")
    return df


def gen_hist_flow(demand_df, plants_df):
    """
    hist_flow.csv
    ─────────────────────────────────────────────────────
    ProductID    : links to products.csv
    PlantID      : SCN plant code
    CG           : customer group / hub
    HistFlow     : 1 = this route existed historically for this product
                   0 = new route (no historical precedent)
    HistVolShare : historical volume share across plants for this product-CG
                   (optional; used for reference only, not in the MILP)

    Source: Amrith's SCD flow description in Snowflake (table TBD).
    For POC v2: DUMMY DATA. Historical flows are approximated as the
    routes that would be used in a sensible baseline network (what a
    well-informed user would expect to see as "today's SCD").

    Impact on optimisation:
      - Routes with HistFlow=0 incur a soft penalty (α × mean UnitCost per unit)
        in the OM objective → model prefers historical routes when cost is similar.
      - New-route alerts are generated for any Alloc > 0 where HistFlow = 0.
    """
    # KRD901252/11 historical network:
    #   CG_1 (EMEA): split between Poland (70%) and CN-ESS (30%)
    #   CG_3 (SEAO): entirely from CN-ESS (near hub)
    #   CG_5 (US):   entirely from EFA-USA (local)
    # KRC4464B1B3B7 historical network:
    #   CG_1 (EMEA): split Poland (60%) and CN-ESS (40%)
    #   CG_3 (SEAO): entirely from CN-ESS
    #   CG_4 (LatAm): from Mexico GUAD
    #   CG_6 (MNEA):  from CN-EMS

    hist_routes = {
        ("KRD901252/11",  "ERI/CN-ESS",  "CG_1"): 0.30,
        ("KRD901252/11",  "ERI/PL-TC",   "CG_1"): 0.70,
        ("KRD901252/11",  "ERI/CN-ESS",  "CG_3"): 1.00,
        ("KRD901252/11",  "ERI/US-EFA",  "CG_5"): 1.00,
        ("KRC4464B1B3B7", "ERI/CN-ESS",  "CG_1"): 0.40,
        ("KRC4464B1B3B7", "ERI/PL-TC",   "CG_1"): 0.60,
        ("KRC4464B1B3B7", "ERI/CN-ESS",  "CG_3"): 1.00,
        ("KRC4464B1B3B7", "ERI/MX-GUAD", "CG_4"): 1.00,
        ("KRC4464B1B3B7", "ERI/CN-EMS",  "CG_6"): 1.00,
    }

    all_plants = plants_df.PlantID.tolist()
    rows = []
    for prod in PRODUCTS:
        for plant in all_plants:
            for cg in CGS:
                key = (prod, plant, cg)
                share = hist_routes.get(key, 0.0)
                rows.append({
                    "ProductID":    prod,
                    "PlantID":      plant,
                    "CG":           cg,
                    "HistFlow":     1 if share > 0 else 0,
                    "HistVolShare": round(share, 2),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "hist_flow.csv", index=False)
    hist_count = (df.HistFlow == 1).sum()
    print(f"  hist_flow.csv       — {len(df)} rows  ({hist_count} historical routes)")
    return df


def gen_coverage_params():
    """
    coverage_params.csv
    ─────────────────────────────────────────────────────
    Type  : Geo | Cat
    Name  : geography code (CN/EU/LAT/US) or category (ESS/EMS)
    Value : minimum number of globally open plants required
            (applied to OpenGlobal[p], not per-product Open[i,p])

    These are strategic policy constraints, configurable per scenario.
    They are NOT hard rules from ERIDOC. Default values reflect the
    baseline supply chain strategy.
    """
    rows = [
        ("Geo", "CN",  1, "At least 1 CN plant globally open"),
        ("Geo", "EU",  1, "At least 1 EU plant globally open"),
        ("Geo", "LAT", 0, "No minimum — LAT is optional"),
        ("Geo", "US",  0, "No minimum — US cost covered by route cost"),
        ("Cat", "ESS", 1, "At least 1 ESS plant (Ericsson-owned)"),
        ("Cat", "EMS", 1, "At least 1 EMS plant (contract manufacturer)"),
    ]
    df = pd.DataFrame(rows, columns=["Type","Name","Value","Description"])
    df.to_csv(DATA / "coverage_params.csv", index=False)
    print(f"  coverage_params.csv — {len(df)} rows")
    return df


def main():
    print("\n" + "═"*56)
    print("  SCD POC v2 — Dummy Data Generator")
    print("  Schema contract for data engineering team")
    print("═"*56)
    print(f"\nWriting to: {DATA}/\n")

    plants_df  = gen_plants()
    products_df = gen_products()
    demand_df  = gen_demand()
    gen_routes()
    gen_route_allowed()
    gen_cg_map()
    gen_hist_flow(demand_df, plants_df)
    gen_coverage_params()

    print(f"\n{'─'*56}")
    print("  All files generated.")
    print("  Replace with real Snowflake data before production use.")
    print(f"{'─'*56}\n")


if __name__ == "__main__":
    main()
