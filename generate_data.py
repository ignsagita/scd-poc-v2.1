#!/usr/bin/env python3
"""
SCD POC v2 — Dummy Data Generator
===================================
Generates all input CSVs with realistic dummy data aligned to a generic
production-grade data warehouse schema.

PURPOSE FOR DATA ENGINEERS:
  Each CSV generated here defines the exact table/column contract
  that must be delivered from the data warehouse before real data can be
  connected.  Column names, data types, and allowed values are
  intentional — do not rename without updating scd_poc_v2.py.

PILOT PRODUCTS (FICTIONAL):
  Product_A  — high-volume product
  Product_B  — broader-coverage product

Plants use the format Plant_1 ... Plant_N. Hubs use Hub_1 ... Hub_N.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["Product_A", "Product_B"]
HUBS     = ["Hub_1","Hub_2","Hub_3","Hub_4","Hub_5","Hub_6","Hub_7"]

LIFECYCLE_YEARS = 4

# ─── PLANT ATTRIBUTES ────────────────────────────────────────────────────────
#   PlantID            : unique plant identifier
#   PlantName          : human-readable factory name
#   Geo                : geographic region (Region_A/B/C/D)
#   Category           : OWN  = company-owned site
#                        EXT  = external / contract-manufactured site
#   IsPilotSite        : 1 = approved for new-product/pilot production
#   IsVolumeSite       : 1 = regular high-volume production site
#   MinProd_ifOpen     : economic minimum lifecycle volume per product if opened
#   FixedCost_MSEK_yr  : annual plant overhead (line setup, qualification, staffing)
#                        NOT already embedded in the per-unit landed cost.
#   Cap_*              : eco-fulfilment capacity per product (placeholder)
# ─────────────────────────────────────────────────────────────────────────────

PLANT_ROWS = [
    # PlantID    Name              Geo         Cat   Pilot Vol  MinProd  FC_yr  Cap_A   Cap_B
    ("Plant_1",  "Factory Alpha",  "Region_A", "OWN", 1,   1,   2000,   2.0, 999999, 999999),
    ("Plant_2",  "Factory Beta",   "Region_A", "EXT", 0,   1,   1500,   1.5, 999999, 999999),
    ("Plant_3",  "Factory Gamma",  "Region_B", "OWN", 1,   1,   1000,   4.0, 999999, 999999),
    ("Plant_4",  "Factory Delta",  "Region_B", "EXT", 1,   0,    800,   3.5, 999999, 999999),
    ("Plant_5",  "Factory Epsilon","Region_B", "EXT", 0,   1,   1200,   3.8, 999999, 999999),
    ("Plant_6",  "Factory Zeta",   "Region_C", "OWN", 0,   1,   1500,   3.5, 999999, 999999),
    ("Plant_7",  "Factory Eta",    "Region_C", "EXT", 0,   1,   1200,   3.0, 999999, 999999),
    ("Plant_8",  "Factory Theta",  "Region_D", "OWN", 0,   1,   1500,   2.5, 999999, 999999),
]

PLANTS_COLS = ["PlantID","PlantName","Geo","Category","IsPilotSite","IsVolumeSite",
               "MinProd_ifOpen","FixedCost_MSEK_yr","Cap_Product_A","Cap_Product_B"]

# ─── LANDED-COST ADDER MATRIX ────────────────────────────────────────────────
# "Landed cost" = total per-unit cost of moving the product from factory to
# the receiving hub. Bundles transportation, customs/duties, inventory cost-of-
# capital, and distribution surcharges into a single multiplier.
# AdderPct[plant, hub] = fraction added on top of base unit cost.
# LCmult = 1 + AdderPct.  None = trade corridor not feasible.
# ─────────────────────────────────────────────────────────────────────────────
#                  Hub_1  Hub_2  Hub_3  Hub_4  Hub_5  Hub_6  Hub_7
ADDER = {
    "Plant_1": [0.18,  0.20,  0.08,  None,  None,  0.06,  0.07],
    "Plant_2": [0.19,  0.21,  0.09,  None,  None,  0.07,  0.08],
    "Plant_3": [0.05,  0.10,  0.18,  0.22,  0.20,  0.12,  0.15],
    "Plant_4": [0.06,  0.11,  0.19,  0.23,  0.21,  0.13,  0.16],
    "Plant_5": [0.07,  0.10,  0.19,  0.22,  0.21,  0.12,  0.15],
    "Plant_6": [0.20,  0.22,  0.22,  0.12,  0.08,  0.20,  0.22],
    "Plant_7": [0.21,  0.23,  0.23,  0.13,  0.09,  0.21,  0.23],
    "Plant_8": [0.22,  None,  None,  0.10,  0.12,  None,  None],
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
    ProductID       : product code (unique key)
    ProductFamily   : grouping label
    ProductCategory : sub-category that determines which adder row applies
                      in the landed-cost tool. Both pilot products share the
                      same category so they share the same adder per route.
    UnitCost_SEK    : base manufacturing cost per unit (a.k.a. ex-works cost)
    Description     : human-readable label
    """
    rows = [
        ("Product_A", "Family_X", "Cat_1", 18000, "Fictional pilot product A"),
        ("Product_B", "Family_X", "Cat_1", 17506, "Fictional pilot product B"),
    ]
    df = pd.DataFrame(rows, columns=["ProductID","ProductFamily","ProductCategory",
                                     "UnitCost_SEK","Description"])
    df.to_csv(DATA / "products.csv", index=False)
    print(f"  products.csv        — {len(df)} rows")
    return df


def gen_demand():
    """
    demand.csv
    ─────────────────────────────────────────────────────
    ProductID : links to products.csv
    Hub       : receiving hub (one-to-one with a customer-facing region)
    Demand    : lifecycle volume (3–5 year total) per product-hub

    Source: long-range forecast aggregated to lifecycle horizon.
    Rows with Demand = 0 are OMITTED.

    Hub_2 and Hub_7 have no demand for either pilot product in this dummy
    set — reflects realistic coverage gaps (not every product covers every
    region).
    """
    rows = [
        ("Product_A", "Hub_1", 30000),
        ("Product_A", "Hub_3",  5000),
        ("Product_A", "Hub_5",  2000),
        ("Product_B", "Hub_1", 40500),
        ("Product_B", "Hub_3",  3600),
        ("Product_B", "Hub_4",   450),
        ("Product_B", "Hub_6",   450),
    ]
    df = pd.DataFrame(rows, columns=["ProductID","Hub","Demand"])
    df.to_csv(DATA / "demand.csv", index=False)
    print(f"  demand.csv          — {len(df)} rows  (zero-demand hubs omitted)")
    return df


def gen_routes():
    """
    routes.csv
    ─────────────────────────────────────────────────────
    PlantID         : factory shipping the product
    Hub             : receiving hub
    ProductCategory : category that determines adder applicability
    AdderPct        : landed-cost adder as decimal (e.g. 0.18 = 18%)
    LCmult          : 1 + AdderPct (convenience column; redundant but useful)

    Source: company landed-cost tool / cost tree.
    Only ALLOWED routes appear in this file.
    """
    rows = []
    plants_list = [r[0] for r in PLANT_ROWS]
    for plant in plants_list:
        for j, hub in enumerate(HUBS):
            adder = ADDER[plant][j]
            if adder is not None:
                rows.append({
                    "PlantID":         plant,
                    "Hub":             hub,
                    "ProductCategory": "Cat_1",
                    "AdderPct":        round(adder, 4),
                    "LCmult":          round(1 + adder, 4),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "routes.csv", index=False)
    print(f"  routes.csv          — {len(df)} rows  (allowed routes only)")
    return df


def gen_route_allowed():
    """
    route_allowed.csv
    ─────────────────────────────────────────────────────
    PlantID      : factory
    Hub          : receiving hub
    RouteAllowed : 1 = trade corridor allowed for all products
                   0 = corridor FORBIDDEN (geopolitical / tariff /
                       regulatory restriction or no logistics path)

    Product-agnostic: a forbidden corridor is forbidden for every product.
    """
    rows = []
    plants_list = [r[0] for r in PLANT_ROWS]
    for plant in plants_list:
        for j, hub in enumerate(HUBS):
            adder = ADDER[plant][j]
            rows.append({
                "PlantID":      plant,
                "Hub":          hub,
                "RouteAllowed": 0 if adder is None else 1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "route_allowed.csv", index=False)
    allowed   = df.RouteAllowed.sum()
    forbidden = (df.RouteAllowed == 0).sum()
    print(f"  route_allowed.csv   — {len(df)} rows  ({allowed} allowed, {forbidden} forbidden)")
    return df


def gen_hubs():
    """
    hubs.csv
    ─────────────────────────────────────────────────────
    Hub        : hub identifier
    HubName    : human-readable hub name
    HubGeo     : geo region the hub serves (used for reporting)
    Region     : commercial region label (free text)

    Used for display/reporting only — NOT a constraint parameter.
    Each hub has a one-to-one mapping with one customer-facing region.
    """
    rows = [
        ("Hub_1","Hub Alpha",   "Region_B", "Region B Primary"),
        ("Hub_2","Hub Beta",    "Region_C", "Region C Secondary"),
        ("Hub_3","Hub Gamma",   "Region_A", "Region A Primary"),
        ("Hub_4","Hub Delta",   "Region_D", "Region D Primary"),
        ("Hub_5","Hub Epsilon", "Region_C", "Region C Primary"),
        ("Hub_6","Hub Zeta",    "Region_A", "Region A Secondary"),
        ("Hub_7","Hub Eta",     "Region_A", "Region A Tertiary"),
    ]
    df = pd.DataFrame(rows, columns=["Hub","HubName","HubGeo","Region"])
    df.to_csv(DATA / "hubs.csv", index=False)
    print(f"  hubs.csv            — {len(df)} rows")
    return df


def gen_hist_flow(demand_df, plants_df):
    """
    hist_flow.csv
    ─────────────────────────────────────────────────────
    ProductID    : links to products.csv
    PlantID      : factory
    Hub          : receiving hub
    HistFlow     : 1 = route used historically; 0 = new route
    HistVolShare : historical volume share across plants for the same
                   (product, hub) pair. Optional reference column.

    For POC v2: DUMMY DATA constructed as a sensible "today's network".
    Real source in production: historical flow description in the data
    warehouse.

    Impact:
      - HistFlow = 0 routes incur a soft penalty in the OM objective.
      - "New-route alerts" raised when allocation > 0 on HistFlow = 0 routes.
    """
    hist_routes = {
        ("Product_A", "Plant_1", "Hub_1"): 0.30,
        ("Product_A", "Plant_3", "Hub_1"): 0.70,
        ("Product_A", "Plant_1", "Hub_3"): 1.00,
        ("Product_A", "Plant_6", "Hub_5"): 1.00,
        ("Product_B", "Plant_1", "Hub_1"): 0.40,
        ("Product_B", "Plant_3", "Hub_1"): 0.60,
        ("Product_B", "Plant_1", "Hub_3"): 1.00,
        ("Product_B", "Plant_8", "Hub_4"): 1.00,
        ("Product_B", "Plant_2", "Hub_6"): 1.00,
    }

    all_plants = plants_df.PlantID.tolist()
    rows = []
    for prod in PRODUCTS:
        for plant in all_plants:
            for hub in HUBS:
                key = (prod, plant, hub)
                share = hist_routes.get(key, 0.0)
                rows.append({
                    "ProductID":    prod,
                    "PlantID":      plant,
                    "Hub":          hub,
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
    Type        : Geo | Cat
    Name        : geo code (Region_A/B/C/D) or category (OWN/EXT)
    Value       : minimum number of globally open plants required
    Description : free-text rationale

    Strategic policy constraints, configurable per scenario.
    """
    rows = [
        ("Geo", "Region_A", 1, "At least 1 plant in Region_A globally open"),
        ("Geo", "Region_B", 1, "At least 1 plant in Region_B globally open"),
        ("Geo", "Region_C", 0, "No minimum — Region_C is optional"),
        ("Geo", "Region_D", 0, "No minimum — Region_D is optional"),
        ("Cat", "OWN",      1, "At least 1 own-site (company-operated) plant"),
        ("Cat", "EXT",      1, "At least 1 external-site (contract-mfg) plant"),
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

    plants_df   = gen_plants()
    products_df = gen_products()
    demand_df   = gen_demand()
    gen_routes()
    gen_route_allowed()
    gen_hubs()
    gen_hist_flow(demand_df, plants_df)
    gen_coverage_params()

    print(f"\n{'─'*56}")
    print("  All files generated.")
    print("  Replace with real data warehouse exports before production use.")
    print(f"{'─'*56}\n")


if __name__ == "__main__":
    main()
