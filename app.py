"""
app.py — SCD POC v2  Streamlit Application
============================================
Run with:  streamlit run app.py

Tabs
----
1. Data         — download templates, upload modified CSVs, preview tables
2. Configuration — scenario setup, parameters with explanations, Run button
3. Results      — KPI cards, Plotly charts, Sankey, alerts, downloads
4. About        — model documentation, constraint reference, data lineage
"""

from __future__ import annotations
import copy, io, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scd_engine import (
    SCDData, SCHEMAS, GEOS, CATS, LIFECYCLE_YEARS, ALPHA_DEFAULT,
    run_b1, run_b2, run_om, run_gm, compute_kpis, validate_dataframe,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SCD POC v2",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

PRODUCT_COLORS  = {"KRD901252/11": "#1565C0", "KRC4464B1B3B7": "#2E7D32"}
SCENARIO_COLORS = {
    "B1-Nearest":   "#546E7A",
    "B2-Baseline":  "#78909C",
    "OM-Optimised": "#1976D2",
    "GM-GeoMix":    "#7B1FA2",
}
GEO_COLORS = {"CN": "#EF5350", "EU": "#42A5F5", "LAT": "#66BB6A", "US": "#FFA726"}
CAT_COLORS = {"ESS": "#0288D1", "EMS": "#F57C00"}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "data":              None,       # SCDData | None
        "data_source":       "default",  # "default" | "uploaded"
        "upload_errors":     {},         # {filename: [error strings]}
        "results":           {},         # {label: kpis_dict}
        "alpha_results":     None,       # sensitivity analysis results
        "demand_adj":        {},         # {cg: float pct}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading default data…")
def _load_default_data() -> SCDData | None:
    try:
        return SCDData(DATA_DIR)
    except Exception:
        return None


def get_data() -> SCDData | None:
    if st.session_state.data is not None:
        return st.session_state.data
    default = _load_default_data()
    if default is not None:
        st.session_state.data = default
    return st.session_state.data

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _results_zip(results: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, kpis in results.items():
            safe = label.replace(" ", "_")
            zf.writestr(f"alloc_{safe}.csv",
                        kpis["alloc_df"].to_csv(index=False))
            zf.writestr(f"plant_summary_{safe}.csv",
                        kpis["plant_df"].to_csv(index=False))
            # alerts
            alert_frames = []
            if not kpis["new_routes_df"].empty:
                nr = kpis["new_routes_df"].copy()
                nr["AlertType"] = "new_scd_flow"
                nr["Detail"]    = nr.apply(
                    lambda r: f"{r.Qty:.0f} units, LCmult={r.LCmult}", axis=1)
                alert_frames.append(
                    nr[["AlertType","ProductID","PlantID","CG","Detail"]])
            if not kpis["consol_df"].empty:
                cs = kpis["consol_df"].copy()
                cs["AlertType"] = "consolidation_candidate"
                cs["PlantID"]   = "ALL"
                cs["CG"]        = "ALL"
                cs["Detail"]    = cs.apply(
                    lambda r: f"Demand {r.TotalDemand} < MinProd {r.MinProd_threshold}", axis=1)
                alert_frames.append(
                    cs[["AlertType","ProductID","PlantID","CG","Detail"]])
            if alert_frames:
                zf.writestr(f"alerts_{safe}.csv",
                            pd.concat(alert_frames).to_csv(index=False))
        # KPI comparison
        rows = []
        for label, k in results.items():
            r = {
                "Scenario":           label,
                "TotalLandedCost_MSEK": k["total_lc_sek"]/1e6,
                "FixedCost_MSEK":     k["fixed_cost_sek"]/1e6,
                "TotalCost_MSEK":     k["total_cost_sek"]/1e6,
                "CostPerUnit_SEK":    k["cost_per_unit_sek"],
                "WeightedLC_Pct":     k["weighted_lc_pct"],
                "N_OpenPlants":       k["n_open_plants"],
            }
            for g in GEOS:
                r[f"Pct_{g}"] = k["geo_pct"].get(g, 0)
            rows.append(r)
        zf.writestr("kpi_comparison.csv", pd.DataFrame(rows).to_csv(index=False))
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _chart_cost_comparison(results: dict) -> go.Figure:
    labels = list(results.keys())
    lc     = [results[l]["total_lc_sek"]/1e6 for l in labels]
    fc     = [results[l]["fixed_cost_sek"]/1e6 for l in labels]
    total  = [l + f for l, f in zip(lc, fc)]
    colors = [SCENARIO_COLORS.get(l, "#607D8B") for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Landed Cost", x=labels, y=lc,
        marker_color=colors,
        text=[f"{v:.1f}" for v in lc],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        name="Fixed Plant Cost", x=labels, y=fc,
        marker_color="#90A4AE",
        text=[f"{v:.1f}" for v in fc],
        textposition="inside", insidetextanchor="middle",
    ))
    # Annotate total on top of each stacked bar
    for i, (x_val, t_val) in enumerate(zip(labels, total)):
        fig.add_annotation(
            x=x_val, y=t_val, text=f"<b>{t_val:.1f}</b>",
            showarrow=False, yshift=8, font=dict(size=12),
        )
    fig.update_layout(
        barmode="stack", title="Total Lifecycle Cost (MSEK)",
        yaxis_title="MSEK",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420, margin=dict(t=80, b=20, l=60, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


def _chart_geo(results: dict) -> go.Figure:
    labels = list(results.keys())
    fig = go.Figure()
    for geo in GEOS:
        vals = [results[l]["geo_pct"].get(geo, 0) for l in labels]
        fig.add_trace(go.Bar(
            name=geo, x=labels, y=vals,
            marker_color=GEO_COLORS[geo],
            text=[f"{v:.0f}%" for v in vals],
            textposition="inside", insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack", title="Volume by Geography (%)",
        yaxis_title="% of Lifecycle Volume",
        yaxis_range=[0, 115],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420, margin=dict(t=80, b=20, l=60, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


def _chart_sankey(om_kpis: dict, baseline_kpis: dict, data: SCDData) -> go.Figure:
    """
    Diff Sankey: OM vs a selected baseline.
    Link colours:
      Blue   — route in BOTH OM and baseline (continuing)
      Green  — route in OM only (new proposal)
      Red    — route in baseline only (dropped by OM)
    """
    def _agg(kpis):
        df = kpis["alloc_df"]
        return (df.groupby(["PlantID","CG"])["Qty"].sum()
                  .reset_index()
                  .set_index(["PlantID","CG"])["Qty"].to_dict())

    om_flows = _agg(om_kpis)
    bl_flows = _agg(baseline_kpis)

    all_flows = set(om_flows) | set(bl_flows)

    # Only include plants/CGs that appear in at least one flow
    active_plants = sorted({p for p,g in all_flows})
    active_cgs    = sorted({g for p,g in all_flows})

    plant_idx = {p: i for i, p in enumerate(active_plants)}
    cg_idx    = {g: len(active_plants) + i for i, g in enumerate(active_cgs)}

    # Spread nodes evenly: plants left (x=0.01), CGs right (x=0.99)
    n_p = len(active_plants)
    n_g = len(active_cgs)
    plant_y = [round(i / max(n_p - 1, 1), 4) for i in range(n_p)]
    cg_y    = [round(i / max(n_g - 1, 1), 4) for i in range(n_g)]

    # Node labels & colours
    cg_ma = dict(zip(data.cg_map.CG, data.cg_map.MarketArea))
    node_labels = (
        [f"{p}<br>{data.plant_geo.get(p,'?')} | {data.plant_cat.get(p,'?')}"
         for p in active_plants] +
        [f"{g}<br>{cg_ma.get(g,'')}" for g in active_cgs]
    )
    node_colors = (
        ["#1565C0"] * n_p +   # plants → blue
        ["#2E7D32"] * n_g     # CGs    → green
    )

    sources, targets, values, link_colors, custom = [], [], [], [], []

    CONTINUING = "rgba(100,149,237,0.55)"
    NEW_ROUTE  = "rgba(46,125,50,0.65)"
    DROPPED    = "rgba(183,28,28,0.45)"

    for plant, cg in sorted(all_flows):
        in_om = (plant, cg) in om_flows
        in_bl = (plant, cg) in bl_flows
        qty   = om_flows.get((plant, cg), bl_flows.get((plant, cg), 0))

        if in_om and in_bl:
            color = CONTINUING
            tag   = "Continuing"
        elif in_om:
            color = NEW_ROUTE
            tag   = "NEW in OM"
        else:
            qty   = bl_flows.get((plant, cg), 0) * 0.25  # thin ghost
            color = DROPPED
            tag   = "Dropped by OM"

        sources.append(plant_idx[plant])
        targets.append(cg_idx[cg])
        values.append(max(qty, 200))   # floor so thin routes stay visible
        link_colors.append(color)
        custom.append(
            f"{plant} → {cg}<br>"
            f"Volume: {om_flows.get((plant,cg),0):,.0f} units (OM)<br>"
            f"Baseline: {bl_flows.get((plant,cg),0):,.0f} units<br>"
            f"Status: {tag}"
        )

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=25, thickness=20,
            label=node_labels,
            color=node_colors,
            x=[0.08]*n_p + [0.88]*n_g,
            y=plant_y + cg_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=f"Network Flow — Optimised (OM) vs {baseline_kpis['label']}",
        height=620,
        margin=dict(l=140, r=160, t=60, b=20),
        font=dict(size=12),
    )
    return fig


def _chart_alpha_sensitivity(alpha_results: list[dict]) -> go.Figure:
    alphas  = [r["alpha"] for r in alpha_results]
    costs   = [r["total_lc_sek"]/1e6 for r in alpha_results]
    plants  = [r["n_open_plants"] for r in alpha_results]
    new_rts = [len(r["new_routes_df"]) for r in alpha_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=alphas, y=costs, name="Landed Cost (MSEK)",
        mode="lines+markers", line=dict(color="#1976D2", width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=alphas, y=plants, name="# Open Plants",
        mode="lines+markers", line=dict(color="#E65100", width=2, dash="dash"),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=alphas, y=new_rts, name="New-Route Alerts",
        mode="lines+markers", line=dict(color="#2E7D32", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Alpha (α) Sensitivity — Effect on Cost & Network Structure",
        xaxis=dict(title="α value", tickformat=".3f"),
        yaxis=dict(title="Landed Cost (MSEK)", side="left"),
        yaxis2=dict(title="Count", side="right", overlaying="y"),
        legend=dict(orientation="h", y=1.1),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=70, b=40),
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


def _chart_demand_check(results: dict, data: SCDData) -> go.Figure:
    """Small multiples: demand vs allocated per (product, CG) per scenario."""
    scenarios = list(results.keys())
    products  = data.products_list

    n_cols = len(scenarios)
    n_rows = len(products)

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{s}<br>{p[:18]}" for p in products for s in scenarios],
        shared_yaxes=False, vertical_spacing=0.12, horizontal_spacing=0.06,
    )

    for pi, prod in enumerate(products):
        active = [(i,g) for (i,g) in data.active_ig if i == prod]
        cgs    = [g for _,g in active]

        for si, (label, kpis) in enumerate(results.items()):
            d_vals = [data.demand_map.get((prod,cg),0)/1000 for cg in cgs]
            a_vals = []
            for cg in cgs:
                sub = kpis["alloc_df"]
                qty = sub[(sub.ProductID==prod)&(sub.CG==cg)].Qty.sum()
                a_vals.append(qty/1000)

            r, c = pi + 1, si + 1
            color = PRODUCT_COLORS.get(prod, "#888")
            fig.add_trace(
                go.Bar(name="Demand", x=cgs, y=d_vals, marker_color="#90CAF9",
                       showlegend=(r == 1 and c == 1)),
                row=r, col=c,
            )
            fig.add_trace(
                go.Bar(name="Allocated", x=cgs, y=a_vals, marker_color=color,
                       opacity=0.85, showlegend=(r == 1 and c == 1)),
                row=r, col=c,
            )

    fig.update_layout(
        title="Demand Satisfaction by Scenario & Product (k units)",
        barmode="group", height=280 * n_rows,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════

def render_data_tab():
    st.header("Input Data")
    st.caption(
        "Download the CSV templates, edit them in Excel or any spreadsheet tool, "
        "then upload your modified files. Only upload files you have changed — "
        "the rest default to the built-in dummy dataset."
    )

    data = get_data()

    # ── Current data status ───────────────────────────────────────────────────
    source_badge = (
        "**Uploaded data active**"
        if st.session_state.data_source == "uploaded"
        else "**Default dummy data active**"
    )
    st.info(source_badge + "  ·  Replace any file below to customise.")

    if data:
        s = data.summary_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plants",       s["plants"])
        c2.metric("Products",     len(s["products"]))
        c3.metric("Active CG-pairs", s["active_pairs"])
        c4.metric("Total Demand (k)", f"{s['total_demand']/1000:.0f}k")

    st.divider()

    # ── Download templates ────────────────────────────────────────────────────
    st.subheader("Download Templates")
    st.caption("Each file includes only the columns the model expects. Column descriptions are below each download button.")

    file_cols = st.columns(4)
    for idx, (fname, schema) in enumerate(SCHEMAS.items()):
        col = file_cols[idx % 4]
        with col:
            # Load current data for this file
            try:
                df = pd.read_csv(DATA_DIR / fname)
                csv_bytes = _df_to_csv_bytes(df)
            except Exception:
                csv_bytes = b""
            col.download_button(
                label=fname,
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{fname}",
            )
            with col.expander("Column guide", expanded=False):
                st.caption(schema["doc"])
                req_str = ", ".join([f"`{c}`" for c in schema["required"]])
                st.markdown(f"**Required columns:** {req_str}")
                if schema.get("choices"):
                    for col_name, vals in schema["choices"].items():
                        st.markdown(f"**`{col_name}`** must be one of: `{vals}`")
                if schema.get("non_neg"):
                    st.markdown(f"**Must be ≥ 0:** {schema['non_neg']}")

    st.divider()

    # ── Upload section ────────────────────────────────────────────────────────
    st.subheader("Upload Modified Files")
    st.caption(
        "Upload one or more CSV files. Each file is validated before being accepted. "
        "Unmodified files continue using the built-in defaults."
    )

    uploaded = st.file_uploader(
        "Drag and drop CSV files here (or click to browse)",
        type="csv",
        accept_multiple_files=True,
        key="csv_uploader",
    )

    if uploaded:
        all_valid = True
        new_dfs: dict[str, pd.DataFrame] = {}

        for f in uploaded:
            fname = f.name
            if fname not in SCHEMAS:
                st.error(f"{fname} — not a recognised input file. "
                         f"Expected one of: {list(SCHEMAS.keys())}")
                all_valid = False
                continue

            df = pd.read_csv(f)
            errs = validate_dataframe(fname, df)

            if errs:
                st.error(f"{fname} — {len(errs)} validation error(s):")
                for e in errs:
                    st.markdown(f"  - {e}")
                all_valid = False
            else:
                new_dfs[fname] = df
                st.success(f"{fname} — {len(df)} rows, all checks passed")

        if all_valid and new_dfs:
            if st.button("Apply Uploaded Files", type="primary"):
                with st.spinner("Building data model from uploaded files…"):
                    try:
                        new_data = SCDData.from_dataframes(new_dfs)
                        st.session_state.data        = new_data
                        st.session_state.data_source = "uploaded"
                        st.session_state.results     = {}   # clear stale results
                        st.session_state.alpha_results = None
                        st.success("Data updated. Previous results cleared — re-run scenarios in the Configuration tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to build data model: {e}")
        elif not all_valid:
            st.warning("Fix the errors above before applying.")

    if st.session_state.data_source == "uploaded":
        if st.button("Reset to Default Data"):
            st.session_state.data        = None
            st.session_state.data_source = "default"
            st.session_state.results     = {}
            st.session_state.alpha_results = None
            st.rerun()

    st.divider()

    # ── Data preview ──────────────────────────────────────────────────────────
    if data:
        st.subheader("Preview Current Data")
        preview_file = st.selectbox("Select file to preview", list(SCHEMAS.keys()),
                                    key="preview_select")
        file_map = {
            "plants.csv":          data.plants,
            "products.csv":        data.products,
            "demand.csv":          data.demand,
            "routes.csv":          data.routes,
            "route_allowed.csv":   data.allowed,
            "cg_map.csv":          data.cg_map,
            "hist_flow.csv":       data.hist,
            "coverage_params.csv": data.coverage,
        }
        preview_df = file_map.get(preview_file, pd.DataFrame())
        st.dataframe(preview_df, use_container_width=True, height=320)
        st.caption(f"{len(preview_df)} rows · {len(preview_df.columns)} columns")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def render_config_tab():
    st.header("Scenario Configuration")

    data = get_data()
    if data is None:
        st.error("No data loaded. Check the Data tab first.")
        return

    st.caption(
        "Select which scenarios to run and configure their parameters. "
        "Click Run Scenarios at the bottom when ready."
    )

    # ── Demand adjustments (what-if) ──────────────────────────────────────────
    with st.expander("What-If Demand Adjustment (optional)", expanded=False):
        st.caption(
            "Adjust lifecycle demand per hub by a percentage before running. "
            "Applies to ALL scenarios and ALL products for that hub. "
            "Useful for stress-testing: 'what if EMEA demand grows 20%?'"
        )
        adj_cols = st.columns(len(data.cgs_list))
        adj: dict[str, float] = {}
        cg_ma = dict(zip(data.cg_map.CG, data.cg_map.MarketArea))
        for idx, cg in enumerate(data.cgs_list):
            base_demand = sum(
                v for (i,g),v in data.demand_map.items() if g == cg
            )
            label = f"**{cg}**\n{cg_ma.get(cg,'')}"
            if base_demand == 0:
                adj_cols[idx].caption(f"{cg}\n*(no demand)*")
                adj[cg] = 0.0
            else:
                pct = adj_cols[idx].slider(
                    label, min_value=-50, max_value=50, value=0, step=5,
                    format="%d%%", key=f"adj_{cg}",
                    help=f"Base demand: {base_demand:,.0f} units",
                )
                adj[cg] = float(pct)
        st.session_state.demand_adj = adj

        active_adj = {cg: v for cg,v in adj.items() if v != 0}
        if active_adj:
            st.info("Active adjustments: " + ", ".join(
                f"{cg}: {v:+.0f}%" for cg,v in active_adj.items()
            ))

    st.divider()

    # ── Scenario checkboxes ───────────────────────────────────────────────────
    st.subheader("Select Scenarios to Run")
    sc1, sc2, sc3, sc4 = st.columns(4)
    run_b1_flag = sc1.checkbox("**B1** — Nearest-Plant", value=True,  key="cb_b1",
                               help="Assigns each hub to the cheapest allowed plant. Deterministic.")
    run_b2_flag = sc2.checkbox("**B2** — Baseline Plant", value=True, key="cb_b2",
                               help="All demand through one reference plant. Used as TPI cost-avoidance benchmark.")
    run_om_flag = sc3.checkbox("**OM** — Optimised (MILP)", value=True, key="cb_om",
                               help="Minimises total landed cost subject to all constraints.")
    run_gm_flag = sc4.checkbox("**GM** — Geo-Mix", value=True, key="cb_gm",
                               help="You specify CN%/EU%/LAT%/US% — cheapest plant in each geo is used.")

    if not any([run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag]):
        st.warning("Select at least one scenario.")
        return

    st.divider()

    # ── Per-scenario parameters ───────────────────────────────────────────────
    cfg: dict = {}

    if run_b2_flag:
        st.subheader("B2 — Baseline Plant")
        st.caption(
            "All lifecycle demand for each product is assigned to this single plant. "
            "Used as the **TPI cost-avoidance reference** — the SCB compares OM savings against this scenario."
        )
        b2_plants = {}
        b2_cols = st.columns(len(data.products_list))
        for idx, prod in enumerate(data.products_list):
            options = data.plants_list
            labels  = [f"{p}  ({data.plant_name[p]}, {data.plant_geo[p]}, {data.plant_cat[p]})"
                       for p in options]
            # Default: TC Poland (ERI/PL-TC) if available
            default_idx = next((i for i,p in enumerate(options) if "PL" in p), 0)
            sel = b2_cols[idx].selectbox(
                f"Baseline for {prod}", options=options,
                index=default_idx, format_func=lambda p: f"{p} ({data.plant_name[p]})",
                key=f"b2_plant_{prod}",
            )
            b2_plants[prod] = sel
        cfg["b2_baseline_plant"] = b2_plants
        st.divider()

    if run_om_flag:
        st.subheader("OM — Optimised MILP Parameters")

        # Alpha penalty
        st.markdown("##### Historical Route Penalty (α)")
        st.caption(
            "Controls how strongly the model prefers **historical routes** over new ones. "
            "The penalty added per unit on a non-historical route = α × mean(UnitCost)."
        )
        alpha_val = st.slider(
            "α value", min_value=0.0, max_value=0.05,
            value=ALPHA_DEFAULT, step=0.005, format="%.3f",
            key="om_alpha",
        )
        penalty_sek = alpha_val * (
            sum(data.unit_cost.values()) / len(data.unit_cost) if data.unit_cost else 17000
        )
        equiv_pct = alpha_val * 100
        c_left, c_right = st.columns(2)
        c_left.info(
            f"**Equivalent to ≈ {equiv_pct:.1f}% LC difference per unit** on non-historical routes.  \n"
            f"At α=0: pure cost optimisation — freely uses new routes.  \n"
            f"At α=0.05: strong preference for history — only switches routes if saving > 5%."
        )
        c_right.metric("Penalty per non-hist. unit", f"{penalty_sek:,.0f} SEK")

        cfg["om_alpha"] = alpha_val

        # Coverage constraints
        st.markdown("##### Geo & Category Coverage (optional)")
        st.caption(
            "Set minimum number of open plants per geography and category. "
            "These are **strategic policy constraints** — not ERIDOC rules. "
            "Set to 0 to disable a constraint."
        )
        use_cov = st.checkbox("Enable coverage constraints", value=True, key="om_use_cov")
        cfg["om_use_coverage"] = use_cov

        if use_cov:
            cov_cols = st.columns(6)
            geo_par, cat_par = {}, {}
            for idx, geo in enumerate(GEOS):
                geo_par[geo] = cov_cols[idx].number_input(
                    f"Min {geo} plants", min_value=0, max_value=5,
                    value=data.geo_params.get(geo, 0),
                    key=f"geo_{geo}",
                )
            for idx, cat in enumerate(CATS):
                cat_par[cat] = cov_cols[4 + idx].number_input(
                    f"Min {cat} plants", min_value=0, max_value=5,
                    value=data.cat_params.get(cat, 0),
                    key=f"cat_{cat}",
                )
            cfg["om_geo_params"] = geo_par
            cfg["om_cat_params"] = cat_par

        st.divider()

    if run_gm_flag:
        st.subheader("GM — Geo-Mix Configuration")
        st.caption(
            "Specify what **percentage of total volume** should come from each geography. "
            "Within each geo, the **cheapest allowed plant per hub** is used. "
            "**Percentages must sum to 100.**"
        )
        gm_cols = st.columns(4)
        gm_vals: dict[str, float] = {}
        for idx, geo in enumerate(GEOS):
            defaults = {"CN": 30, "EU": 70, "LAT": 0, "US": 0}
            gm_vals[geo] = float(gm_cols[idx].number_input(
                f"{geo} %", min_value=0.0, max_value=100.0,
                value=float(defaults[geo]), step=5.0,
                key=f"gm_{geo}",
            ))
        gm_total = sum(gm_vals.values())
        if abs(gm_total - 100) > 0.01:
            st.error(f"⚠ Geo-mix percentages sum to **{gm_total:.1f}%** — must equal 100%.")
            run_gm_flag = False
        else:
            st.success(f"Total = {gm_total:.0f}% — valid")
        cfg["gm_geo_mix"] = gm_vals
        st.divider()

    # ── Run button ─────────────────────────────────────────────────────────────
    st.subheader("▶ Run")
    if st.button("🚀 Run Selected Scenarios", type="primary", use_container_width=True):
        _run_scenarios(data, cfg, run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag)


def _run_scenarios(data: SCDData, cfg: dict,
                   run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag):
    """Execute selected scenarios, store results in session_state."""
    # Apply demand adjustments
    adj = st.session_state.demand_adj
    if any(v != 0 for v in adj.values()):
        data = data.with_demand_adjustments(adj)
        st.info(f"Demand adjustments applied: {adj}")

    results = {}
    progress = st.progress(0, text="Preparing…")
    total = sum([run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag])
    done  = 0

    def _step(label):
        nonlocal done
        done += 1
        progress.progress(done / total, text=f"Running {label}…")

    try:
        if run_b1_flag:
            _step("B1 Nearest-Plant")
            alloc, alerts = run_b1(data, {})
            results["B1-Nearest"]  = compute_kpis(alloc, data, "B1-Nearest", alerts=alerts)

        if run_b2_flag:
            _step("B2 Baseline Plant")
            b2_cfg = {"baseline_plant": cfg.get("b2_baseline_plant", {})}
            alloc, alerts = run_b2(data, b2_cfg)
            results["B2-Baseline"] = compute_kpis(alloc, data, "B2-Baseline", alerts=alerts)

        if run_om_flag:
            _step("OM Optimised (MILP — may take 30–60 seconds)")
            om_cfg = {
                "alpha":        cfg.get("om_alpha", ALPHA_DEFAULT),
                "use_coverage": cfg.get("om_use_coverage", True),
                "geo_params":   cfg.get("om_geo_params", data.geo_params),
                "cat_params":   cfg.get("om_cat_params", data.cat_params),
            }
            with st.spinner("Solving MILP… (CBC solver, up to 5 min timeout)"):
                alloc, meta = run_om(data, om_cfg)
            results["OM-Optimised"] = compute_kpis(
                alloc, data, "OM-Optimised", meta=meta)

        if run_gm_flag:
            _step("GM Geo-Mix")
            gm_cfg = {"geo_mix": cfg.get("gm_geo_mix", {"CN":30,"EU":70,"LAT":0,"US":0})}
            alloc, alerts = run_gm(data, gm_cfg)
            results["GM-GeoMix"] = compute_kpis(alloc, data, "GM-GeoMix", alerts=alerts)

        progress.progress(1.0, text="Done")
        st.session_state.results = results
        st.success(f"{len(results)} scenario(s) complete. Switch to the Results tab to explore.")

    except Exception as e:
        progress.empty()
        st.error(f"Run failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def render_results_tab():
    st.header("Results")

    results = st.session_state.results
    data    = get_data()

    if not results:
        st.info("No results yet. Configure scenarios in the Configuration tab and click Run Scenarios.")
        return
    if data is None:
        st.error("Data not loaded.")
        return

    # ── Scenario legend ───────────────────────────────────────────────────────
    st.caption(
        "**B1-Nearest** = demand assigned to cheapest allowed plant per hub (deterministic)  |  "
        "**B2-Baseline** = all demand through one reference plant — used as the cost benchmark  |  "
        "**OM-Optimised** = MILP solution minimising total cost subject to all constraints  |  "
        "**GM-GeoMix** = user-specified geography volume split (e.g. 30% CN / 70% EU)"
    )

    # ── KPI glossary ──────────────────────────────────────────────────────────
    with st.expander("KPI definitions — click to expand", expanded=False):
        st.markdown("""
| KPI | Definition |
|-----|-----------|
| **Total Cost (MSEK)** | Landed cost + fixed plant overhead across all products and hubs for the full lifecycle horizon. This is the primary cost figure for SCB. |
| **Landed Cost (MSEK)** | Volume × LC multiplier × unit cost (TK) per route, summed across all products. Excludes fixed plant overhead. |
| **Fixed Cost (MSEK)** | Annual plant overhead (line setup, qualification, staffing) × lifecycle years, for each open plant. This cost is incurred regardless of volume. |
| **Cost / Unit (SEK)** | Total landed cost divided by total lifecycle volume. Volume-weighted average cost per unit shipped. |
| **Weighted LC %** | Volume-weighted average of the landed cost multiplier (LCmult = 1 + Adder%) across all active routes. A lower % means cheaper routing overall. |
| **# Open Plants** | Number of distinct plants used in this network design. More plants = more operational complexity and higher fixed cost. |
| **TPI Avoidance vs B2** | Cost saving vs the B2 Baseline Plant scenario. **Positive = savings** (this scenario is cheaper than B2). **Negative = extra cost** (this scenario is more expensive than B2). The primary KPI presented to the Supply Chain Board. |
| **New-Route Alerts** | Number of plant-to-hub flows proposed that have no historical precedent. Each alert requires SCB review before implementation. |
""")

    st.divider()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    b2_cost = results.get("B2-Baseline", {}).get("total_cost_sek")

    kpi_cols = st.columns(len(results))
    for col, (label, k) in zip(kpi_cols, results.items()):
        if b2_cost and label != "B2-Baseline":
            val = (b2_cost - k["total_cost_sek"]) / 1e6
            avoidance = f"{val:+.1f} MSEK"
        else:
            avoidance = "— (reference)"
        col.metric("Scenario",           label)
        col.metric("Total Cost (MSEK)",  f"{k['total_cost_sek']/1e6:.1f}",
                   help="Landed cost + fixed plant overhead. Primary SCB cost figure.")
        col.metric("Landed Cost (MSEK)", f"{k['total_lc_sek']/1e6:.1f}",
                   help="Volume × LCmult × UnitCost per route. Excludes fixed plant overhead.")
        col.metric("Fixed Cost (MSEK)",  f"{k['fixed_cost_sek']/1e6:.1f}",
                   help="Annual plant overhead × lifecycle years, for each open plant.")
        col.metric("Cost / Unit (SEK)",  f"{k['cost_per_unit_sek']:,.0f}",
                   help="Total landed cost ÷ total lifecycle volume.")
        col.metric("Weighted LC %",      f"{k['weighted_lc_pct']:.2f}%",
                   help="Volume-weighted average LC multiplier across all active routes. Lower is better.")
        col.metric("# Open Plants",      k["n_open_plants"],
                   help="Number of distinct plants used. More plants = more complexity and fixed cost.")
        col.metric("TPI Avoidance vs B2", avoidance,
                   help="Positive = cheaper than B2 baseline (cost saving). Negative = more expensive than B2.")
        col.metric("New-Route Alerts",   len(k["new_routes_df"]),
                   help="Flows with no historical precedent. Each requires SCB review.")

    st.divider()

    # ── Cost & Geo Charts ─────────────────────────────────────────────────────
    st.subheader("Cost & Geography Breakdown")
    st.caption(
        "Left chart: total lifecycle cost split into landed cost (coloured) and fixed plant overhead (grey). "
        "Right chart: share of total volume produced in each geography."
    )
    ch1, ch2 = st.columns(2)
    ch1.plotly_chart(_chart_cost_comparison(results), use_container_width=True)
    ch2.plotly_chart(_chart_geo(results),             use_container_width=True)

    st.divider()

    # ── Sankey ────────────────────────────────────────────────────────────────
    st.subheader("Network Flow Diagram")
    st.caption(
        "Shows which plants supply which hubs, and how the Optimised (OM) network differs "
        "from a selected baseline. Link width represents volume. Hover over any link for details."
    )
    if "OM-Optimised" not in results:
        st.info("Run the Optimised (OM) scenario to see the network flow diagram.")
    else:
        baseline_options = [l for l in results if l != "OM-Optimised"]
        if not baseline_options:
            st.caption("Run at least one baseline scenario (B1 or B2) to compare against.")
        else:
            compare_to = st.selectbox(
                "Compare Optimised (OM) result against:",
                baseline_options, key="sankey_baseline",
            )
            st.caption(
                "Blue = route continues from the selected baseline  |  "
                "Green = new route proposed by OM (no historical precedent — check Alerts tab)  |  "
                "Red = route that existed in baseline but dropped by OM (shown thin).  "
                "Hover over any link for exact volumes."
            )
            fig_sankey = _chart_sankey(
                results["OM-Optimised"], results[compare_to], data
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

    st.divider()

    # ── Demand satisfaction ───────────────────────────────────────────────────
    with st.expander("Demand Satisfaction Check", expanded=False):
        st.caption("Demand = Allocated confirms constraint C1 is satisfied for all scenarios.")
        st.plotly_chart(_chart_demand_check(results, data), use_container_width=True)

    st.divider()

    # ── Alpha Sensitivity ─────────────────────────────────────────────────────
    with st.expander("Alpha (α) Sensitivity Analysis", expanded=False):
        st.caption(
            "Re-runs the OM scenario at multiple α values to show how the penalty weight "
            "affects total cost, number of open plants, and new-route alerts."
        )
        if st.button("▶ Run Alpha Sensitivity Analysis", key="run_alpha"):
            alpha_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
            alpha_kpis = []
            prog = st.progress(0, text="Running sensitivity…")
            base_cfg = {
                "alpha":        0.01,
                "use_coverage": True,
                "geo_params":   data.geo_params,
                "cat_params":   data.cat_params,
            }
            for i, av in enumerate(alpha_vals):
                prog.progress((i+1)/len(alpha_vals), text=f"α = {av:.3f}…")
                cfg_i = {**base_cfg, "alpha": av}
                try:
                    alloc, meta = run_om(data, cfg_i)
                    k = compute_kpis(alloc, data, f"α={av}", meta=meta)
                    k["alpha"] = av
                    alpha_kpis.append(k)
                except Exception as e:
                    st.warning(f"α={av} failed: {e}")
            prog.empty()
            st.session_state.alpha_results = alpha_kpis
            st.success("Done.")

        if st.session_state.alpha_results:
            st.plotly_chart(
                _chart_alpha_sensitivity(st.session_state.alpha_results),
                use_container_width=True,
            )
            # Tabular summary
            summary = pd.DataFrame([
                {
                    "α": r["alpha"],
                    "Landed Cost (MSEK)": round(r["total_lc_sek"]/1e6, 1),
                    "Fixed Cost (MSEK)":  round(r["fixed_cost_sek"]/1e6, 1),
                    "Total Cost (MSEK)":  round(r["total_cost_sek"]/1e6, 1),
                    "# Open Plants":      r["n_open_plants"],
                    "New-Route Alerts":   len(r["new_routes_df"]),
                }
                for r in st.session_state.alpha_results
            ])
            st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── Alerts Panel ──────────────────────────────────────────────────────────
    st.subheader("Alerts")
    alert_tab_labels = list(results.keys())
    alert_tabs = st.tabs(alert_tab_labels)

    for tab, (label, k) in zip(alert_tabs, results.items()):
        with tab:
            # Run-time alerts (routing fallbacks)
            if k.get("run_alerts"):
                with st.expander(f"Routing fallbacks ({len(k['run_alerts'])})", expanded=True):
                    for a in k["run_alerts"]:
                        st.warning(a)

            # New-route alerts
            if not k["new_routes_df"].empty:
                st.markdown("**New SCD Flows (no historical precedent)**")
                st.caption(
                    "These routes exist in the proposed network but were NOT used historically. "
                    "Review with SCB before approving — each is a change-management signal."
                )
                nr = k["new_routes_df"][["ProductID","PlantID","CG","Qty","LCmult"]].copy()
                nr["Qty"] = nr["Qty"].round().astype(int)
                st.dataframe(nr, use_container_width=True, hide_index=True)
            else:
                st.success("No new-route alerts — all flows match historical patterns.")

            # Consolidation alerts
            if not k["consol_df"].empty:
                st.markdown("**Consolidation Candidates**")
                st.caption(
                    "Products whose total lifecycle demand is below the MinProd_ifOpen threshold. "
                    "Consider consolidating production to a single plant."
                )
                st.dataframe(k["consol_df"], use_container_width=True, hide_index=True)

    st.divider()

    # ── Allocation Detail Tables ───────────────────────────────────────────────
    with st.expander("Full Allocation Tables", expanded=False):
        detail_scenario = st.selectbox(
            "Show allocation for:", list(results.keys()), key="detail_scen"
        )
        kpis_detail = results[detail_scenario]
        st.dataframe(
            kpis_detail["alloc_df"],
            use_container_width=True, height=400,
        )
        st.markdown("**Plant Summary**")
        st.dataframe(kpis_detail["plant_df"], use_container_width=True, hide_index=True)

    st.divider()

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("Download Results")
    dc1, dc2 = st.columns(2)

    dc1.download_button(
        "Download All Results (ZIP)",
        data=_results_zip(results),
        file_name="scd_poc_v2_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

    dl_scen = dc2.selectbox("Or download single scenario:", list(results.keys()),
                             key="dl_single_sel")
    dc2.download_button(
        f"Download {dl_scen} — alloc.csv",
        data=_df_to_csv_bytes(results[dl_scen]["alloc_df"]),
        file_name=f"alloc_{dl_scen}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about_tab():
    st.header("About This Model")

    st.subheader("Business Purpose")
    st.markdown(
        "This tool answers: **How does a change in supply chain setup impact landed cost "
        "at the product level?** It simulates alternative plant-to-hub network designs and "
        "computes cost impact and risk indicators to support Supply Chain Board (SCB) decisions."
    )

    st.subheader("Optimisation Objective (OM Scenario)")
    st.latex(r"""
    \min \sum_{i,p,g} \text{Alloc}_{i,p,g} \cdot \text{LCmult}_{p,g} \cdot \text{UnitCost}_i
         + \sum_p \text{FixedCost}_p \cdot T \cdot \text{OpenGlobal}_p
         + \alpha \cdot \overline{UC} \cdot \sum_{\{i,p,g:\,\text{HistFlow}=0\}} \text{Alloc}_{i,p,g}
    """)
    st.caption("Where T = lifecycle years. The third term penalises volume on non-historical routes.")

    st.subheader("Constraints")
    constraint_data = {
        "ID": ["C1","C2","C3","C4","C5","C6","C7","C8"],
        "Name": [
            "Demand Satisfaction","Route Feasibility","Plant Feasibility",
            "Allocation Gating","Min Production","CG Coverage",
            "Geo/Category Coverage","Capacity",
        ],
        "Type": ["Hard","Hard","Hard","Hard","Hard","Hard","Optional","Placeholder"],
        "Description": [
            "All lifecycle demand per product and hub must be fully served.",
            "No flow on geopolitically forbidden routes (RouteAllowed=0). Product-agnostic.",
            "A route can only be used if the plant is open for that product.",
            "Volume can only flow on an open route (big-M link: Alloc ≤ M × RouteOpen).",
            "If a plant is open for a product, it must produce ≥ MinProd_ifOpen units total.",
            "Every (product, hub) pair with demand > 0 must have ≥ 1 supplying route.",
            "Min open plants per geography (CN/EU/LAT/US) and category (ESS/EMS). User-configurable.",
            "Eco-fulfilment capacity per product-plant. Set to 999,999 (non-binding) in POC v2.",
        ],
    }
    st.dataframe(pd.DataFrame(constraint_data), use_container_width=True, hide_index=True)

    st.subheader("Data Lineage")
    st.markdown("""
| File | Content | Production Source |
|------|---------|-------------------|
| `plants.csv` | Plant master (SCN code, geo, category, costs) | Plant/site master → Snowflake |
| `products.csv` | Product TK and category | CRM cost tree / product costing |
| `demand.csv` | Lifecycle demand per product and hub | MAF + LRFP — *source TBD with SuPM* |
| `routes.csv` | LC adder per plant-hub-category | Henrik's landed cost tool |
| `route_allowed.csv` | Geopolitical route feasibility | ERIDOC BNEW-24:038086Uen (Owner: Clovis Hiroshi Kawai) |
| `cg_map.csv` | Customer group → market area → hub | MA/hub master data |
| `hist_flow.csv` | Historical flow flags | *Amrith's SCD flow description — Snowflake table TBD* |
| `coverage_params.csv` | Coverage constraint defaults | User/scenario configuration |
""")

    st.subheader("Open Points (Before Real Data Connection)")
    st.markdown("""
1. **Demand source** — confirm LRFP vs MAF (or combination) for lifecycle volume per hub. Owner: SuPM team.
2. **MinProd_ifOpen values** — collect realistic minimums from factory SMEs (current placeholder: 500–2,000 units).
3. **Historical flow data** — identify Snowflake table for `hist_flow.csv`. Owner: Amrith.
""")

    st.subheader("Known Simplifications (POC v2 vs Production)")
    simplification_data = {
        "Dimension": ["Products","Time horizon","Echelon","Capacity","Fixed plant cost",
                      "Allocation type","Historical flows","Lead time"],
        "POC v2": ["2 Radio products (dummy data)","Single lifecycle aggregate",
                   "Plant → hub only","Placeholder (non-binding)",
                   "Approximate cost bands","Continuous (UI rounds)",
                   "Dummy data; soft penalty + alert","Out of scope"],
        "Production": ["~60 products/year across Radio, RAN Compute, Site Material",
                       "Multi-period (annual rolling)","Multi-echelon with vendor layer",
                       "Hard constraints from industrial planning",
                       "Precise values from industrial finance",
                       "Integer with per-route MOQ if needed",
                       "Real data from Snowflake (Amrith)","Future KPI / constraint"],
    }
    st.dataframe(pd.DataFrame(simplification_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown(
        "<h1 style='margin-bottom:0'>Supply Chain Design — POC v2</h1>"
        "<p style='color:#666;margin-top:4px'>Ericsson SCD Optimisation Engine  ·  "
        "Pilot: KRD901252/11 &amp; KRC4464B1B3B7</p>",
        unsafe_allow_html=True,
    )

    # Data load status banner
    data = get_data()
    if data is None:
        st.error(
            "No input data found. Run `python generate_data.py` to create dummy data, "
            "or upload CSV files in the Data tab."
        )
    else:
        s = data.summary_dict()
        st.caption(
            f"Data: **{s['plants']} plants** · **{len(s['products'])} products** · "
            f"**{s['active_pairs']} active demand pairs** · "
            f"**{s['total_demand']/1000:.0f}k units** lifecycle demand"
            + ("  |  Uploaded dataset" if st.session_state.data_source == "uploaded" else "")
        )

    tab_data, tab_config, tab_results, tab_about = st.tabs([
        "Data",
        "Configuration",
        "Results",
        "About",
    ])

    with tab_data:
        render_data_tab()

    with tab_config:
        render_config_tab()

    with tab_results:
        render_results_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
