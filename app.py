"""
app.py — SCD POC v2  Streamlit Application
============================================
Supply Chain Design Optimisation — Proof of Concept
Generic implementation, no company-specific terminology.

Run with:  streamlit run app.py

Tabs
----
1. Data          — download templates, upload, preview
2. Configuration — scenario setup, parameters, Run button
3. Results       — KPI table, charts, network matrix, Sankey, alerts, downloads
4. About         — model documentation
"""

from __future__ import annotations
import copy, io, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scd_engine import (
    SCDData, SCHEMAS, GEOS, CATS, LIFECYCLE_YEARS, ALPHA_DEFAULT,
    run_b1, run_b2, run_om, run_gm, compute_kpis, validate_dataframe,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SCD Optimisation POC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"

# ── Colour palette ────────────────────────────────────────────────────────────
SCENARIO_COLORS = {
    "B1-Nearest":   "#546E7A",
    "B2-Baseline":  "#78909C",
    "OM-Optimised": "#1976D2",
    "GM-GeoMix":    "#7B1FA2",
}
GEO_COLORS = {
    "Region_A": "#EF5350",
    "Region_B": "#42A5F5",
    "Region_C": "#66BB6A",
    "Region_D": "#FFA726",
}
CAT_COLORS = {"OWN": "#0288D1", "EXT": "#F57C00"}
PRODUCT_PALETTE = ["#1565C0", "#2E7D32", "#E65100", "#880E4F", "#4A148C"]


def _product_color(products_list: list, prod: str) -> str:
    idx = products_list.index(prod) if prod in products_list else 0
    return PRODUCT_PALETTE[idx % len(PRODUCT_PALETTE)]


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "data":          None,
        "data_source":   "default",
        "results":       {},
        "alpha_results": None,
        "demand_adj":    {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading default data...")
def _load_default() -> SCDData | None:
    try:
        return SCDData(DATA_DIR)
    except Exception:
        return None


def get_data() -> SCDData | None:
    if st.session_state.data is not None:
        return st.session_state.data
    d = _load_default()
    if d:
        st.session_state.data = d
    return st.session_state.data


# ── Download helpers ──────────────────────────────────────────────────────────
def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _results_zip(results: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, k in results.items():
            safe = label.replace(" ", "_")
            zf.writestr(f"alloc_{safe}.csv", k["alloc_df"].to_csv(index=False))
            zf.writestr(f"plant_summary_{safe}.csv", k["plant_df"].to_csv(index=False))
            frames = []
            if not k["new_routes_df"].empty:
                nr = k["new_routes_df"].copy()
                nr["AlertType"] = "new_network_flow"
                nr["Detail"] = nr.apply(
                    lambda r: f"{r.Qty:.0f} units, LCmult={r.LCmult}", axis=1)
                frames.append(nr[["AlertType","ProductID","PlantID","Hub","Detail"]])
            if not k["consol_df"].empty:
                cs = k["consol_df"].copy()
                cs["AlertType"] = "consolidation_candidate"
                cs["PlantID"] = "ALL"; cs["Hub"] = "ALL"
                cs["Detail"] = cs.apply(
                    lambda r: f"Demand {r.TotalDemand} < MinProd {r.MinProd_threshold}", axis=1)
                frames.append(cs[["AlertType","ProductID","PlantID","Hub","Detail"]])
            if frames:
                zf.writestr(f"alerts_{safe}.csv",
                            pd.concat(frames).to_csv(index=False))
        # KPI comparison
        rows = []
        for label, k in results.items():
            r = {"Scenario": label,
                 "LandedCost_MSEK":  round(k["total_lc_sek"]/1e6, 2),
                 "FixedCost_MSEK":   round(k["fixed_cost_sek"]/1e6, 2),
                 "TotalCost_MSEK":   round(k["total_cost_sek"]/1e6, 2),
                 "CostPerUnit":      round(k["cost_per_unit_sek"]),
                 "WeightedLC_Pct":   round(k["weighted_lc_pct"], 2),
                 "N_OpenPlants":     k["n_open_plants"]}
            for g in GEOS:
                r[f"Pct_{g}"] = round(k["geo_pct"].get(g, 0), 1)
            rows.append(r)
        zf.writestr("kpi_comparison.csv", pd.DataFrame(rows).to_csv(index=False))
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _chart_cost(results: dict) -> go.Figure:
    labels = list(results.keys())
    lc     = [results[l]["total_lc_sek"]/1e6 for l in labels]
    fc     = [results[l]["fixed_cost_sek"]/1e6 for l in labels]
    total  = [a + b for a, b in zip(lc, fc)]
    colors = [SCENARIO_COLORS.get(l, "#607D8B") for l in labels]

    y_min = max(0, min(total) * 0.92)
    y_max = max(total) * 1.07

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Landed Cost", x=labels, y=lc, marker_color=colors,
        text=[f"{v:.1f}" for v in lc],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=12),
    ))
    fig.add_trace(go.Bar(
        name="Fixed Plant Cost", x=labels, y=fc, marker_color="#90A4AE",
        text=[f"+{v:.1f}" for v in fc],
        textposition="outside", textfont=dict(size=10),
    ))
    for x_val, t_val in zip(labels, total):
        fig.add_annotation(
            x=x_val, y=t_val, text=f"<b>{t_val:.1f}</b>",
            showarrow=False, yshift=22, font=dict(size=12),
        )
    fig.update_layout(
        barmode="stack",
        title="Total Lifecycle Cost (MSEK) — zoomed to show differences",
        yaxis_title="MSEK", yaxis_range=[y_min, y_max],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=460, margin=dict(t=90, b=30, l=70, r=20),
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
            marker_color=GEO_COLORS.get(geo, "#888"),
            text=[f"{v:.0f}%" for v in vals],
            textposition="inside", insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack", title="Volume by Region (%)",
        yaxis_title="% of Lifecycle Volume", yaxis_range=[0, 115],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=460, margin=dict(t=90, b=30, l=60, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


def _chart_network_matrix(kpis: dict, data: SCDData) -> go.Figure:
    """
    Plant × Hub allocation matrix. Shows which plants supply which hubs.
    Cell value = total volume across all products. Empty = no flow.
    Modelled on the Network Summary view (Image 2 from user feedback).
    """
    df = kpis["alloc_df"]
    plants = data.plants_list
    hubs   = sorted(df[df.Qty > 0].Hub.unique())

    # Build matrix: rows = plants, cols = hubs
    z       = []   # volume values
    text    = []   # cell text
    y_labels = []

    for p in plants:
        row_z, row_t = [], []
        is_open = p in kpis["open_plants"]
        y_labels.append(
            f"{p}  [{data.plant_geo.get(p)}|{data.plant_cat.get(p)}]"
            + (" ●" if is_open else " ○")
        )
        for h in hubs:
            sub = df[(df.PlantID == p) & (df.Hub == h)]
            qty = sub.Qty.sum()
            if qty > 0.5:
                # Break down by product in cell text
                parts = [
                    f"{r.ProductID}: {r.Qty:,.0f}"
                    for _, r in sub.iterrows()
                ]
                row_z.append(qty)
                row_t.append("<br>".join(parts))
            else:
                row_z.append(0)
                row_t.append("—")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=hubs, y=y_labels,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#F5F5F5"], [0.01, "#BBDEFB"], [1, "#0D47A1"]],
        showscale=True,
        colorbar=dict(title="Total Volume", thickness=12),
        hovertemplate="Plant: %{y}<br>Hub: %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Network Summary — Plant × Hub Allocation  (● open plant  ○ closed)",
        xaxis_title="Hub",
        yaxis_title="",
        height=max(350, len(plants) * 52),
        margin=dict(t=60, b=60, l=200, r=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _chart_sankey(om_kpis: dict, bl_kpis: dict, data: SCDData) -> go.Figure:
    """
    Diff Sankey: OM vs a selected baseline.
    Blue = continuing, Green = new in OM, Red = dropped by OM.
    """
    def _agg(kpis):
        df = kpis["alloc_df"]
        return (df.groupby(["PlantID","Hub"])["Qty"].sum()
                  .reset_index()
                  .set_index(["PlantID","Hub"])["Qty"].to_dict())

    om_flows = _agg(om_kpis)
    bl_flows = _agg(bl_kpis)
    all_flows = set(om_flows) | set(bl_flows)

    active_plants = sorted({p for p, h in all_flows})
    active_hubs   = sorted({h for p, h in all_flows})
    n_p = len(active_plants)
    n_h = len(active_hubs)

    plant_idx = {p: i for i, p in enumerate(active_plants)}
    hub_idx   = {h: n_p + i for i, h in enumerate(active_hubs)}

    plant_y = [round(i / max(n_p - 1, 1), 4) for i in range(n_p)]
    hub_y   = [round(i / max(n_h - 1, 1), 4) for i in range(n_h)]

    node_labels = (
        [f"{p}  {data.plant_geo.get(p,'?')} | {data.plant_cat.get(p,'?')}"
         for p in active_plants]
        + [f"{h}  {data.hub_region.get(h,'')}" for h in active_hubs]
    )
    node_colors = ["#1565C0"] * n_p + ["#2E7D32"] * n_h

    sources, targets, values, link_colors, custom = [], [], [], [], []
    CONT  = "rgba(100,149,237,0.55)"
    NEW   = "rgba(46,125,50,0.65)"
    DROP  = "rgba(183,28,28,0.45)"

    for plant, hub in sorted(all_flows):
        in_om = (plant, hub) in om_flows
        in_bl = (plant, hub) in bl_flows
        qty   = om_flows.get((plant, hub), bl_flows.get((plant, hub), 0))
        if in_om and in_bl:
            color, tag = CONT, "Continuing"
        elif in_om:
            color, tag = NEW, "NEW in Optimised"
        else:
            qty   = bl_flows.get((plant, hub), 0) * 0.25
            color, tag = DROP, "Dropped by Optimised"
        sources.append(plant_idx[plant])
        targets.append(hub_idx[hub])
        values.append(max(qty, 200))
        link_colors.append(color)
        custom.append(
            f"{plant} → {hub}<br>"
            f"Optimised volume: {om_flows.get((plant,hub),0):,.0f} units<br>"
            f"Baseline volume: {bl_flows.get((plant,hub),0):,.0f} units<br>"
            f"Status: {tag}"
        )

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=25, thickness=20,
            label=node_labels, color=node_colors,
            x=[0.08]*n_p + [0.88]*n_h,
            y=plant_y + hub_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
            customdata=custom,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=f"Network Flow — Optimised vs {bl_kpis['label']}",
        height=620, margin=dict(l=140, r=160, t=60, b=20),
        font=dict(size=12),
    )
    return fig


def _chart_alpha_sensitivity(alpha_results: list) -> go.Figure:
    alphas  = [r["alpha"] for r in alpha_results]
    costs   = [r["total_lc_sek"]/1e6 for r in alpha_results]
    plants  = [r["n_open_plants"] for r in alpha_results]
    new_rts = [len(r["new_routes_df"]) for r in alpha_results]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Landed Cost (MSEK) — how cost changes with α",
            "Network structure — open plants and new-route alerts",
        ],
        vertical_spacing=0.18,
    )
    fig.add_trace(go.Scatter(
        x=alphas, y=costs, name="Landed Cost (MSEK)",
        mode="lines+markers", line=dict(color="#1976D2", width=2.5),
        marker=dict(size=8),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=alphas, y=plants, name="# Open Plants",
        mode="lines+markers", line=dict(color="#E65100", width=2.5, dash="dash"),
        marker=dict(size=8),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=alphas, y=new_rts, name="New-Route Alerts",
        mode="lines+markers", line=dict(color="#2E7D32", width=2.5, dash="dot"),
        marker=dict(size=8),
    ), row=2, col=1)

    cost_pad = max((max(costs) - min(costs)) * 0.5, 0.5)
    fig.update_yaxes(title_text="MSEK",
                     range=[min(costs)-cost_pad, max(costs)+cost_pad],
                     gridcolor="#F0F0F0", row=1, col=1)
    fig.update_yaxes(title_text="Count",
                     range=[0, max(max(plants), max(new_rts)) + 1],
                     dtick=1, gridcolor="#F0F0F0", row=2, col=1)
    fig.update_xaxes(title_text="α value", tickformat=".3f", gridcolor="#F0F0F0")
    fig.update_layout(
        title="Alpha (α) Sensitivity — higher α = stronger preference for historical routes",
        height=520, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=90, b=40, l=70, r=20),
    )
    return fig


def _chart_demand_check(results: dict, data: SCDData) -> go.Figure:
    products  = data.products_list
    scenarios = list(results.keys())
    n_rows    = len(products)
    scen_colors = [SCENARIO_COLORS.get(s, "#607D8B") for s in scenarios]

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=[f"Demand Satisfaction — {p}" for p in products],
        vertical_spacing=0.14,
    )
    for pi, prod in enumerate(products):
        active_hubs = [h for (i, h) in data.active_ig if i == prod]
        row = pi + 1
        d_vals = [data.demand_map.get((prod, h), 0) / 1000 for h in active_hubs]
        fig.add_trace(go.Bar(
            name="Demand", x=active_hubs, y=d_vals,
            marker_color="#CFD8DC",
            marker_line=dict(color="#607D8B", width=1.5),
            showlegend=(pi == 0),
        ), row=row, col=1)
        for si, (label, kpis) in enumerate(results.items()):
            a_vals = [
                kpis["alloc_df"][
                    (kpis["alloc_df"].ProductID == prod) &
                    (kpis["alloc_df"].Hub == h)
                ].Qty.sum() / 1000
                for h in active_hubs
            ]
            fig.add_trace(go.Bar(
                name=label, x=active_hubs, y=a_vals,
                marker_color=scen_colors[si], opacity=0.75,
                showlegend=(pi == 0),
            ), row=row, col=1)
        fig.update_yaxes(title_text="k units", gridcolor="#F0F0F0", row=row, col=1)
        fig.update_xaxes(gridcolor="#F0F0F0", tickfont=dict(size=12), row=row, col=1)

    fig.update_layout(
        title="Demand Satisfaction — grey = required demand, coloured = allocated by scenario",
        barmode="group", height=320 * n_rows,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=100, b=40, l=70, r=20),
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
        "the rest use the built-in defaults automatically."
    )
    data = get_data()

    src_msg = ("**Uploaded data active**" if st.session_state.data_source == "uploaded"
               else "**Default dummy data active**")
    st.info(src_msg + "  ·  Upload any file below to replace it.")

    if data:
        s = data.summary_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plants",              s["plants"])
        c2.metric("Products",            len(s["products"]))
        c3.metric("Active demand pairs", s["active_pairs"])
        c4.metric("Total demand (k)",    f"{s['total_demand']/1000:.0f}k")

    st.divider()

    # Download templates
    st.subheader("Download Templates")
    st.caption("Each template contains only the required columns. "
               "Expand the column guide under each file before editing.")
    file_cols = st.columns(4)
    for idx, (fname, schema) in enumerate(SCHEMAS.items()):
        col = file_cols[idx % 4]
        try:
            df_t = pd.read_csv(DATA_DIR / fname)
            csv_b = _to_csv(df_t)
        except Exception:
            csv_b = b""
        col.download_button(
            label=fname, data=csv_b, file_name=fname,
            mime="text/csv", use_container_width=True,
            key=f"dl_{fname}",
        )
        with col.expander("Column guide", expanded=False):
            st.caption(schema["doc"])
            st.markdown("**Required columns:** " +
                        ", ".join(f"`{c}`" for c in schema["required"]))
            if schema.get("choices"):
                for cn, vals in schema["choices"].items():
                    st.markdown(f"**`{cn}`** must be one of: `{vals}`")
            if schema.get("non_neg"):
                st.markdown(f"**Must be ≥ 0:** {schema['non_neg']}")

    st.divider()

    # Upload
    st.subheader("Upload Modified Files")
    st.caption("Each file is validated before being accepted. "
               "Unmodified files keep using the built-in defaults.")
    uploaded = st.file_uploader(
        "Drag and drop CSV files here (or click to browse)",
        type="csv", accept_multiple_files=True, key="csv_uploader",
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
                with st.spinner("Building data model..."):
                    try:
                        nd = SCDData.from_dataframes(new_dfs)
                        st.session_state.data        = nd
                        st.session_state.data_source = "uploaded"
                        st.session_state.results     = {}
                        st.session_state.alpha_results = None
                        st.success("Data updated. Re-run scenarios in the Configuration tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
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

    # Preview
    if data:
        st.subheader("Preview Current Data")
        sel = st.selectbox("Select file to preview", list(SCHEMAS.keys()),
                           key="preview_select")
        file_map = {
            "plants.csv":          data.plants,
            "products.csv":        data.products,
            "demand.csv":          data.demand,
            "routes.csv":          data.routes,
            "route_allowed.csv":   data.allowed,
            "hubs.csv":            data.hubs,
            "hist_flow.csv":       data.hist,
            "coverage_params.csv": data.coverage,
        }
        preview_df = file_map.get(sel, pd.DataFrame())
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

    st.caption("Select which scenarios to run, configure parameters, "
               "then click Run Scenarios at the bottom.")

    # What-if demand adjustment
    with st.expander("What-If Demand Adjustment (optional)", expanded=False):
        st.caption(
            "Adjust lifecycle demand per hub by a percentage before running. "
            "Applies to ALL scenarios and ALL products for that hub. "
            "Use to stress-test: 'what if demand in Region B grows 20%?'"
        )
        adj_cols = st.columns(len(data.hubs_list))
        adj: dict[str, float] = {}
        for idx, hub in enumerate(data.hubs_list):
            base = sum(v for (i, h), v in data.demand_map.items() if h == hub)
            hub_label = f"{hub}\n{data.hub_region.get(hub, '')}"
            if base == 0:
                adj_cols[idx].caption(f"{hub}\n*(no demand)*")
                adj[hub] = 0.0
            else:
                pct = adj_cols[idx].slider(
                    hub_label, min_value=-50, max_value=50, value=0, step=5,
                    format="%d%%", key=f"adj_{hub}",
                    help=f"Base demand: {base:,.0f} units",
                )
                adj[hub] = float(pct)
        st.session_state.demand_adj = adj
        active_adj = {h: v for h, v in adj.items() if v != 0}
        if active_adj:
            st.info("Active adjustments: " +
                    ", ".join(f"{h}: {v:+.0f}%" for h, v in active_adj.items()))

    st.divider()

    # Scenario checkboxes
    st.subheader("Select Scenarios to Run")
    c1, c2, c3, c4 = st.columns(4)
    run_b1_flag = c1.checkbox("B1 — Nearest-Plant",     value=True, key="cb_b1",
        help="Assigns each hub to the cheapest allowed plant. Deterministic, no optimisation.")
    run_b2_flag = c2.checkbox("B2 — Baseline Plant",    value=True, key="cb_b2",
        help="All demand through one reference plant. Used as the cost-avoidance benchmark.")
    run_om_flag = c3.checkbox("OM — Optimised (MILP)",  value=True, key="cb_om",
        help="Minimises total landed cost subject to all constraints. Solver: CBC.")
    run_gm_flag = c4.checkbox("GM — Regional Mix",      value=True, key="cb_gm",
        help="You specify the % of volume from each region. Cheapest plant in each region is used.")

    if not any([run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag]):
        st.warning("Select at least one scenario.")
        return

    st.divider()
    cfg: dict = {}

    # B2 configuration
    if run_b2_flag:
        st.subheader("B2 — Baseline Plant")
        st.caption(
            "All lifecycle demand (all products, all hubs) is assigned to this single plant. "
            "Used as the cost-avoidance reference — compare OM savings against this scenario."
        )
        default_idx = next(
            (i for i, p in enumerate(data.plants_list)
             if data.plant_cat.get(p) == "OWN"), 0)
        b2_plant = st.selectbox(
            "Baseline plant for all products:",
            options=data.plants_list,
            index=default_idx,
            format_func=lambda p: (
                f"{p}  ({data.plant_name[p]} — "
                f"{data.plant_geo[p]}, {data.plant_cat[p]})"
            ),
            key="b2_plant_single",
        )
        cfg["b2_baseline_plant"] = {prod: b2_plant for prod in data.products_list}
        st.divider()

    # OM configuration
    if run_om_flag:
        st.subheader("OM — Optimised MILP Parameters")

        st.markdown("##### Historical Route Penalty (α)")
        st.caption(
            "Controls how strongly the model prefers **existing (historical) routes** "
            "over new ones. The penalty added per unit on a non-historical route = "
            "α × mean(UnitCost). At α=0: pure cost optimisation — freely proposes new routes. "
            "At α=0.05: only switches to a new route if saving exceeds ~5% of unit cost."
        )
        alpha_val = st.slider(
            "α value", min_value=0.0, max_value=0.05,
            value=ALPHA_DEFAULT, step=0.005, format="%.3f", key="om_alpha",
        )
        mean_uc = (sum(data.unit_cost.values()) / len(data.unit_cost)
                   if data.unit_cost else 17000)
        penalty_sek = alpha_val * mean_uc
        col_l, col_r = st.columns(2)
        col_l.info(
            f"Equivalent to ≈ **{alpha_val*100:.1f}% cost difference** per unit "
            f"on non-historical routes."
        )
        col_r.metric("Penalty per non-historical unit",
                     f"{penalty_sek:,.0f} (cost units)")
        cfg["om_alpha"] = alpha_val

        st.markdown("##### Regional & Category Coverage (optional)")
        st.caption(
            "Minimum number of open plants per region and category. "
            "Strategic policy constraints — not hard regulatory rules. "
            "Set any value to 0 to disable that constraint."
        )
        use_cov = st.checkbox("Enable coverage constraints", value=True, key="om_use_cov")
        cfg["om_use_coverage"] = use_cov

        if use_cov:
            cov_cols = st.columns(len(GEOS) + len(CATS))
            geo_par, cat_par = {}, {}
            for idx, geo in enumerate(GEOS):
                geo_par[geo] = cov_cols[idx].number_input(
                    f"Min {geo}", min_value=0, max_value=5,
                    value=data.geo_params.get(geo, 0), key=f"geo_{geo}",
                )
            for idx, cat in enumerate(CATS):
                cat_par[cat] = cov_cols[len(GEOS) + idx].number_input(
                    f"Min {cat}", min_value=0, max_value=5,
                    value=data.cat_params.get(cat, 0), key=f"cat_{cat}",
                )
            cfg["om_geo_params"] = geo_par
            cfg["om_cat_params"] = cat_par
        st.divider()

    # GM configuration
    if run_gm_flag:
        st.subheader("GM — Regional Mix Configuration")
        st.caption(
            "Specify the target % of total volume from each region. "
            "Within each region, the cheapest allowed plant per hub is used. "
            "Percentages must sum to 100."
        )
        gm_cols = st.columns(len(GEOS))
        gm_vals: dict[str, float] = {}
        defaults = {GEOS[0]: 30, GEOS[1]: 70, GEOS[2]: 0, GEOS[3]: 0}
        for idx, geo in enumerate(GEOS):
            gm_vals[geo] = float(gm_cols[idx].number_input(
                f"{geo} %", min_value=0.0, max_value=100.0,
                value=float(defaults.get(geo, 0)), step=5.0, key=f"gm_{geo}",
            ))
        total_pct = sum(gm_vals.values())
        if abs(total_pct - 100) > 0.01:
            st.error(f"Percentages sum to {total_pct:.1f}% — must equal 100%.")
            run_gm_flag = False
        else:
            st.success(f"Total = {total_pct:.0f}% — valid")
        cfg["gm_geo_mix"] = gm_vals
        st.divider()

    # Run button
    st.subheader("Run")
    if st.button("Run Selected Scenarios", type="primary", use_container_width=True):
        _run_scenarios(data, cfg, run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag)


def _run_scenarios(data, cfg, run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag):
    adj = st.session_state.demand_adj
    if any(v != 0 for v in adj.values()):
        data = data.with_demand_adjustments(adj)
        st.info("Demand adjustments applied.")

    results = {}
    total = sum([run_b1_flag, run_b2_flag, run_om_flag, run_gm_flag])
    done  = 0
    progress = st.progress(0, text="Preparing...")

    def _step(lbl):
        nonlocal done
        done += 1
        progress.progress(done / total, text=f"Running {lbl}...")

    try:
        if run_b1_flag:
            _step("B1 Nearest-Plant")
            alloc, alerts = run_b1(data, {})
            results["B1-Nearest"] = compute_kpis(alloc, data, "B1-Nearest", alerts=alerts)

        if run_b2_flag:
            _step("B2 Baseline Plant")
            alloc, alerts = run_b2(data, {"baseline_plant": cfg.get("b2_baseline_plant", {})})
            results["B2-Baseline"] = compute_kpis(alloc, data, "B2-Baseline", alerts=alerts)

        if run_om_flag:
            _step("OM Optimised (MILP)")
            om_cfg = {
                "alpha":        cfg.get("om_alpha",        ALPHA_DEFAULT),
                "use_coverage": cfg.get("om_use_coverage", True),
                "geo_params":   cfg.get("om_geo_params",   data.geo_params),
                "cat_params":   cfg.get("om_cat_params",   data.cat_params),
            }
            with st.spinner("Solving MILP... (CBC solver, up to 5 min timeout)"):
                alloc, meta = run_om(data, om_cfg)
            results["OM-Optimised"] = compute_kpis(alloc, data, "OM-Optimised", meta=meta)

        if run_gm_flag:
            _step("GM Regional Mix")
            gm_cfg = {"geo_mix": cfg.get("gm_geo_mix", {g: 0 for g in GEOS})}
            alloc, alerts = run_gm(data, gm_cfg)
            results["GM-GeoMix"] = compute_kpis(alloc, data, "GM-GeoMix", alerts=alerts)

        progress.progress(1.0, text="Done")
        st.session_state.results = results
        st.success(f"{len(results)} scenario(s) complete. "
                   "Switch to the Results tab to explore.")

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
        st.info("No results yet. Configure and run scenarios in the Configuration tab.")
        return
    if data is None:
        st.error("Data not loaded.")
        return

    # Scenario legend
    st.caption(
        "**B1-Nearest** = cheapest allowed plant per hub (deterministic)  |  "
        "**B2-Baseline** = all demand through one reference plant (cost benchmark)  |  "
        "**OM-Optimised** = MILP solution minimising total cost  |  "
        "**GM-GeoMix** = user-specified regional volume split"
    )

    # KPI glossary
    with st.expander("KPI definitions", expanded=False):
        st.markdown("""
| KPI | Definition |
|-----|-----------|
| **Total Cost** | Landed cost + fixed plant overhead across all products and hubs for the lifecycle horizon. Primary cost figure. |
| **Landed Cost** | Volume × LC multiplier × unit cost per route, summed across all products. Excludes fixed plant overhead. |
| **Fixed Cost** | Annual plant overhead (line setup, qualification, staffing) × lifecycle years, per open plant. Incurred regardless of volume. |
| **Cost / Unit** | Total landed cost ÷ total lifecycle volume. Volume-weighted average cost per unit shipped. |
| **Weighted LC %** | Volume-weighted average of the LC multiplier across all active routes. Lower is better. LCmult = 1 + adder%, so 110% means 10% added on top of base unit cost. |
| **# Open Plants** | Distinct plants used in this network design. More plants = more operational complexity and fixed cost. |
| **Cost Avoidance vs B2** | Cost saving vs the B2 Baseline Plant. Positive = cheaper than B2. Negative = more expensive than B2. |
| **New-Route Alerts** | Plant-to-hub flows proposed with no historical precedent. Each requires review before implementation. |
""")

    st.divider()

    # ── Consolidated KPI comparison table (Image 1 from user feedback) ─────────
    st.subheader("Scenario Comparison")
    b2_cost = results.get("B2-Baseline", {}).get("total_cost_sek")

    table_rows = []
    for label, k in results.items():
        if b2_cost and label != "B2-Baseline":
            avoid = f"{(b2_cost - k['total_cost_sek'])/1e6:+.1f}"
        else:
            avoid = "— (reference)"
        table_rows.append({
            "Scenario":             label,
            "Landed Cost (MSEK)":   f"{k['total_lc_sek']/1e6:.1f}",
            "Fixed Cost (MSEK)":    f"{k['fixed_cost_sek']/1e6:.1f}",
            "Total Cost (MSEK)":    f"{k['total_cost_sek']/1e6:.1f}",
            "Cost / Unit":          f"{k['cost_per_unit_sek']:,.0f}",
            "Weighted LC %":        f"{k['weighted_lc_pct']:.2f}%",
            "# Open Plants":        k["n_open_plants"],
            "Cost Avoidance vs B2": avoid,
            "New-Route Alerts":     len(k["new_routes_df"]),
        })
    comparison_df = pd.DataFrame(table_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── KPI metric cards ────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(len(results))
    for col, (label, k) in zip(kpi_cols, results.items()):
        if b2_cost and label != "B2-Baseline":
            val = (b2_cost - k["total_cost_sek"]) / 1e6
            avoidance = f"{val:+.1f} MSEK"
        else:
            avoidance = "— (reference)"
        col.metric("Scenario",            label)
        col.metric("Total Cost (MSEK)",   f"{k['total_cost_sek']/1e6:.1f}",
                   help="Landed cost + fixed plant overhead.")
        col.metric("Landed Cost (MSEK)",  f"{k['total_lc_sek']/1e6:.1f}",
                   help="Volume × LCmult × UnitCost. Excludes fixed overhead.")
        col.metric("Fixed Cost (MSEK)",   f"{k['fixed_cost_sek']/1e6:.1f}",
                   help="Plant overhead × lifecycle years, per open plant.")
        col.metric("Cost / Unit",         f"{k['cost_per_unit_sek']:,.0f}",
                   help="Total landed cost ÷ total volume.")
        col.metric("Weighted LC %",       f"{k['weighted_lc_pct']:.2f}%",
                   help="Volume-weighted average LC multiplier. Lower is better.")
        col.metric("# Open Plants",       k["n_open_plants"],
                   help="More plants = more complexity and fixed cost.")
        col.metric("Cost Avoidance vs B2", avoidance,
                   help="Positive = cheaper than B2 (cost saving). Negative = more expensive.")
        col.metric("New-Route Alerts",    len(k["new_routes_df"]),
                   help="Flows with no historical precedent. Each requires review.")

    st.divider()

    # ── Cost & Geo Charts ───────────────────────────────────────────────────────
    st.subheader("Cost & Regional Breakdown")
    st.caption("Left: lifecycle cost split into landed cost and fixed overhead.  "
               "Right: share of total volume produced per region.")
    ch1, ch2 = st.columns(2)
    ch1.plotly_chart(_chart_cost(results),   use_container_width=True)
    ch2.plotly_chart(_chart_geo(results),    use_container_width=True)

    st.divider()

    # ── Network Matrix ──────────────────────────────────────────────────────────
    st.subheader("Network Summary Matrix")
    st.caption("Shows which plants supply which hubs. "
               "Cell = total volume allocated on that route. "
               "Hover for product-level breakdown.")
    matrix_scen = st.selectbox(
        "Show matrix for scenario:", list(results.keys()), key="matrix_scen")
    st.plotly_chart(
        _chart_network_matrix(results[matrix_scen], data),
        use_container_width=True,
    )

    st.divider()

    # ── Sankey ──────────────────────────────────────────────────────────────────
    st.subheader("Network Flow Diagram")
    st.caption("Shows how the Optimised (OM) network differs from a baseline. "
               "Link width = volume. Hover for details.")
    if "OM-Optimised" not in results:
        st.info("Run the Optimised (OM) scenario to see the network flow diagram.")
    else:
        baseline_options = [l for l in results if l != "OM-Optimised"]
        if not baseline_options:
            st.caption("Run at least one baseline (B1 or B2) to compare against.")
        else:
            compare_to = st.selectbox(
                "Compare Optimised (OM) against:",
                baseline_options, key="sankey_baseline",
            )
            st.caption(
                "🔵 Blue = route continues from baseline  |  "
                "🟢 Green = new route proposed by OM  |  "
                "🔴 Red = route dropped by OM (shown thin).  "
                "Hover over any link for exact volumes."
            )
            st.plotly_chart(
                _chart_sankey(results["OM-Optimised"], results[compare_to], data),
                use_container_width=True,
            )

    st.divider()

    # ── Demand satisfaction ─────────────────────────────────────────────────────
    with st.expander("Demand Satisfaction Check", expanded=False):
        st.caption("Confirms that all lifecycle demand is fully served (constraint C1). "
                   "Grey bar = required demand. Coloured bars = allocated by scenario.")
        st.plotly_chart(_chart_demand_check(results, data), use_container_width=True)

    st.divider()

    # ── Alpha sensitivity ───────────────────────────────────────────────────────
    with st.expander("Alpha (α) Sensitivity Analysis", expanded=False):
        st.caption(
            "Re-runs the OM scenario at 5 different α values to show how the "
            "historical-route preference affects total cost, open plants, and new-route alerts."
        )
        if st.button("Run Alpha Sensitivity Analysis", key="run_alpha"):
            alpha_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
            alpha_kpis = []
            prog = st.progress(0, text="Running sensitivity...")
            base_cfg = {"alpha": 0.01, "use_coverage": True,
                        "geo_params": data.geo_params, "cat_params": data.cat_params}
            for i, av in enumerate(alpha_vals):
                prog.progress((i+1)/len(alpha_vals), text=f"α = {av:.3f}...")
                try:
                    alloc, meta = run_om(data, {**base_cfg, "alpha": av})
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
            summary = pd.DataFrame([
                {"α": r["alpha"],
                 "Landed Cost (MSEK)":  round(r["total_lc_sek"]/1e6, 1),
                 "Fixed Cost (MSEK)":   round(r["fixed_cost_sek"]/1e6, 1),
                 "Total Cost (MSEK)":   round(r["total_cost_sek"]/1e6, 1),
                 "# Open Plants":       r["n_open_plants"],
                 "New-Route Alerts":    len(r["new_routes_df"])}
                for r in st.session_state.alpha_results
            ])
            st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # ── Alerts ──────────────────────────────────────────────────────────────────
    st.subheader("Alerts")
    for tab, (label, k) in zip(st.tabs(list(results.keys())), results.items()):
        with tab:
            if k.get("run_alerts"):
                with st.expander(f"Routing fallbacks ({len(k['run_alerts'])})",
                                 expanded=True):
                    for a in k["run_alerts"]:
                        st.warning(a)

            if not k["new_routes_df"].empty:
                st.markdown("**New Network Flows (no historical precedent)**")
                st.caption(
                    "These routes exist in the proposed network but were NOT used historically. "
                    "Review with the planning board before approving — "
                    "each is a change-management signal."
                )
                nr = k["new_routes_df"][["ProductID","PlantID","Hub","Qty","LCmult"]].copy()
                nr["Qty"] = nr["Qty"].round().astype(int)
                st.dataframe(nr, use_container_width=True, hide_index=True)
            else:
                st.success("No new-route alerts — all flows match historical patterns.")

            if not k["consol_df"].empty:
                st.markdown("**Consolidation Candidates**")
                st.caption(
                    "Products whose total lifecycle demand is below the MinProd_ifOpen "
                    "threshold. Consider consolidating production to a single plant."
                )
                st.dataframe(k["consol_df"], use_container_width=True, hide_index=True)

    st.divider()

    # ── Full allocation tables ──────────────────────────────────────────────────
    with st.expander("Full Allocation Tables", expanded=False):
        sel_scen = st.selectbox("Show allocation for:", list(results.keys()),
                                key="detail_scen")
        st.dataframe(results[sel_scen]["alloc_df"],
                     use_container_width=True, height=400)
        st.markdown("**Plant Summary**")
        st.dataframe(results[sel_scen]["plant_df"],
                     use_container_width=True, hide_index=True)

    st.divider()

    # ── Downloads ───────────────────────────────────────────────────────────────
    st.subheader("Download Results")
    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "Download All Results (ZIP)", data=_results_zip(results),
        file_name="scd_poc_v2_results.zip", mime="application/zip",
        use_container_width=True,
    )
    dl_scen = dc2.selectbox("Or download single scenario:",
                             list(results.keys()), key="dl_single")
    dc2.download_button(
        f"Download {dl_scen} — alloc.csv",
        data=_to_csv(results[dl_scen]["alloc_df"]),
        file_name=f"alloc_{dl_scen}.csv", mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about_tab():
    st.header("About This Model")

    st.subheader("Business Purpose")
    st.markdown(
        "This tool answers: **How does a change in supply chain setup impact "
        "landed cost at the product level?**  "
        "It simulates alternative plant-to-hub network designs and computes cost "
        "impact and risk indicators to support supply chain board decisions."
    )

    st.subheader("Optimisation Objective (OM Scenario)")
    st.latex(r"""
    \min \sum_{i,p,h} \text{Alloc}_{i,p,h} \cdot \text{LCmult}_{p,h} \cdot \text{UnitCost}_i
         + \sum_p \text{FixedCost}_p \cdot T \cdot \text{OpenGlobal}_p
         + \alpha \cdot \overline{UC} \cdot \sum_{\{i,p,h:\,\text{HistFlow}=0\}} \text{Alloc}_{i,p,h}
    """)
    st.caption(
        "T = lifecycle years. Term 1 = landed cost. Term 2 = fixed plant overhead. "
        "Term 3 = soft penalty for using routes with no historical precedent."
    )
    st.markdown(
        "**Landed cost** is a standard supply-chain term. It is the total per-unit "
        "cost of moving a product from the factory to the receiving hub. It bundles "
        "transportation, customs/duties, inventory cost-of-capital, and distribution "
        "surcharges into a single multiplier: `LCmult = 1 + AdderPct`."
    )

    st.subheader("Constraints")
    st.dataframe(pd.DataFrame({
        "ID":   ["C1","C2","C3","C4","C5","C6","C7","C8"],
        "Name": ["Demand Satisfaction","Route Feasibility","Plant Feasibility",
                 "Allocation Gating","Min Production","Hub Coverage",
                 "Regional/Category Coverage","Capacity"],
        "Type": ["Hard","Hard","Hard","Hard","Hard","Hard","Optional","Placeholder"],
        "Description": [
            "All lifecycle demand per product and hub must be fully served.",
            "No flow on forbidden trade corridors (RouteAllowed=0). Product-agnostic.",
            "A route can only be used if the plant is open for that product.",
            "Volume only flows on open routes (big-M link: Alloc ≤ M × RouteOpen).",
            "If a plant is open for a product, it must produce ≥ MinProd_ifOpen units.",
            "Every (product, hub) pair with demand > 0 must have ≥ 1 supplying route.",
            "Min open plants per region (Region_A/B/C/D) and category (OWN/EXT). Configurable.",
            "Manufacturing capacity per product-plant. Set to 999,999 (non-binding) in POC v2.",
        ],
    }), use_container_width=True, hide_index=True)

    st.subheader("Input File Reference")
    st.dataframe(pd.DataFrame({
        "File": list(SCHEMAS.keys()),
        "Content": [s["doc"][:120] + "..." for s in SCHEMAS.values()],
        "Status": ["Replace with real data"] * len(SCHEMAS),
    }), use_container_width=True, hide_index=True)

    st.subheader("Open Points (Before Real Data Connection)")
    st.markdown("""
1. **Demand source** — confirm which forecasting system to use for lifecycle volume per hub.
2. **MinProd_ifOpen values** — collect realistic minimum production volumes per plant from operations teams (placeholder: 500–2,000 units).
3. **Historical flow data** — identify the data warehouse table for `hist_flow.csv` and confirm ownership.
4. **Fixed plant costs** — collect approximate annual overhead per plant from finance (placeholder: cost bands by region/category).
""")

    st.subheader("Known Simplifications (POC v2 vs Production)")
    st.dataframe(pd.DataFrame({
        "Dimension":   ["Products","Time horizon","Echelon","Capacity",
                        "Fixed plant cost","Allocation type","Historical flows","Lead time"],
        "POC v2":      ["2 fictional products","Single lifecycle aggregate",
                        "Plant → hub only","Placeholder (non-binding)",
                        "Approximate bands","Continuous","Dummy data; soft penalty + alert",
                        "Out of scope"],
        "Production":  ["All products across all product families","Multi-period (annual rolling)",
                        "Multi-echelon with vendor and customer layers",
                        "Hard constraints from industrial planning",
                        "Precise values from finance",
                        "Integer with per-route MOQ if needed",
                        "Real data from data warehouse","Future KPI / constraint"],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(
        "<h1 style='margin-bottom:0'>Supply Chain Design Optimisation</h1>"
        "<p style='color:#666;margin-top:4px'>"
        "Multi-plant, multi-product network design POC  ·  "
        "Scenarios: B1 Nearest-Plant | B2 Baseline | OM Optimised | GM Regional Mix"
        "</p>",
        unsafe_allow_html=True,
    )

    data = get_data()
    if data is None:
        st.error("No input data found. Run `python generate_data.py` to create dummy data, "
                 "or upload CSV files in the Data tab.")
    else:
        s = data.summary_dict()
        st.caption(
            f"Data: **{s['plants']} plants** · **{len(s['products'])} products** · "
            f"**{s['hubs']} hubs** · **{s['active_pairs']} active demand pairs** · "
            f"**{s['total_demand']/1000:.0f}k units** lifecycle demand"
            + ("  |  Uploaded dataset" if st.session_state.data_source == "uploaded" else "")
        )

    t1, t2, t3, t4 = st.tabs(["Data", "Configuration", "Results", "About"])
    with t1: render_data_tab()
    with t2: render_config_tab()
    with t3: render_results_tab()
    with t4: render_about_tab()


if __name__ == "__main__":
    main()
