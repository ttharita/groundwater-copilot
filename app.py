"""
Groundwater Intelligence Copilot (GraphRAG + GeoAI)
Main Streamlit application.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import branca.colormap as cm

from utils.data_loader import (
    load_wells, load_head_grid, load_flow_arrows,
    load_intervals, load_surface_points, load_orientations,
)
from utils.graph import build_graph, KnowledgeGraph
from utils.geospatial import get_well_id_col, nearby_wells as compute_nearby
from utils.retrieval import retrieve
from utils.gemini_client import is_available as gemini_available, generate_answer
from utils.templates import fallback_answer, build_report_html

# ─── Page config ───
st.set_page_config(
    page_title="Groundwater Intelligence Copilot",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    div[data-testid="stExpander"] details summary p { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ───
def _init_state():
    defaults = {
        "selected_well": None,
        "chat_history": [],
        "last_evidence": {},
        "last_answer": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Data loading ───
@st.cache_resource(show_spinner="Initializing …")
def init_all():
    wells = load_wells()
    head_grid = load_head_grid()
    flow_arrows = load_flow_arrows()
    intervals = load_intervals()
    surface_pts = load_surface_points()
    orientations = load_orientations()
    kg = build_graph(wells, head_grid, flow_arrows, intervals, orientations)
    return wells, head_grid, flow_arrows, intervals, surface_pts, orientations, kg


wells, head_grid, flow_arrows, intervals, surface_pts, orientations, kg = init_all()

# Identify columns
wid_col = get_well_id_col(wells) if not wells.empty else "well_id"
well_ids = sorted(wells[wid_col].astype(str).unique().tolist()) if not wells.empty else []

# Gemini status
has_gemini = gemini_available()

# ─── Header ───
st.markdown("## 💧 Groundwater Intelligence Copilot")
st.caption("GraphRAG + GeoAI  |  Executive Decision Support")
if not has_gemini:
    st.warning(
        "OpenAI API key not found. Add it to `.streamlit/secrets.toml` or set "
        "env var `OPENAI_API_KEY`. LLM answers disabled – using template fallback.",
        icon="🔑",
    )

# ═══════════════════════════════════════════════════
# LAYOUT: Map (left ~70%) | Copilot (right ~30%)
# ═══════════════════════════════════════════════════
map_col, copilot_col = st.columns([7, 3], gap="medium")

# ─────────────────────────────────────────────────
# LEFT COLUMN – MAP
# ─────────────────────────────────────────────────
with map_col:
    st.markdown("### Map")

    # Layer toggles
    tog1, tog2, tog3 = st.columns(3)
    show_wells = tog1.checkbox("Wells", value=True)
    show_head = tog2.checkbox("Head Grid", value=True)
    show_flow = tog3.checkbox("Flow Arrows", value=True)

    # Compute map center
    if not wells.empty:
        center_lat = wells.geometry.y.mean()
        center_lon = wells.geometry.x.mean()
    elif not head_grid.empty:
        c = head_grid.geometry.centroid
        center_lat, center_lon = c.y.mean(), c.x.mean()
    else:
        center_lat, center_lon = 15.0, 100.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="CartoDB positron")

    # --- Head grid layer ---
    head_col_name = None
    if show_head and not head_grid.empty:
        for c in ("head", "HEAD", "Head", "value", "VALUE", "z", "Z"):
            if c in head_grid.columns:
                head_col_name = c
                break
        if head_col_name:
            vmin = float(head_grid[head_col_name].min())
            vmax = float(head_grid[head_col_name].max())
            colormap = cm.LinearColormap(
                colors=["#d73027", "#fee08b", "#1a9850"],
                vmin=vmin, vmax=vmax,
                caption=f"Hydraulic Head ({head_col_name})",
            )
            colormap.add_to(m)

            def head_style(feature):
                val = feature["properties"].get(head_col_name, vmin)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = vmin
                return {
                    "fillColor": colormap(val),
                    "color": "#666",
                    "weight": 0.3,
                    "fillOpacity": 0.55,
                }

            folium.GeoJson(
                head_grid.__geo_interface__,
                style_function=head_style,
                name="Head Grid",
                tooltip=folium.GeoJsonTooltip(
                    fields=[head_col_name],
                    aliases=["Head:"],
                ),
            ).add_to(m)

    # --- Flow arrows layer ---
    if show_flow and not flow_arrows.empty:
        for _, row in flow_arrows.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            coords = [(c[1], c[0]) for c in geom.coords]
            folium.PolyLine(
                coords,
                color="#2471a3",
                weight=2,
                opacity=0.7,
            ).add_to(m)
            # Small arrowhead at end
            if len(coords) >= 2:
                end = coords[-1]
                folium.RegularPolygonMarker(
                    location=end,
                    number_of_sides=3,
                    radius=5,
                    color="#2471a3",
                    fill=True,
                    fill_color="#2471a3",
                    fill_opacity=0.8,
                    rotation=0,
                ).add_to(m)

    # --- Wells layer ---
    if show_wells and not wells.empty:
        for _, row in wells.iterrows():
            wid = str(row[wid_col])
            pt = row.geometry
            is_selected = (wid == st.session_state.selected_well)
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=7 if is_selected else 5,
                color="#e74c3c" if is_selected else "#2c3e50",
                fill=True,
                fill_color="#e74c3c" if is_selected else "#3498db",
                fill_opacity=0.9,
                popup=folium.Popup(f"<b>{wid}</b>", max_width=200),
                tooltip=wid,
            ).add_to(m)

    map_data = st_folium(m, width=None, height=520, returned_objects=["last_object_clicked_tooltip"])

    # Handle map click
    if map_data and map_data.get("last_object_clicked_tooltip"):
        clicked = map_data["last_object_clicked_tooltip"]
        if clicked in well_ids and clicked != st.session_state.selected_well:
            st.session_state.selected_well = clicked
            st.rerun()

    # ─── Well Summary (below map) ───
    sel = st.session_state.selected_well
    if sel and sel in kg.wells:
        wnode = kg.wells[sel]
        st.markdown(f"#### Well: {sel}")
        info_cols = st.columns(4)
        info_cols[0].metric("Total Depth", f"{wnode.total_depth} m")
        info_cols[1].metric("Ground Elev", f"{wnode.ground_elev} m")
        cell = kg.get_cell_for_well(sel)
        if cell:
            info_cols[2].metric("Head", f"{cell.head}")
            info_cols[3].metric("Cell", f"({cell.row}, {cell.col})")

        # Lithology profile
        ivs = kg.get_intervals(sel)
        if ivs:
            with st.expander("Lithology Profile", expanded=False):
                iv_data = [{"From (m)": iv.from_m, "To (m)": iv.to_m,
                            "Thickness (m)": round(iv.to_m - iv.from_m, 1),
                            "Lithology": iv.lithology} for iv in ivs]
                st.dataframe(pd.DataFrame(iv_data), use_container_width=True, hide_index=True)

        # Nearby wells
        nw = kg.get_nearby(sel)
        if nw:
            with st.expander(f"Nearby Wells (≤ 2 km): {len(nw)}", expanded=False):
                nw_df = pd.DataFrame(nw, columns=["Well ID", "Distance (km)"])
                st.dataframe(nw_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────
# RIGHT COLUMN – COPILOT
# ─────────────────────────────────────────────────
with copilot_col:
    st.markdown("### Copilot")

    # ── Well selector ──
    sel_idx = well_ids.index(st.session_state.selected_well) if st.session_state.selected_well in well_ids else 0
    chosen = st.selectbox(
        "Select well (or click on map)",
        options=["(none)"] + well_ids,
        index=(sel_idx + 1) if st.session_state.selected_well else 0,
        key="well_selector",
    )
    if chosen != "(none)" and chosen != st.session_state.selected_well:
        st.session_state.selected_well = chosen
        st.rerun()
    elif chosen == "(none)" and st.session_state.selected_well is not None:
        st.session_state.selected_well = None
        st.rerun()

    # ── Top Insights ──
    with st.expander("Top Insights", expanded=True):
        ins_tabs = st.tabs(["Lowest Head", "Deepest Wells", "Downstream Exposure"])
        with ins_tabs[0]:
            low5 = kg.lowest_head_cells(5)
            for i, c in enumerate(low5, 1):
                ws = ", ".join(kg.cell_wells.get(c.key, [])) or "—"
                st.caption(f"{i}. Cell({c.row},{c.col}) head={c.head} → {ws}")
        with ins_tabs[1]:
            deep5 = kg.deepest_wells(5)
            for i, w in enumerate(deep5, 1):
                st.caption(f"{i}. {w.well_id}  depth={w.total_depth}m")
        with ins_tabs[2]:
            # Compute downstream exposure for all wells (top 3)
            exposures = []
            for wid in list(kg.wells.keys())[:50]:  # limit for perf
                exp = kg.downstream_exposure(wid, hops=3)
                exposures.append((wid, exp))
            exposures.sort(key=lambda x: -x[1])
            for i, (wid, exp) in enumerate(exposures[:3], 1):
                st.caption(f"{i}. {wid}  → {exp} reachable wells (3 hops)")

    st.divider()

    # ── Example question chips ──
    st.caption("Quick questions:")
    chip_cols = st.columns(1)
    example_qs = [
        "NB4 ชั้นดินเป็นอย่างไรตั้งแต่ 0–200m?",
        "น้ำจากบ่อ NB4 มีแนวโน้มไหลไปทางไหน?",
        "บ่อไหนอยู่ในบริเวณ head ต่ำสุด 10 อันดับ?",
        "มีบ่อน้ำทั้งหมดกี่บ่อ และบ่อไหนลึกที่สุด?",
        "ภาพรวม head น้ำบาดาลในพื้นที่เป็นอย่างไร?",
    ]
    for eq in example_qs:
        if st.button(eq, key=f"chip_{eq[:20]}", use_container_width=True):
            st.session_state["pending_question"] = eq
            st.rerun()

    st.divider()

    # ── Chat input ──
    user_q = st.chat_input("Ask about any well or groundwater in general …")
    if user_q:
        st.session_state["pending_question"] = user_q

    # ── Process pending question ──
    if st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
        well_ctx = st.session_state.selected_well

        # Retrieval
        evidence = retrieve(question, well_ctx, kg)
        st.session_state.last_evidence = evidence

        # LLM or fallback
        if has_gemini:
            answer = generate_answer(question, evidence)
        else:
            answer = fallback_answer(question, evidence)

        st.session_state.last_answer = answer
        st.session_state.chat_history.append({"q": question, "a": answer})

    # ── Display chat history (newest first) ──
    for msg in reversed(st.session_state.chat_history[-5:]):
        with st.chat_message("user"):
            st.write(msg["q"])
        with st.chat_message("assistant"):
            st.markdown(msg["a"])

    # ── Evidence Cards ──
    if st.session_state.last_evidence:
        with st.expander("Evidence Cards", expanded=False):
            ev = st.session_state.last_evidence
            for key in ("SelectedWell", "HeadCell", "NearbyWells",
                        "DownstreamPath", "LithologySummary",
                        "HotspotGlobal", "HotspotNear", "DataReferences"):
                if key in ev:
                    st.markdown(f"**{key}**")
                    val = ev[key]
                    if isinstance(val, (dict, list)):
                        st.json(val)
                    else:
                        st.write(val)

    st.divider()

    # ── Generate 1-page Report ──
    if st.button("Generate 1-page Report", use_container_width=True, type="primary"):
        well_summary = {}
        sel_w = st.session_state.selected_well
        if sel_w and sel_w in kg.wells:
            wn = kg.wells[sel_w]
            well_summary = {
                "Well ID": wn.well_id,
                "Latitude": wn.lat,
                "Longitude": wn.lon,
                "Total Depth (m)": wn.total_depth,
                "Ground Elevation (m)": wn.ground_elev,
            }
            cell = kg.get_cell_for_well(sel_w)
            if cell:
                well_summary["Head Cell"] = f"({cell.row}, {cell.col})"
                well_summary["Head Value"] = cell.head

        report_html = build_report_html(
            well_summary,
            st.session_state.last_answer,
            st.session_state.last_evidence,
        )
        with st.expander("Report Preview", expanded=True):
            st.components.v1.html(report_html, height=600, scrolling=True)

        st.download_button(
            "Download Report (HTML)",
            data=report_html,
            file_name="groundwater_report.html",
            mime="text/html",
            use_container_width=True,
        )
