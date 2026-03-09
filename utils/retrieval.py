"""Deterministic retrieval over the knowledge graph based on keyword intent."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from utils.graph import KnowledgeGraph, WellNode, HeadCellNode


def classify_intent(question: str) -> str:
    """Simple keyword-based intent classification."""
    q = question.lower()
    if any(k in q for k in ["ชั้นดิน", "lithology", "formation", "depth", "ความลึก",
                             "0–200", "0-200", "หิน", "ดิน", "rock", "interval"]):
        return "lithology"
    if any(k in q for k in ["ไหล", "downstream", "upstream", "flow", "ทิศทาง", "direction"]):
        return "flow"
    if any(k in q for k in ["head ต่ำ", "ต่ำสุด", "hotspot", "lowest", "สูงสุด",
                             "highest", "อันดับ", "top", "rank"]):
        return "hotspot"
    return "general"


def _extract_well_from_question(question: str, kg: KnowledgeGraph) -> Optional[str]:
    """Try to find a well ID mentioned in the question."""
    q_upper = question.upper()
    # Try longest match first to avoid partial hits
    for wid in sorted(kg.wells.keys(), key=len, reverse=True):
        if wid.upper() in q_upper:
            return wid
    return None


def retrieve(question: str, well_id: Optional[str], kg: KnowledgeGraph) -> Dict[str, Any]:
    """Run deterministic retrieval and return evidence dict."""
    intent = classify_intent(question)

    # If no well selected, try to detect one from the question text
    if not well_id:
        well_id = _extract_well_from_question(question, kg)

    evidence: Dict[str, Any] = {
        "intent": intent,
        "question": question,
        "selected_well_id": well_id,
    }

    # ---- selected well context ----
    well_node = kg.get_well(well_id) if well_id else None
    if well_node:
        evidence["SelectedWell"] = {
            "well_id": well_node.well_id,
            "lat": well_node.lat,
            "lon": well_node.lon,
            "total_depth": well_node.total_depth,
            "ground_elev": well_node.ground_elev,
        }
        cell = kg.get_cell_for_well(well_id)
        if cell:
            evidence["HeadCell"] = {
                "row": cell.row, "col": cell.col, "head": cell.head,
                "centroid_lat": cell.centroid_lat, "centroid_lon": cell.centroid_lon,
            }
        nearby = kg.get_nearby(well_id)
        if nearby:
            evidence["NearbyWells"] = [{"well_id": w, "distance_km": d} for w, d in nearby[:10]]

    # ---- intent-specific ----
    if intent == "lithology":
        evidence.update(_retrieve_lithology(well_id, kg, question))
    elif intent == "flow":
        evidence.update(_retrieve_flow(well_id, kg))
    elif intent == "hotspot":
        evidence.update(_retrieve_hotspot(well_id, kg))
    else:
        # general: include a bit of everything
        if well_id:
            evidence.update(_retrieve_lithology(well_id, kg, question))
            evidence.update(_retrieve_flow(well_id, kg))
        else:
            # No well context — provide global overview
            evidence["AllWells"] = [
                {"well_id": w.well_id, "total_depth": w.total_depth,
                 "ground_elev": w.ground_elev, "lat": w.lat, "lon": w.lon}
                for w in kg.wells.values()
            ]
            evidence["HeadSummary"] = {
                "total_cells": len(kg.head_cells),
                "min_head": round(min((c.head for c in kg.head_cells.values()), default=0), 3),
                "max_head": round(max((c.head for c in kg.head_cells.values()), default=0), 3),
            }
            evidence.update(_retrieve_hotspot(None, kg))

    evidence["DataReferences"] = _data_refs(intent)
    return evidence


def _retrieve_lithology(well_id: Optional[str], kg: KnowledgeGraph, question: str) -> Dict:
    if not well_id:
        return {"LithologySummary": {"note": "No well selected"}}
    intervals = kg.get_intervals(well_id)
    if not intervals:
        return {"LithologySummary": {"note": f"No interval data for {well_id}"}}

    # Parse optional depth range from question
    depth_min, depth_max = 0.0, 9999.0
    m = re.search(r"(\d+)\s*[–\-]\s*(\d+)", question)
    if m:
        depth_min, depth_max = float(m.group(1)), float(m.group(2))

    filtered = [iv for iv in intervals if iv.to_m >= depth_min and iv.from_m <= depth_max]
    # Summarize
    lith_thickness: Dict[str, float] = {}
    depth_ranges: Dict[str, List[str]] = {}
    for iv in filtered:
        thickness = iv.to_m - iv.from_m
        lith_thickness[iv.lithology] = lith_thickness.get(iv.lithology, 0) + thickness
        depth_ranges.setdefault(iv.lithology, []).append(f"{iv.from_m}-{iv.to_m}m")

    sorted_liths = sorted(lith_thickness.items(), key=lambda x: -x[1])
    return {
        "LithologySummary": {
            "well_id": well_id,
            "depth_range_filter": f"{depth_min}-{depth_max}m",
            "total_intervals": len(filtered),
            "lithologies": [
                {"lithology": l, "thickness_m": round(t, 1), "depth_ranges": depth_ranges.get(l, [])}
                for l, t in sorted_liths
            ],
        }
    }


def _retrieve_flow(well_id: Optional[str], kg: KnowledgeGraph) -> Dict:
    if not well_id:
        return {"DownstreamPath": {"note": "No well selected"}}
    cell = kg.get_cell_for_well(well_id)
    if not cell:
        return {"DownstreamPath": {"note": f"Well {well_id} not mapped to any head cell"}}

    downstream = kg.traverse_downstream(cell.key, max_hops=3)
    upstream = kg.traverse_upstream(cell.key, max_hops=3)
    ds_wells = kg.wells_in_cells(downstream)
    us_wells = kg.wells_in_cells(upstream)

    return {
        "DownstreamPath": {
            "source_cell": {"row": cell.row, "col": cell.col, "head": cell.head},
            "downstream_cells": [
                {"row": c.row, "col": c.col, "head": c.head} for c in downstream
            ],
            "downstream_wells": ds_wells,
            "upstream_cells": [
                {"row": c.row, "col": c.col, "head": c.head} for c in upstream
            ],
            "upstream_wells": us_wells,
        }
    }


def _retrieve_hotspot(well_id: Optional[str], kg: KnowledgeGraph) -> Dict:
    lowest_global = kg.lowest_head_cells(10)
    result: Dict[str, Any] = {
        "HotspotGlobal": [
            {"row": c.row, "col": c.col, "head": c.head,
             "wells_inside": kg.cell_wells.get(c.key, [])}
            for c in lowest_global
        ],
    }
    if well_id:
        w = kg.get_well(well_id)
        if w:
            lowest_near = kg.lowest_head_near(w.lat, w.lon, radius_km=5.0, n=5)
            result["HotspotNear"] = [
                {"row": c.row, "col": c.col, "head": c.head,
                 "wells_inside": kg.cell_wells.get(c.key, [])}
                for c in lowest_near
            ]
    return result


def _data_refs(intent: str) -> Dict:
    base = ["wells_wgs84.geojson", "head_grid_wgs84.geojson"]
    if intent == "lithology":
        base.append("intervals.csv")
    if intent == "flow":
        base.append("flow_dir_arrows_wgs84.geojson")
    return {"files": base, "method": "GraphRAG in-memory traversal"}
