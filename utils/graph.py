"""Build an in-memory knowledge graph (lightweight GraphRAG)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st

from utils.geospatial import (
    haversine_km,
    find_cell_for_point,
    nearest_cell_centroid,
    get_well_id_col,
)


# --------------- node types ---------------
@dataclass
class WellNode:
    well_id: str
    lat: float
    lon: float
    total_depth: float = 0.0
    ground_elev: float = 0.0
    cell_key: Optional[str] = None  # row_col


@dataclass
class IntervalNode:
    well_id: str
    from_m: float
    to_m: float
    lithology: str = ""
    formation: str = ""


@dataclass
class FormationNode:
    name: str


@dataclass
class HeadCellNode:
    row: int
    col: int
    head: float
    centroid_lat: float
    centroid_lon: float

    @property
    def key(self) -> str:
        return f"{self.row}_{self.col}"


# --------------- graph ---------------
class KnowledgeGraph:
    def __init__(self):
        self.wells: Dict[str, WellNode] = {}
        self.intervals: Dict[str, List[IntervalNode]] = {}  # well_id -> list
        self.formations: Dict[str, FormationNode] = {}
        self.head_cells: Dict[str, HeadCellNode] = {}  # "row_col"
        # edges
        self.well_cell: Dict[str, str] = {}  # well_id -> cell_key
        self.cell_wells: Dict[str, List[str]] = {}  # cell_key -> [well_ids]
        self.near_wells: Dict[str, List[Tuple[str, float]]] = {}  # well_id -> [(wid, km)]
        self.flows_to: Dict[str, List[str]] = {}  # cell_key -> [cell_key]
        self.flows_from: Dict[str, List[str]] = {}  # cell_key -> [cell_key] (reverse)

    # ------ queries ------
    def get_well(self, wid: str) -> Optional[WellNode]:
        return self.wells.get(wid)

    def get_intervals(self, wid: str) -> List[IntervalNode]:
        return self.intervals.get(wid, [])

    def get_cell_for_well(self, wid: str) -> Optional[HeadCellNode]:
        ck = self.well_cell.get(wid)
        return self.head_cells.get(ck) if ck else None

    def get_nearby(self, wid: str) -> List[Tuple[str, float]]:
        return self.near_wells.get(wid, [])

    def traverse_downstream(self, cell_key: str, max_hops: int = 3) -> List[HeadCellNode]:
        """BFS downstream from *cell_key*."""
        visited_keys: List[str] = []
        frontier = [cell_key]
        for _ in range(max_hops):
            next_frontier = []
            for ck in frontier:
                for dest in self.flows_to.get(ck, []):
                    if dest not in visited_keys and dest != cell_key:
                        visited_keys.append(dest)
                        next_frontier.append(dest)
            frontier = next_frontier
            if not frontier:
                break
        return [self.head_cells[k] for k in visited_keys if k in self.head_cells]

    def traverse_upstream(self, cell_key: str, max_hops: int = 3) -> List[HeadCellNode]:
        visited_keys: List[str] = []
        frontier = [cell_key]
        for _ in range(max_hops):
            next_frontier = []
            for ck in frontier:
                for src in self.flows_from.get(ck, []):
                    if src not in visited_keys and src != cell_key:
                        visited_keys.append(src)
                        next_frontier.append(src)
            frontier = next_frontier
            if not frontier:
                break
        return [self.head_cells[k] for k in visited_keys if k in self.head_cells]

    def wells_in_cells(self, cells: List[HeadCellNode]) -> List[str]:
        wids: List[str] = []
        for c in cells:
            wids.extend(self.cell_wells.get(c.key, []))
        return list(dict.fromkeys(wids))  # unique, order-preserved

    def lowest_head_cells(self, n: int = 10) -> List[HeadCellNode]:
        cells = sorted(self.head_cells.values(), key=lambda c: c.head)
        return cells[:n]

    def lowest_head_near(self, lat: float, lon: float, radius_km: float = 5.0, n: int = 5) -> List[HeadCellNode]:
        nearby = [
            c for c in self.head_cells.values()
            if haversine_km(lat, lon, c.centroid_lat, c.centroid_lon) <= radius_km
        ]
        nearby.sort(key=lambda c: c.head)
        return nearby[:n]

    def deepest_wells(self, n: int = 5) -> List[WellNode]:
        ws = sorted(self.wells.values(), key=lambda w: w.total_depth, reverse=True)
        return ws[:n]

    def downstream_exposure(self, wid: str, hops: int = 3) -> int:
        ck = self.well_cell.get(wid)
        if not ck:
            return 0
        ds = self.traverse_downstream(ck, hops)
        return len(self.wells_in_cells(ds))


# --------------- builder ---------------
def _normalize_formation(lith: str) -> str:
    """Simple normalization: lowercase, strip, collapse whitespace."""
    if not isinstance(lith, str) or not lith.strip():
        return "unknown"
    s = re.sub(r"\s+", " ", lith.strip().lower())
    # Remove numbers / depth markers
    s = re.sub(r"\d+\.?\d*\s*(m|meter|เมตร)", "", s).strip()
    return s if s else "unknown"


def build_graph(
    wells: gpd.GeoDataFrame,
    head_grid: gpd.GeoDataFrame,
    flow_arrows: gpd.GeoDataFrame,
    intervals: pd.DataFrame,
    orientations: pd.DataFrame,
) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    id_col = get_well_id_col(wells)

    # ---- Head cells ----
    row_col = _detect_row_col(head_grid)
    head_col = _detect_col(head_grid, ["head", "HEAD", "Head", "value", "VALUE", "z", "Z"])
    for idx, row in head_grid.iterrows():
        r, c = _get_row_col(row, row_col, idx)
        h = float(row[head_col]) if head_col and pd.notna(row.get(head_col)) else 0.0
        centroid = row.geometry.centroid
        node = HeadCellNode(row=r, col=c, head=h,
                            centroid_lat=centroid.y, centroid_lon=centroid.x)
        kg.head_cells[node.key] = node

    # ---- Wells ----
    depth_col = _detect_col(wells, ["total_depth", "TOTAL_DEPTH", "depth", "DEPTH", "TD"])
    elev_col = _detect_col(wells, ["ground_elev", "GROUND_ELEV", "elev", "ELEV", "elevation", "GE"])

    for _, row in wells.iterrows():
        wid = str(row[id_col])
        pt = row.geometry
        td = float(row[depth_col]) if depth_col and pd.notna(row.get(depth_col)) else 0.0
        ge = float(row[elev_col]) if elev_col and pd.notna(row.get(elev_col)) else 0.0
        wn = WellNode(well_id=wid, lat=pt.y, lon=pt.x, total_depth=td, ground_elev=ge)

        # Well -> HeadCell
        cell_row = find_cell_for_point(pt, head_grid)
        if cell_row is not None:
            r2, c2 = _get_row_col(cell_row, row_col, 0)
            ck = f"{r2}_{c2}"
            if ck in kg.head_cells:
                wn.cell_key = ck
                kg.well_cell[wid] = ck
                kg.cell_wells.setdefault(ck, []).append(wid)

        kg.wells[wid] = wn

    # ---- Near wells (<=2 km) ----
    well_list = list(kg.wells.values())
    for i, w1 in enumerate(well_list):
        for w2 in well_list[i + 1:]:
            d = haversine_km(w1.lat, w1.lon, w2.lat, w2.lon)
            if d <= 2.0:
                kg.near_wells.setdefault(w1.well_id, []).append((w2.well_id, round(d, 3)))
                kg.near_wells.setdefault(w2.well_id, []).append((w1.well_id, round(d, 3)))

    # ---- Intervals & Formations ----
    if not intervals.empty:
        wid_col_i = _detect_col(intervals, ["well_id", "WELL_ID", "Well_ID", "wellid", "name"])
        from_col = _detect_col(intervals, ["from_m", "FROM_M", "from", "top", "TOP", "from_depth"])
        to_col = _detect_col(intervals, ["to_m", "TO_M", "to", "bottom", "BOTTOM", "to_depth"])
        lith_col = _detect_col(intervals, ["lithology", "LITHOLOGY", "lith", "LITH", "description", "rock_type"])

        if wid_col_i and from_col and to_col:
            for _, row in intervals.iterrows():
                wid = str(row[wid_col_i])
                fm = float(row[from_col]) if pd.notna(row.get(from_col)) else 0.0
                tm = float(row[to_col]) if pd.notna(row.get(to_col)) else 0.0
                lith = str(row[lith_col]) if lith_col and pd.notna(row.get(lith_col)) else ""
                formation = _normalize_formation(lith)
                inode = IntervalNode(well_id=wid, from_m=fm, to_m=tm,
                                     lithology=lith, formation=formation)
                kg.intervals.setdefault(wid, []).append(inode)
                if formation not in kg.formations:
                    kg.formations[formation] = FormationNode(name=formation)

    # ---- Flow arrows -> FLOWS_TO edges ----
    if not flow_arrows.empty and not head_grid.empty:
        for _, row in flow_arrows.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            start_pt = Point(coords[0])
            end_pt = Point(coords[-1])
            src_idx = nearest_cell_centroid(start_pt, head_grid)
            dst_idx = nearest_cell_centroid(end_pt, head_grid)
            if src_idx is not None and dst_idx is not None and src_idx != dst_idx:
                src_row = head_grid.iloc[src_idx] if isinstance(src_idx, int) else head_grid.loc[src_idx]
                dst_row = head_grid.iloc[dst_idx] if isinstance(dst_idx, int) else head_grid.loc[dst_idx]
                r1, c1 = _get_row_col(src_row, row_col, src_idx)
                r2, c2 = _get_row_col(dst_row, row_col, dst_idx)
                sk, dk = f"{r1}_{c1}", f"{r2}_{c2}"
                if dk not in kg.flows_to.get(sk, []):
                    kg.flows_to.setdefault(sk, []).append(dk)
                    kg.flows_from.setdefault(dk, []).append(sk)

    return kg


# --------------- helpers ---------------
def _detect_row_col(gdf: gpd.GeoDataFrame):
    """Return (row_col_name, col_col_name) or None."""
    row_names = ["row", "ROW", "Row", "i", "grid_row"]
    col_names = ["col", "COL", "Col", "j", "grid_col", "column"]
    r = _detect_col(gdf, row_names)
    c = _detect_col(gdf, col_names)
    if r and c:
        return (r, c)
    return None


def _detect_col(df, candidates: list) -> Optional[str]:
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    cols = list(df.columns) if hasattr(df, "columns") else []
    for c in candidates:
        if c in cols:
            return c
    return None


def _get_row_col(row, row_col_tuple, fallback_idx):
    if row_col_tuple:
        r = int(row[row_col_tuple[0]]) if pd.notna(row.get(row_col_tuple[0])) else fallback_idx
        c = int(row[row_col_tuple[1]]) if pd.notna(row.get(row_col_tuple[1])) else 0
        return r, c
    return int(fallback_idx), 0
