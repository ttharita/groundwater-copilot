"""Geospatial helper utilities."""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two WGS-84 points."""
    R = 6371.0
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_cell_for_point(point: Point, head_grid: gpd.GeoDataFrame) -> pd.Series | None:
    """Return the head-grid row whose polygon contains *point*, or None."""
    if head_grid.empty:
        return None
    try:
        mask = head_grid.geometry.contains(point)
        hits = head_grid[mask]
        if not hits.empty:
            return hits.iloc[0]
    except Exception:
        pass
    # Fallback: nearest centroid
    try:
        centroids = head_grid.geometry.centroid
        dists = centroids.distance(point)
        idx = dists.idxmin()
        return head_grid.loc[idx]
    except Exception:
        return None


def nearby_wells(well_geom: Point, wells: gpd.GeoDataFrame, radius_km: float = 2.0, exclude_id: str | None = None):
    """Return DataFrame of wells within *radius_km* of *well_geom*."""
    if wells.empty:
        return pd.DataFrame()
    results = []
    wlat, wlon = well_geom.y, well_geom.x
    id_col = _well_id_col(wells)
    for _, row in wells.iterrows():
        rid = row.get(id_col, "")
        if exclude_id and str(rid) == str(exclude_id):
            continue
        pt = row.geometry
        d = haversine_km(wlat, wlon, pt.y, pt.x)
        if d <= radius_km:
            results.append({"well_id": rid, "distance_km": round(d, 3),
                            "lat": pt.y, "lon": pt.x})
    return pd.DataFrame(results).sort_values("distance_km") if results else pd.DataFrame()


def _well_id_col(gdf: gpd.GeoDataFrame) -> str:
    for c in ("well_id", "WELL_ID", "Well_ID", "name", "NAME", "id", "ID"):
        if c in gdf.columns:
            return c
    return gdf.columns[0] if len(gdf.columns) > 0 else "well_id"


def get_well_id_col(gdf: gpd.GeoDataFrame) -> str:
    return _well_id_col(gdf)


def nearest_cell_centroid(point: Point, head_grid: gpd.GeoDataFrame) -> int | None:
    """Return index of head-grid row whose centroid is nearest to *point*."""
    if head_grid.empty:
        return None
    centroids = head_grid.geometry.centroid
    dists = centroids.distance(point)
    return int(dists.idxmin())
