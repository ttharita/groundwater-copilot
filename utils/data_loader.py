"""Load and validate all local GIS data files."""

import os
import pathlib
import pandas as pd
import geopandas as gpd
import streamlit as st

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _safe_read_geojson(path: str) -> gpd.GeoDataFrame:
    """Read a GeoJSON file, return empty GeoDataFrame on failure."""
    full = DATA_DIR / path
    if not full.exists():
        st.warning(f"Missing file: {full}")
        return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_file(str(full))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        st.warning(f"Error reading {full}: {e}")
        return gpd.GeoDataFrame()


def _safe_read_csv(path: str) -> pd.DataFrame:
    full = DATA_DIR / path
    if not full.exists():
        st.warning(f"Missing file: {full}")
        return pd.DataFrame()
    try:
        return pd.read_csv(str(full))
    except Exception as e:
        st.warning(f"Error reading {full}: {e}")
        return pd.DataFrame()


def load_wells() -> gpd.GeoDataFrame:
    return _safe_read_geojson("wells_wgs84.geojson")


def load_head_grid() -> gpd.GeoDataFrame:
    return _safe_read_geojson("head_grid_wgs84.geojson")


def load_flow_arrows() -> gpd.GeoDataFrame:
    return _safe_read_geojson("flow_dir_arrows_wgs84.geojson")


def load_intervals() -> pd.DataFrame:
    return _safe_read_csv("intervals.csv")


def load_surface_points() -> pd.DataFrame:
    return _safe_read_csv("surface_points.csv")


def load_orientations() -> pd.DataFrame:
    return _safe_read_csv("orientations.csv")
