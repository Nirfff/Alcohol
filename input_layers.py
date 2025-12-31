from pathlib import Path
import warnings
import json

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

# ----------------- SETTINGS -----------------
ox.settings.use_cache = True
ox.settings.log_console = False

PLACE = "Mannheim, Germany"

BASE_OUTDIR = Path(r"your path")
BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

# File paths (single-layer GPKGs)
NETWORK_EDGES_GPKG   = BASE_OUTDIR / "mannheim_walk_edges.gpkg"           # routable network (edges only)
NETWORK_NODES_GPKG   = BASE_OUTDIR / "mannheim_walk_nodes_cache.gpkg"     # internal cache for snapping

STORES_RAW_GPKG      = BASE_OUTDIR / "mannheim_stores_raw.gpkg"
TRANSIT_RAW_GPKG     = BASE_OUTDIR / "mannheim_transit_raw.gpkg"
BUILDINGS_RAW_GPKG   = BASE_OUTDIR / "mannheim_residential_buildings_raw.gpkg"

# FINALS (node geometry)
TRANSIT_ORIGINS_GPKG = BASE_OUTDIR / "mannheim_transit_origins_nodes.gpkg"   # transit-only origins, aggregated per node (count)
RES_NODES_GPKG       = BASE_OUTDIR / "mannheim_residential_buildings_noded.gpkg" # residential buildings snapped to nodes, NOT aggregated
MIDPOINTS_GPKG       = BASE_OUTDIR / "mannheim_midpoints_nodes.gpkg"        # stores/petrol snapped to nodes (not aggregated)

COUNTS_CSV           = BASE_OUTDIR / "mannheim_layer_counts.csv"

# CRSs
CRS_LL = 4326    # WGS84
CRS_M  = 25832   # ETRS89 / UTM 32N

# Parameters
SNAP_MAX_M_TRANSIT = 75.0  # maximum snapping distance for transit-only origins

# OSM tag definitions
TRANSIT_HIGHWAY = "bus_stop"
TRANSIT_RAIL    = ["station", "halt", "stop"]

MID_SHOP_TAGS  = ["kiosk", "convenience", "supermarket", "alcohol", "beverages", "wine"]
MID_AMENITY_FUEL = "fuel"

RES_BUILDING_TAGS = ["residential", "house", "apartments", "detached", "terrace"]

# ----------------- HELPERS -----------------
def to_geodf(df_like, crs=CRS_LL):
    if isinstance(df_like, gpd.GeoDataFrame):
        if df_like.crs is None:
            df_like = df_like.set_crs(crs)
        return df_like.to_crs(crs) if df_like.crs.to_epsg() != crs else df_like
    return gpd.GeoDataFrame(df_like, geometry="geometry", crs=crs)


def centroid_polylike(gdf_ll):
    """Centroid polygons/lines in metric CRS, return to LL. Points unchanged."""
    if gdf_ll is None or gdf_ll.empty:
        return gdf_ll
    gdf_ll = to_geodf(gdf_ll, CRS_LL)
    poly_like = gdf_ll.geom_type.isin(
        ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
    )
    if not poly_like.any():
        return gdf_ll
    gdf_m = gdf_ll.to_crs(CRS_M)
    gdf_m.loc[poly_like, "geometry"] = gdf_m.loc[poly_like, "geometry"].centroid
    return gdf_m.to_crs(CRS_LL)


def features_from_polygon_ll(poly_ll, tags):
    g = ox.features_from_polygon(poly_ll, tags=tags)
    if g is None or g.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS_LL)
    return to_geodf(g, CRS_LL)


def _is_scalarish(v):
    return (
        v is None
        or isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_))
        or (pd.isna(v) if isinstance(v, float) else False)
    )


def sanitize_gdf_for_write(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure all non-geometry columns are scalar so Fiona/GDAL can write them.
    Any list/dict/set/ndarray/tuple objects are JSON-stringified.
    """
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        series = out[col]
        needs_cast = series.map(lambda v: not _is_scalarish(v)).fillna(False)
        if needs_cast.any():
            out[col] = series.apply(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
                else json.dumps(v, ensure_ascii=False)
            )
    return out


def save_single_layer_gpkg(gdf: gpd.GeoDataFrame, path: Path):
    """Write a single-layer GeoPackage safely (no list fields, no empties)."""
    gdf = to_geodf(gdf, CRS_LL)
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS_LL)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = sanitize_gdf_for_write(gdf)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GPKG")

def add_unit_id(gdf: gpd.GeoDataFrame, prefix: str = "") -> gpd.GeoDataFrame:
    """
    Add a 'unit_id' column like:
      prefix + 'pol0', 'pol1', ...
      prefix + 'p0',   'p1',   ...
      prefix + 'l0',   'l1',   ...
    based on geometry type (Polygon, Point, LineString, etc.).
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    def type_code(geom):
        gtype = geom.geom_type
        if gtype in ("Point", "MultiPoint"):
            return "p"
        if gtype in ("LineString", "MultiLineString"):
            return "l"
        if gtype in ("Polygon", "MultiPolygon"):
            return "pol"
        return "g"  # fallback

    codes = gdf.geometry.apply(type_code)

    counters = {}
    ids = []
    for code in codes:
        counters.setdefault(code, 0)
        ids.append(f"{prefix}{code}{counters[code]}")
        counters[code] += 1

    gdf["unit_id"] = ids
    return gdf

# ----------------- CORE: 1-to-1 NEAREST -----------------
def _sjoin_nearest_one(left_m: gpd.GeoDataFrame,
                       right_m: gpd.GeoDataFrame,
                       distance_col: str = "snap_dist_m",
                       right_id_col: str = "node_id") -> gpd.GeoDataFrame:
    """
    Nearest spatial join with exactly one match per left row.
    Deterministic tie-break: (distance, right_id_col).
    Inputs must be in a metric CRS.
    Robust to pandas/geopandas version differences.
    """
    joined = gpd.sjoin_nearest(
        left_m,
        right_m[[right_id_col, "geometry"]],
        how="left",
        distance_col=distance_col,
    )

    # Robustly preserve original left index as '_srcidx'
    joined = joined.reset_index()
    if "_srcidx" not in joined.columns:
        if "index" in joined.columns:
            joined = joined.rename(columns={"index": "_srcidx"})
        else:
            orig_name = left_m.index.name or "index"
            if orig_name in joined.columns:
                joined = joined.rename(columns={orig_name: "_srcidx"})
            else:
                joined.insert(0, "_srcidx", np.arange(len(joined), dtype=int))

    if right_id_col not in joined.columns and "index_right" in joined.columns:
        joined[right_id_col] = joined["index_right"]

    joined = (
        joined
        .sort_values(["_srcidx", distance_col, right_id_col])
        .drop_duplicates("_srcidx", keep="first")
        .set_index("_srcidx")
    )
    return joined


def snap_points_to_nodes(points_ll: gpd.GeoDataFrame,
                         nodes_ll: gpd.GeoDataFrame,
                         max_dist_m: float | None = None) -> gpd.GeoDataFrame:
    """
    Snap each point to its nearest walking-network node, **1:1**.
    Returned geometry = node geometry (not original point).
    Adds ['node_id', 'snap_dist_m'].
    If max_dist_m is provided, rows beyond that distance are dropped.
    """
    if points_ll is None or points_ll.empty:
        base_cols = []
        if isinstance(points_ll, gpd.GeoDataFrame):
            base_cols = [c for c in points_ll.columns if c != "geometry"]
        cols = base_cols + ["node_id", "snap_dist_m", "geometry"]
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=CRS_LL)

    pts_ll   = to_geodf(points_ll, CRS_LL)
    nodes_ll = to_geodf(nodes_ll, CRS_LL)

    pts_m   = pts_ll.to_crs(CRS_M).copy()
    nodes_m = nodes_ll.to_crs(CRS_M).copy()

    # Prepare nodes with a simple id column
    nodes_for_join = (
        nodes_m.reset_index()[["osmid", "geometry"]]
        .rename(columns={"osmid": "node_id"})
    )

    # 1-to-1 nearest join (ties collapsed deterministically)
    joined_m = _sjoin_nearest_one(
        left_m=pts_m,
        right_m=nodes_for_join.rename(columns={"osmid": "node_id"}),
        distance_col="snap_dist_m",
        right_id_col="node_id",
    )

    # Apply distance threshold if requested
    if max_dist_m is not None:
        joined_m = joined_m[joined_m["snap_dist_m"] <= max_dist_m].copy()

    # Bring back to LL
    joined_ll = joined_m.to_crs(CRS_LL)

    # Attach node geometries as the output geometry
    nodes_geom = (
        nodes_ll.reset_index()[["osmid", "geometry"]]
        .rename(columns={"osmid": "node_id", "geometry": "node_geom"})
    )
    joined_ll = joined_ll.merge(nodes_geom, on="node_id", how="left")

    available_cols = set(joined_ll.columns)
    base_cols = [c for c in pts_ll.columns if c in available_cols and c != "geometry"]
    extra_cols = [c for c in ["node_id", "snap_dist_m"] if c in available_cols]
    keep_cols = list(dict.fromkeys(base_cols + extra_cols + ["node_geom"]))

    out = gpd.GeoDataFrame(
        joined_ll[keep_cols],
        geometry="node_geom",
        crs=CRS_LL,
    ).rename_geometry("geometry")

    # Assert 1-to-1 cardinality with possible drops only from max_dist_m
    assert out.index.nunique() == out.shape[0], "Output should have unique rows"
    return out


def map_midpoint_type(row):
    shop = str(row.get("shop", "")).strip().lower()
    amen = str(row.get("amenity", "")).strip().lower()
    if amen == MID_AMENITY_FUEL:
        return "petrol"
    if shop == "kiosk":
        return "kiosk"
    if shop in ("supermarket", "convenience"):
        return "supermarket"
    if shop in ("alcohol", "beverages", "wine"):
        return "liquor"
    return None

# ----------------- MAIN PIPELINE -----------------

def main():
    rows_counts = []

    # 1) Mannheim boundary & walking network
    print("1) Mannheim boundary & walking network…")
    boundary = ox.geocode_to_gdf(PLACE).to_crs(CRS_LL)
    city_poly_ll = boundary.geometry.iloc[0]

    if NETWORK_EDGES_GPKG.exists() and NETWORK_NODES_GPKG.exists():
        print("   Loading walking network from disk…")
        edges_ll = gpd.read_file(NETWORK_EDGES_GPKG)
        nodes_ll = gpd.read_file(NETWORK_NODES_GPKG)
    else:
        print("   Downloading walking network from OSM…")
        G_walk = ox.graph_from_polygon(city_poly_ll, network_type="walk", simplify=True)
        nodes_full, edges_full = ox.graph_to_gdfs(G_walk, nodes=True, edges=True)

        nodes_full = to_geodf(nodes_full, CRS_LL)
        edges_full = to_geodf(edges_full, CRS_LL)

        nodes_tmp = nodes_full.reset_index()
        node_keep = [c for c in ["osmid", "x", "y", "geometry"] if c in nodes_tmp.columns]
        nodes_ll = gpd.GeoDataFrame(nodes_tmp[node_keep], geometry="geometry", crs=CRS_LL)

        edges_tmp = edges_full.reset_index()
        edge_keep = [c for c in ["u", "v", "key", "length", "geometry"] if c in edges_tmp.columns]
        edges_ll = gpd.GeoDataFrame(edges_tmp[edge_keep], geometry="geometry", crs=CRS_LL)

        save_single_layer_gpkg(edges_ll, NETWORK_EDGES_GPKG)
        save_single_layer_gpkg(nodes_ll, NETWORK_NODES_GPKG)
        print(f"   Network edges saved: {NETWORK_EDGES_GPKG}")

    print(f"   Walk edges: {len(edges_ll)}, nodes: {len(nodes_ll)}")

    # 2) Transit (raw, original geometry)
    print("2) Transit (raw, original geometry)…")
    if TRANSIT_RAW_GPKG.exists():
        print("   Loading transit raw from disk…")
        transit_raw = gpd.read_file(TRANSIT_RAW_GPKG)
    else:
        transit_bus = features_from_polygon_ll(city_poly_ll, tags={"highway": TRANSIT_HIGHWAY})
        transit_rail = features_from_polygon_ll(city_poly_ll, tags={"railway": TRANSIT_RAIL})
        transit_raw = pd.concat([transit_bus, transit_rail], ignore_index=True)

        # NEW: assign IDs per downloaded unit
        transit_raw = add_unit_id(transit_raw, prefix="t_")

        save_single_layer_gpkg(transit_raw, TRANSIT_RAW_GPKG)
        print(f"   Transit saved: {TRANSIT_RAW_GPKG}")
    print(f"   Transit features: {len(transit_raw)}")
    rows_counts.append({"category": "transit_raw", "label": "total_raw", "value": len(transit_raw)})

    # Transit origins = transit only, snapped to nodes within 75 m, aggregated per node
    print("   Snapping transit to nodes (75 m) and aggregating…")
    transit_pts = centroid_polylike(transit_raw).copy()
    transit_pts["origin_type"] = "transit"

    cols_for_snap = ["origin_type", "geometry"]
    if "unit_id" in transit_pts.columns:
        cols_for_snap.insert(0, "unit_id")

    transit_snapped_all = snap_points_to_nodes(
        transit_pts[cols_for_snap],
        nodes_ll,
        max_dist_m=SNAP_MAX_M_TRANSIT,
    )

    n_kept = len(transit_snapped_all)
    n_raw  = len(transit_pts)
    rows_counts.append({"category": "transit_snapping", "label": "kept_within_75m", "value": int(n_kept)})
    rows_counts.append({"category": "transit_snapping", "label": "dropped_over_75m", "value": int(max(0, n_raw - n_kept))})

    o_nodes = (
        transit_snapped_all
        .groupby("node_id")
        .agg(
            n_transit_stops=("origin_type", "size"),
            # take the first unit_id mapped to this node
            unit_id=("unit_id", "first"),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )

    o_nodes = gpd.GeoDataFrame(o_nodes, geometry="geometry", crs=CRS_LL)
    save_single_layer_gpkg(o_nodes, TRANSIT_ORIGINS_GPKG)

    rows_counts.append({"category": "transit_origins_nodes", "label": "unique_nodes", "value": int(o_nodes["node_id"].nunique())})
    rows_counts.append({"category": "transit_origins_nodes", "label": "total_rows", "value": int(len(o_nodes))})

    # 3) Stores (raw) + Midpoints (node geometry, NOT aggregated)
    print("3) Stores (raw) & Midpoints…")
    if STORES_RAW_GPKG.exists():
        stores_raw = gpd.read_file(STORES_RAW_GPKG)
    else:
        shops = features_from_polygon_ll(city_poly_ll, tags={"shop": MID_SHOP_TAGS})
        fuel = features_from_polygon_ll(city_poly_ll, tags={"amenity": MID_AMENITY_FUEL})
        stores_raw = pd.concat([shops, fuel], ignore_index=True)

        # NEW: assign IDs per downloaded unit
        stores_raw = add_unit_id(stores_raw, prefix="s_")

        save_single_layer_gpkg(stores_raw, STORES_RAW_GPKG)
    print(f"   Stores (raw): {len(stores_raw)}")

    rows_counts.append({"category": "stores_raw", "label": "total_raw", "value": len(stores_raw)})

    mids = centroid_polylike(stores_raw)
    for c in ("shop", "amenity"):
        if c not in mids.columns:
            mids[c] = pd.Series(dtype="string")
    mids["mid_type"] = mids.apply(map_midpoint_type, axis=1)
    mids = mids.dropna(subset=["mid_type"]).copy()

    cols_for_snap = ["mid_type", "geometry"]
    if "unit_id" in mids.columns:
        cols_for_snap.insert(0, "unit_id")

    mids_snapped = snap_points_to_nodes(
        mids[cols_for_snap],
        nodes_ll,
        max_dist_m=SNAP_MAX_M_TRANSIT,
    )
    save_single_layer_gpkg(mids_snapped, MIDPOINTS_GPKG)

    print(f"   Midpoints (snapped to nodes): {len(mids_snapped)}")
    if not mids_snapped.empty:
        by_type = mids_snapped["mid_type"].value_counts().to_dict()
        for k, v in by_type.items():
            rows_counts.append({"category": "midpoints_nodes", "label": k, "value": int(v)})
        rows_counts.append({"category": "midpoints_nodes", "label": "total", "value": int(len(mids_snapped))})

    # 4) Residential buildings (raw) + noded (NOT aggregated). Keep original floors and computed area.
    print("4) Residential buildings (raw) & noded (not aggregated)…")
    if BUILDINGS_RAW_GPKG.exists():
        res_bld = gpd.read_file(BUILDINGS_RAW_GPKG)
    else:
        res_bld = features_from_polygon_ll(city_poly_ll, tags={"building": RES_BUILDING_TAGS})
        keep_cols = [c for c in ["osmid", "building", "building:levels", "geometry"] if c in res_bld.columns]
        res_bld = res_bld[keep_cols].copy() if keep_cols else res_bld

        # NEW: assign IDs per downloaded unit
        res_bld = add_unit_id(res_bld, prefix="r_")

        save_single_layer_gpkg(res_bld, BUILDINGS_RAW_GPKG)
    print(f"   Residential buildings (raw): {len(res_bld)}")

    rows_counts.append({"category": "residential_buildings_raw", "label": "total_raw", "value": len(res_bld)})

    # compute area from original geometry (m^2); DO NOT impute floors
    res_bld_m = res_bld.to_crs(CRS_M).copy()
    res_bld_m["area_m2"] = res_bld_m.geometry.area
    if "building:levels" not in res_bld_m.columns:
        res_bld_m["building:levels"] = pd.Series(dtype="string")

    # keep unit_id if present
    cols_res = ["area_m2", "building:levels", "geometry"]
    if "unit_id" in res_bld_m.columns:
        cols_res.insert(0, "unit_id")

    res_keep = to_geodf(res_bld_m[cols_res].to_crs(CRS_LL), CRS_LL)
    res_keep = centroid_polylike(res_keep)

    cols_for_snap_res = ["area_m2", "building:levels", "geometry"]
    if "unit_id" in res_keep.columns:
        cols_for_snap_res.insert(0, "unit_id")

    res_noded = snap_points_to_nodes(
        res_keep[cols_for_snap_res],
        nodes_ll,
        max_dist_m=SNAP_MAX_M_TRANSIT,
    )
    save_single_layer_gpkg(res_noded, RES_NODES_GPKG)

    rows_counts.append({"category": "residential_buildings_noded", "label": "total_rows", "value": int(len(res_noded))})
    rows_counts.append({"category": "residential_buildings_noded", "label": "unique_nodes", "value": int(res_noded["node_id"].nunique())})

    # Sanity check: noded cannot exceed raw (it might be less if some inputs had null geometry)
    if len(res_noded) > len(res_bld):
        print("[WARN] Noded buildings exceed raw count – investigate upstream empties or unintended duplication.")

    # 5) SUMMARY COUNTS CSV
    print("5) Writing summary counts…")
    pd.DataFrame(rows_counts).to_csv(COUNTS_CSV, index=False)

    # DONE
    print("Done.")
    print("Outputs:")
    print(f"  Routeable network (edges): {NETWORK_EDGES_GPKG}")
    print(f"  RAW stores:                {STORES_RAW_GPKG}")
    print(f"  RAW transit:               {TRANSIT_RAW_GPKG}")
    print(f"  RAW residential bldgs:     {BUILDINGS_RAW_GPKG}")
    print(f"  FINAL transit origins:     {TRANSIT_ORIGINS_GPKG}")
    print(f"  FINAL residential noded:   {RES_NODES_GPKG}")
    print(f"  FINAL midpoints (nodes):   {MIDPOINTS_GPKG}")
    print("Internal cache (for snapping speed):")
    print(f"  Network nodes cache:       {NETWORK_NODES_GPKG}")


if __name__ == "__main__":
    main()
