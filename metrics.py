# -*- coding: utf-8 -*-
"""
Transit→Home alcohol exposure (building-level) for Mannheim

Pipelines
  A) Hybrid network-detour (air prefilter → network OD filter → corridor prefilter → exact Δ)
  B) Buffer-only (air prefilter → corridor buffers → sjoin)

Assumptions
  - All inputs have a valid `node_id` matching walk network nodes.
  - Inputs are in WGS84 (EPSG:4326).

Parameters
  - Stop air cutoff: 500 m
  - Stop-home network cutoff: 500 m
  - Corridor half-widths w: {50, 125, 200} m
  - Network detours Δ: {100, 250, 400} m (= 2*w)
  - Store leverage only for Δ = 250 m

"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from tqdm import tqdm
import networkx as nx

# ------------------------------- CONFIG --------------------------------------

CRS_LL = 4326
CRS_M  = 25832

# Inputs
BASE = "base folder"
IN_HOMES  = os.path.join(BASE, "mannheim_residential_buildings_noded.gpkg")
IN_STORES = os.path.join(BASE, "mannheim_midpoints_nodes.gpkg")
IN_STOPS  = os.path.join(BASE, "mannheim_transit_origins_nodes.gpkg")
IN_WALK_NODES = os.path.join(BASE, "mannheim_walk_nodes_cache.gpkg")
IN_WALK_EDGES = os.path.join(BASE, "mannheim_walk_edges.gpkg")

# Outputs
OUT_DIR = "out folder"
os.makedirs(OUT_DIR, exist_ok=True)

# Pipeline A file paths
A1_PAIRS_AIR_GPKG  = os.path.join(OUT_DIR, "A1_home_stop_pairs_air.gpkg")
A1_LINES_AIR_GPKG  = os.path.join(OUT_DIR, "A1_home_stop_lines_air.gpkg")
A2_PAIRS_NET_GPKG  = os.path.join(OUT_DIR, "A2_home_stop_pairs_network.gpkg")
A2_PATHS_NET_GPKG = os.path.join(OUT_DIR, "A2_home_stop_paths_network.gpkg")
A3_BUFFERS_GPKG    = os.path.join(OUT_DIR, "A3_corridor_buffers.gpkg")
A3_CANDIDATES_CSV  = os.path.join(OUT_DIR, "A3_candidate_store_links.csv")
A4_RISKY_CSV       = os.path.join(OUT_DIR, "A4_risky_by_home_stop_w.csv")
A4_TRIPLETS_CSV    = os.path.join(OUT_DIR, "A4_passed_triplets_home_stop_store_w.csv")
A5_BUILDING_GPKG   = os.path.join(OUT_DIR, "A5_building_exposure_network.gpkg")
A6_LEVERAGE_GPKG   = os.path.join(OUT_DIR, "A6_store_leverage_net_delta_250.gpkg")
A6_LEVERAGE_HEX_GPKG = os.path.join(OUT_DIR, "A6_store_leverage_hex.gpkg")

# Pipeline B file paths
B1_PAIRS_AIR_GPKG  = os.path.join(OUT_DIR, "B1_home_stop_pairs_air.gpkg")
B1_LINES_AIR_GPKG  = os.path.join(OUT_DIR, "B1_home_stop_lines_air.gpkg")
B2_BUFFERS_GPKG    = os.path.join(OUT_DIR, "B2_corridor_buffers.gpkg")
B2_HITS_CSV        = os.path.join(OUT_DIR, "B2_buffer_store_hits.csv")
B3_BUILDING_GPKG   = os.path.join(OUT_DIR, "B3_building_exposure_buffer.gpkg")

C1_HEXGRID_GPKG = os.path.join(OUT_DIR, "C1_hexgrid_500m.gpkg")

# Parameters
STOP_AIR_MAX_M = 500.0
STOP_NET_MAX_M = 500.0
WIDTHS_W = [50.0, 125.0, 200.0]
DELTAS   = {50.0: 100.0, 125.0: 250.0, 200.0: 400.0}

LEVERAGE_W = 125.0
LEVERAGE_DELTA = int(DELTAS[LEVERAGE_W])

# Column names
HOME_ID_COL   = "home_id"
STOP_ID_COL   = "stop_id"
STORE_ID_COL  = "store_id"
STORE_TYPE_COL = "mid_type"

# ≈500 m across flats => hex edge length ≈ 250 m
HEX_EDGE_LEN_M = 250.0
# ------------------------------ HELPERS --------------------------------------

def read_ll(path):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(CRS_LL)
    elif g.crs.to_epsg() != CRS_LL:
        g = g.to_crs(CRS_LL)
    return g

def to_metric(gdf): return gdf.to_crs(CRS_M)

def ensure_id(gdf, id_col, prefix):
    if id_col not in gdf.columns:
        gdf = gdf.reset_index(drop=True).reset_index().rename(columns={"index": id_col})
        gdf[id_col] = gdf[id_col].apply(lambda x: f"{prefix}_{x}")
    return gdf

def stable_indexed(gdf, idx_name):
    gdf = gdf.reset_index(drop=True)
    out = gdf.reset_index().rename(columns={"index": idx_name})
    return out

def build_graph(nodes_gpkg, edges_gpkg):
    nodes = read_ll(nodes_gpkg)
    edges = read_ll(edges_gpkg)
    if "osmid" not in nodes.columns:
        raise ValueError("Walk nodes must contain 'osmid'.")
    if not {"u","v"}.issubset(edges.columns):
        raise ValueError("Walk edges must contain 'u' and 'v'.")
    if "length" not in edges.columns:
        em = to_metric(edges)
        edges["length"] = em.length.values
    G = nx.Graph()
    for _, r in nodes[["osmid"]].iterrows():
        G.add_node(int(r["osmid"]))
    for _, r in edges[["u","v","length"]].iterrows():
        G.add_edge(int(r["u"]), int(r["v"]), weight=float(r["length"]))
    return G

def idx_pairs_air(homes_m, stops_m, r):
    sidx = stops_m.sindex
    rows = []
    for i, geom in enumerate(homes_m.geometry):
        cand = list(sidx.query(geom.buffer(r)))
        if not cand:
            continue
        sub = stops_m.iloc[cand]
        d = sub.distance(geom)
        keep = sub[d <= r]
        for j in keep.index:
            rows.append((i, int(j)))
    return pd.DataFrame(rows, columns=["_hid_idx","_sid_idx"])

def od_lines_gdf(homes_idx, stops_idx, pairs_df):
    lines = []
    for _, r in pairs_df.iterrows():
        hp = homes_idx.geometry.iloc[r["_hid_idx"]]
        sp = stops_idx.geometry.iloc[r["_sid_idx"]]
        lines.append(LineString([sp, hp]))
    return gpd.GeoDataFrame(pairs_df.copy(), geometry=gpd.GeoSeries(lines, crs=CRS_LL))

def save_gdf(path, gdf):
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame when writing {path}")
    gdf.to_file(path, driver="GPKG")


def _make_hexagon(cx, cy, a):
    # pointy-top hex centered at (cx, cy) with edge length a (meters)
    import math
    from shapely.geometry import Polygon
    angles = [30, 90, 150, 210, 270, 330]  # pointy-top orientation
    coords = [(cx + a*math.cos(math.radians(t)),
               cy + a*math.sin(math.radians(t))) for t in angles]
    return Polygon(coords)

def _hex_grid_from_bounds(bounds, a):
    # generate pointy-top hex grid covering bounds with edge length a
    import math
    minx, miny, maxx, maxy = bounds
    dx = math.sqrt(3) * a
    dy = 1.5 * a
    hexes = []
    j = 0
    y = miny - 2*a
    while y <= maxy + 2*a:
        x_offset = 0 if (j % 2 == 0) else dx/2
        x = (minx - 2*a) + x_offset
        while x <= maxx + 2*a:
            hexes.append(_make_hexagon(x, y, a))
            x += dx
        y += dy
        j += 1
    return hexes

def _ensure_hexgrid(homes_ll: gpd.GeoDataFrame, path_hex: str, hex_edge_len_m: float):
    """
    Build or load the 500 m hex grid over the convex hull of homes.
    """
    if os.path.exists(path_hex):
        return gpd.read_file(path_hex)
    homes_m = to_metric(homes_ll)
    hull = homes_m.unary_union.convex_hull.buffer(500)  # small safety margin
    hexes = _hex_grid_from_bounds(hull.bounds, hex_edge_len_m)
    grid = gpd.GeoDataFrame(
        {"hex_id": range(len(hexes))},
        geometry=gpd.GeoSeries(hexes, crs=CRS_M)
    )
    grid = grid[grid.geometry.intersects(hull)]
    grid = grid.to_crs(CRS_LL)
    save_gdf(path_hex, grid)
    return grid


def _infer_area_series(df: pd.DataFrame) -> pd.Series:
    """
    Prefer explicit area_m2 from homes; otherwise compute from geometry in metric CRS.
    """
    if "area_m2" in df.columns:
        s = pd.to_numeric(df["area_m2"], errors="coerce")
    else:
        if "geometry" in df.columns:
            s = to_metric(gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_LL)).area
        else:
            s = pd.Series(np.nan, index=df.index)
    return s

def _infer_floors_series(df: pd.DataFrame) -> pd.Series:
    """
    Prefer explicit building:levels; fallback to 1 where missing.
    """
    if "building:levels" in df.columns:
        f = pd.to_numeric(df["building:levels"], errors="coerce")
    else:
        f = pd.Series(np.nan, index=df.index)
    return f.fillna(1).clip(lower=1)
# --------------------------- PIPE A: NETWORK ---------------------------------

def run_pipeline_A():
    print("=== PIPELINE A: Hybrid network-detour ===")

    # Load inputs
    # NEW (A1 minimal change: use unit_id → home_id/stop_id)
    # Load inputs
    homes_ll = read_ll(IN_HOMES)
    stops_ll = read_ll(IN_STOPS)
    stores_ll = read_ll(IN_STORES)

    # assertions
    for df, nm in [(homes_ll, "homes"), (stops_ll, "stops"), (stores_ll, "stores")]:
        assert "unit_id" in df.columns, f"{nm} layer missing 'unit_id'"
        assert "node_id" in df.columns, f"{nm} layer missing 'node_id'"

    # rename unit_id to the canonical columns used elsewhere in the pipeline
    homes_ll  = homes_ll.rename(columns={"unit_id": HOME_ID_COL})
    stops_ll  = stops_ll.rename(columns={"unit_id": STOP_ID_COL})
    stores_ll = stores_ll.rename(columns={"unit_id": STORE_ID_COL})

    # (stores_ll stays as-is for A1; we’ll rename in A4/A6 when needed)

    homes_idx = stable_indexed(homes_ll, "_hid_idx")
    stops_idx = stable_indexed(stops_ll, "_sid_idx")

    # Build graph
    G = build_graph(IN_WALK_NODES, IN_WALK_EDGES)

    # ---------- A1: uniform buffer-based home–stop proximity filter ----------

    if os.path.exists(A1_PAIRS_AIR_GPKG) and os.path.exists(A1_LINES_AIR_GPKG):
        print("A1: loaded from disk.")
        gdf_air = gpd.read_file(A1_PAIRS_AIR_GPKG)
        # keep the non-geometry columns as pairs_air for later steps (A2)
        pairs_air = gdf_air.drop(columns="geometry").copy()
    else:
        print("A1: buffering homes by 500 m and spatial-joining stops...")

        # metric copies with indices
        homes_m = homes_idx.to_crs(CRS_M)
        stops_m = stops_idx.to_crs(CRS_M)

        R_AIR = 500  # same threshold as your old airline 500 m

        # 500 m buffers around homes
        homes_buf = homes_m.copy()
        homes_buf["geometry"] = homes_buf.geometry.buffer(R_AIR)

        # spatial join: stops inside home buffer
        joined = gpd.sjoin(
            stops_m,
            homes_buf[["_hid_idx", "geometry"]],
            predicate="within",
            how="inner",
        )

        # build pairs_air with indices + metadata
        pairs_air = (
            joined.rename(columns={"_hid_idx_right": "_hid_idx"})
            [["_hid_idx", "_sid_idx"]]
            .merge(
                homes_idx[["_hid_idx", HOME_ID_COL, "node_id"]]
                .rename(columns={"node_id": "home_node"}),
                on="_hid_idx"
            )
            .merge(
                stops_idx[["_sid_idx", STOP_ID_COL, "node_id"]]
                .rename(columns={"node_id": "stop_node"}),
                on="_sid_idx"
            )
        )

        # -------- A1 outputs --------

        # 1) pairs: one row per home–stop pair (geometry = home point, like before)
        gdf_air = gpd.GeoDataFrame(
            pairs_air.merge(homes_idx[["_hid_idx", "geometry"]], on="_hid_idx"),
            geometry="geometry",
            crs=CRS_LL,
        )
        save_gdf(A1_PAIRS_AIR_GPKG, gdf_air)

        # 2) "lines" file now = home buffers instead of airline lines
        #    (only for homes that actually got at least one stop)
        used_hids = pairs_air["_hid_idx"].unique()
        homes_buf_used = homes_buf[homes_buf["_hid_idx"].isin(used_hids)].copy()
        homes_buf_ll = homes_buf_used.to_crs(CRS_LL)
        save_gdf(A1_LINES_AIR_GPKG, homes_buf_ll)


    # ---------- A2: network filter ----------
    if os.path.exists(A2_PAIRS_NET_GPKG) and os.path.exists(A2_PATHS_NET_GPKG):
        gdf_net = gpd.read_file(A2_PAIRS_NET_GPKG)
        od_net  = gpd.read_file(A2_PATHS_NET_GPKG)
        print("A2: loaded from disk.")
    else:
        print("A2: Network filter per stop ≤ 500 m ...")

        # group by stop node for efficient single-source search
        by_stop = defaultdict(list)
        for _, r in pairs_air.iterrows():
            by_stop[int(r["stop_node"])].append((int(r["_hid_idx"]), int(r["home_node"])))

        net_rows = []
        for s_node, hid_list in tqdm(by_stop.items(), total=len(by_stop)):
            lengths = nx.single_source_dijkstra_path_length(
                G, source=s_node, cutoff=STOP_NET_MAX_M, weight="weight"
            )
            for hid_idx, h_node in hid_list:
                d = lengths.get(h_node, None)
                if d is not None and d <= STOP_NET_MAX_M:
                    net_rows.append((hid_idx, s_node, d))

        if not net_rows:
            raise RuntimeError("No (home, stop) pairs survived the 500 m network filter.")

        # build filtered dataframe
        net_df = pd.DataFrame(net_rows, columns=["_hid_idx", "stop_node", "d_stop_home"])
        net_df = (
            net_df.merge(homes_idx[["_hid_idx", HOME_ID_COL, "node_id"]].rename(columns={"node_id": "home_node"}),
                         on="_hid_idx", how="left")
                   .merge(stops_idx[["_sid_idx", STOP_ID_COL, "node_id"]].rename(columns={"node_id": "stop_node"}),
                          on="stop_node", how="left")
        )

        # save filtered pairs (for joins / metadata)
        gdf_net = gpd.GeoDataFrame(net_df.merge(homes_idx[["_hid_idx", "geometry"]],
                                                on="_hid_idx", how="left"),
                                   geometry="geometry", crs=CRS_LL)
        save_gdf(A2_PAIRS_NET_GPKG, gdf_net)

        # build actual network path geometries for each valid pair
        print("A2: computing and saving network paths ...")
        nodes_ll = read_ll(IN_WALK_NODES)
        node_xy = nodes_ll.set_index("osmid").geometry.apply(lambda p: (p.x, p.y)).to_dict()

        geoms = []
        for _, r in tqdm(net_df.iterrows(), total=len(net_df)):
            s = int(r["stop_node"]);
            h = int(r["home_node"])
            try:
                path_nodes = nx.shortest_path(G, source=s, target=h, weight="weight")
            except nx.NetworkXNoPath:
                path_nodes = [s, h]  # degenerate fallback

            # map nodes to coords (allow missing)
            coords = [node_xy.get(n) for n in path_nodes]
            coords = [c for c in coords if c is not None]

            if len(coords) >= 2:
                geoms.append(LineString(coords))
                continue

            # fallback: straight segment between endpoints if possible
            p0 = node_xy.get(s);
            p1 = node_xy.get(h)
            if p0 is not None and p1 is not None and p0 != p1:
                geoms.append(LineString([p0, p1]))
            else:
                geoms.append(None)  # last resort; handle None downstream if needed

        od_net = gpd.GeoDataFrame(net_df.copy(), geometry=gpd.GeoSeries(geoms, crs=CRS_LL))
        save_gdf(A2_PATHS_NET_GPKG, od_net)

    # ---------- A3: path buffers (network) & store prefilter ----------
    if os.path.exists(A3_BUFFERS_GPKG) and os.path.exists(A3_CANDIDATES_CSV):
        corr_buf = gpd.read_file(A3_BUFFERS_GPKG)
        cand_df  = pd.read_csv(A3_CANDIDATES_CSV)
        print("A3: loaded from disk.")

    else:
        print("A3: Endpoint-based air buffers & store prefilter ...")

        # read network paths produced in A2 (contains _hid_idx, _sid_idx, d_stop_home, nodes)
        od_net = gpd.read_file(A2_PATHS_NET_GPKG)

        # metric copies for homes, stops, and stores
        homes_m = to_metric(homes_idx[["_hid_idx", "geometry"]].copy())
        stops_m = to_metric(stops_idx[["_sid_idx", "geometry"]].copy())
        stores_m = to_metric(stores_ll[[STORE_ID_COL, "geometry"]].copy())
        stores_idx = stores_m.sindex

        # quick lookup: home/stop geometry in metric CRS by index
        home_geom = homes_m.set_index("_hid_idx").geometry
        stop_geom = stops_m.set_index("_sid_idx").geometry

        buf_rows = []
        cand_rows = []

        # iterate stop–home pairs and widths; use endpoint buffers as candidate region
        for i, r in tqdm(list(od_net.iterrows()), total=len(od_net)):
            hid = int(r["_hid_idx"])
            sid = int(r["_sid_idx"])
            L = float(r["d_stop_home"])  # shortest-path stop–home distance (m)

            h_geom = home_geom.get(hid, None)
            s_geom = stop_geom.get(sid, None)
            if h_geom is None or s_geom is None:
                continue

            for w in WIDTHS_W:
                # full detour budget corresponding to this half-width
                delta_thr = float(DELTAS[w])  # e.g. 100, 250, 400
                radius = L + delta_thr  # air-distance buffer radius (m)

                # intersection of home and stop buffers: candidate region for mapping
                h_buf = h_geom.buffer(radius)
                s_buf = s_geom.buffer(radius)
                inter = h_buf.intersection(s_buf)

                buf_rows.append({"_pair_idx": i, "w": w, "geometry": inter})

                if inter.is_empty:
                    continue

                # candidate stores: must be within radius of both endpoints
                # 1) prefilter via spatial index with home buffer
                cand_idx = list(stores_idx.query(h_buf))
                if not cand_idx:
                    continue

                sub = stores_m.iloc[cand_idx].copy()
                # 2) exact distance to both home and stop
                d_home = sub.geometry.distance(h_geom)
                d_stop = sub.geometry.distance(s_geom)
                mask = (d_home <= radius) & (d_stop <= radius)
                sub = sub[mask]

                for _, srow in sub.iterrows():
                    cand_rows.append({
                        "_pair_idx": i,
                        "w": w,
                        STORE_ID_COL: srow[STORE_ID_COL]
                    })

        # Pair metadata (no geometry to avoid *_x/*_y collisions)
        pair_meta = (
            od_net.reset_index()
            .rename(columns={"index": "_pair_idx"})
            [["_pair_idx",
              "_hid_idx", "_sid_idx",
              HOME_ID_COL, STOP_ID_COL,
              "home_node", "stop_node", "d_stop_home"]]
        )

        # Save "buffers" (here: intersection of endpoint discs) for mapping
        buf_df = pd.DataFrame(buf_rows)
        if "geometry" not in buf_df.columns:
            raise RuntimeError("A3: buf_rows missing 'geometry' key.")
        buf_gdf_m = gpd.GeoDataFrame(buf_df, geometry="geometry", crs=CRS_M)
        buf_gdf_ll = buf_gdf_m.to_crs(CRS_LL)
        corr_buf = gpd.GeoDataFrame(
            buf_gdf_ll.merge(pair_meta, on="_pair_idx", how="left"),
            geometry="geometry", crs=CRS_LL
        )
        save_gdf(A3_BUFFERS_GPKG, corr_buf)

        # Save candidates (O–D–store by w)
        cand_df = pd.DataFrame(cand_rows).merge(pair_meta, on="_pair_idx", how="left")
        cand_df = cand_df[[HOME_ID_COL, STOP_ID_COL,
                           "_hid_idx", "_sid_idx",
                           "home_node", "stop_node", "d_stop_home",
                           "w", STORE_ID_COL]]
        cand_df.to_csv(A3_CANDIDATES_CSV, index=False)

    # ---------- A4: exact network detour ----------
    if os.path.exists(A4_RISKY_CSV) and os.path.exists(A4_TRIPLETS_CSV):
        risky_df   = pd.read_csv(A4_RISKY_CSV)
        triplet_df = pd.read_csv(A4_TRIPLETS_CSV)
        print("A4: loaded from disk.")
    else:
        print("A4: Network Δ checks ...")
        store_node_map = {
            str(k): int(v)
            for k, v in stores_ll.set_index(STORE_ID_COL)["node_id"].items()
        }
        cutoff_store = max(DELTAS.values()) + STOP_NET_MAX_M + 25.0
        stop_to_store, home_to_store = {}, {}

        def dists_from(source):
            return nx.single_source_dijkstra_path_length(
                G, source=source, cutoff=cutoff_store, weight="weight"
            )

        risky_rows = []
        triplet_rows = []

        if len(cand_df):
            key_cols = ["_hid_idx", "_sid_idx", "w", "d_stop_home", "home_node", "stop_node"]
            groups = cand_df.groupby(key_cols)

            for (hid_idx, sid_idx, w, dOD, h_node, s_node), g in tqdm(list(groups), total=groups.ngroups):
                w = float(w)
                delta_thr = DELTAS[w]
                s_node = int(s_node)
                h_node = int(h_node)

                if s_node not in stop_to_store:
                    stop_to_store[s_node] = dists_from(s_node)
                if h_node not in home_to_store:
                    home_to_store[h_node] = dists_from(h_node)

                dO = stop_to_store[s_node]
                dD = home_to_store[h_node]

                risky_any = False
                seen_stores = set()

                for _, sr in g.iterrows():
                    s_id = str(sr[STORE_ID_COL])
                    m_node = store_node_map.get(s_id, None)
                    if m_node is None:
                        continue

                    dOM = dO.get(m_node, None)
                    dMD = dD.get(m_node, None)
                    if dOM is None or dMD is None:
                        continue

                    det = dOM + dMD - dOD
                    if det <= delta_thr:
                        risky_any = True
                        seen_stores.add(s_id)
                        triplet_rows.append({
                            HOME_ID_COL: sr[HOME_ID_COL],
                            STOP_ID_COL: sr[STOP_ID_COL],
                            STORE_ID_COL: s_id,
                            "w": w,
                            "delta_m": det
                        })

                if risky_any:
                    sr0 = g.iloc[0]
                    risky_rows.append({
                        HOME_ID_COL: sr0[HOME_ID_COL],
                        STOP_ID_COL: sr0[STOP_ID_COL],
                        "w": w,
                        "risky": 1,
                        "n_distinct_stores_pass": len(seen_stores)
                    })

        # write triplets (store-level passes)
        triplet_df = pd.DataFrame(triplet_rows)
        if triplet_df.empty:
            triplet_df = pd.DataFrame(
                columns=[HOME_ID_COL, STOP_ID_COL, STORE_ID_COL, "w", "delta_m"]
            )
        triplet_df.to_csv(A4_TRIPLETS_CSV, index=False)

        # write risky summary (home–stop–w level)
        risky_df = pd.DataFrame(risky_rows)
        if risky_df.empty:
            risky_df = pd.DataFrame(
                columns=[HOME_ID_COL, STOP_ID_COL, "w", "risky", "n_distinct_stores_pass"]
            )
        risky_df.to_csv(A4_RISKY_CSV, index=False)


    # ---------- A5: building-level exposure ----------
    if os.path.exists(A5_BUILDING_GPKG):
        print("A5: loaded from disk.")
    else:
        net_pairs = gpd.read_file(A2_PAIRS_NET_GPKG)
        Nd = (net_pairs.groupby(HOME_ID_COL)[STOP_ID_COL].nunique()
                        .rename("N_d").reset_index())
        homes_base = homes_ll[[HOME_ID_COL, "geometry"]].copy()
        out_home = homes_base.merge(Nd, on=HOME_ID_COL, how="left")

        risky_df_local = pd.read_csv(A4_RISKY_CSV)
        for w in WIDTHS_W:
            tmp = risky_df_local[risky_df_local["w"] == w]
            if len(tmp):
                risky_per_home = (tmp.groupby(HOME_ID_COL)["risky"].sum()
                                    .rename(f"n_risky_stops_w{int(w)}").reset_index())
                nstores_per_home = (tmp.groupby(HOME_ID_COL)["n_distinct_stores_pass"].sum()
                                      .rename(f"n_stores_w{int(w)}").reset_index())
                out_home = out_home.merge(risky_per_home, on=HOME_ID_COL, how="left")\
                                   .merge(nstores_per_home, on=HOME_ID_COL, how="left")
            else:
                out_home[f"n_risky_stops_w{int(w)}"] = 0
                out_home[f"n_stores_w{int(w)}"] = 0
            out_home[f"n_risky_stops_w{int(w)}"] = out_home[f"n_risky_stops_w{int(w)}"].fillna(0).astype(int)
            out_home[f"n_stores_w{int(w)}"] = out_home[f"n_stores_w{int(w)}"].fillna(0).astype(int)
            ratio = out_home[f"n_risky_stops_w{int(w)}"] / out_home["N_d"]
            out_home[f"idx_risky_w{int(w)}"] = ratio.where(out_home["N_d"] > 0, np.nan)

        save_gdf(A5_BUILDING_GPKG, out_home)

    # ---------- A6: store leverage at Δ=250 ----------
    if os.path.exists(A6_LEVERAGE_GPKG):
        print("A6: loaded from disk.")
    else:
        triplet_df_local = pd.read_csv(A4_TRIPLETS_CSV)
        g125 = triplet_df_local[triplet_df_local["w"] == LEVERAGE_W].copy()
        if len(g125):
            # 1) distinct stops per (home, store)
            n_sd = (g125.groupby([HOME_ID_COL, STORE_ID_COL])[STOP_ID_COL]
                        .nunique()
                        .rename("n_sd")
                        .reset_index())

            # 2) recompute N_d from A2 network pairs (no dependence on A5)
            pairs_net = gpd.read_file(A2_PAIRS_NET_GPKG)[[HOME_ID_COL, STOP_ID_COL]]
            Nd = (pairs_net.groupby(HOME_ID_COL)[STOP_ID_COL]
                         .nunique()
                         .rename("N_d")
                         .reset_index())
            n_sd = n_sd.merge(Nd, on=HOME_ID_COL, how="left")
            n_sd["N_d"] = n_sd["N_d"].fillna(0)

            # 3) attach building weights w_d = area * floors from homes
            homes_ll = read_ll(IN_HOMES)
            if "unit_id" in homes_ll.columns and HOME_ID_COL not in homes_ll.columns:
                homes_ll = homes_ll.rename(columns={"unit_id": HOME_ID_COL})
            cols = [HOME_ID_COL, "geometry"]
            for extra in ["area_m2", "building:levels"]:
                if extra in homes_ll.columns:
                    cols.append(extra)
            base_attrs = homes_ll[cols].copy()

            n_sd = n_sd.merge(base_attrs, on=HOME_ID_COL, how="left")
            n_sd["w_d"] = (_infer_area_series(n_sd) * _infer_floors_series(n_sd)).astype(float)
            n_sd = n_sd[n_sd["w_d"] > 0]

            # 4) weighted contribution per (home, store): w_d * (n_sd / N_d)
            n_sd["contrib"] = np.where(
                n_sd["N_d"] > 0,
                n_sd["w_d"] * (n_sd["n_sd"] / n_sd["N_d"]),
                0.0
            )

            # 5) aggregate to store-level leverage
            Ls = (n_sd.groupby(STORE_ID_COL)["contrib"].sum()
                  .rename(f"Ls_net_delta_{LEVERAGE_DELTA}")
                  .reset_index())

            keep = [STORE_ID_COL, "geometry"]
            stores_ll_local = read_ll(IN_STORES)
            if "unit_id" in stores_ll_local.columns and STORE_ID_COL not in stores_ll_local.columns:
                stores_ll_local = stores_ll_local.rename(columns={"unit_id": STORE_ID_COL})
            if STORE_TYPE_COL in stores_ll_local.columns:
                keep.append(STORE_TYPE_COL)

            Ls_gdf = (stores_ll_local[keep]
                      .merge(Ls, on=STORE_ID_COL, how="left")
                      .fillna({f"Ls_net_delta_{LEVERAGE_DELTA}": 0.0}))

            # --- NEW: hex-aggregated leverage (sum within each cell) -----------
            # load hex grid (same one used in Section C)
            hexgrid = _ensure_hexgrid(homes_ll, C1_HEXGRID_GPKG, HEX_EDGE_LEN_M)[["hex_id", "geometry"]]


            # spatial join stores -> hex
            stores_hex = gpd.sjoin(
                Ls_gdf,
                hexgrid,
                how="inner",
                predicate="intersects"
            ).drop(columns=["index_right"])

            # sum leverage per hex (all store types)
            lev_col = f"Ls_net_delta_{LEVERAGE_DELTA}"
            hex_lev = (stores_hex
                       .groupby("hex_id")[lev_col]
                       .sum()
                       .rename(f"Lg_sum_delta_{LEVERAGE_DELTA}")
                       .reset_index())

            # optional: sums per store type, if available
            if STORE_TYPE_COL in stores_hex.columns:
                hex_lev_type = (stores_hex
                                .groupby(["hex_id", STORE_TYPE_COL])[lev_col]
                                .sum()
                                .reset_index()
                                .pivot(index="hex_id",
                                       columns=STORE_TYPE_COL,
                                       values=lev_col))
                # e.g. columns like 'kiosk', 'supermarket' with summed leverage
                hex_lev = hex_lev.merge(hex_lev_type, on="hex_id", how="left")

            # merge back onto hexgrid
            hex_lev_gdf = hexgrid.merge(hex_lev, on="hex_id", how="left")
            # fill missing leverage with 0 (no stores / no contributing stores)
            lev_col = f"Lg_sum_delta_{LEVERAGE_DELTA}"
            hex_lev_gdf[lev_col] = hex_lev_gdf[lev_col].fillna(0.0)
            # save both: store-level and hex-level leverage
            save_gdf(A6_LEVERAGE_GPKG, Ls_gdf)
            save_gdf(A6_LEVERAGE_HEX_GPKG, hex_lev_gdf)


print("Pipeline A complete.")

# -------------------------- PIPE B: BUFFER-ONLY -------------------------------
# Simplified corridor-based index, air-only OD, single width w = 125 m
# (Δ = 250 m detour threshold)

# Use the same "middle" width as in the main network pipeline
BUF_W = LEVERAGE_W  # 125.0 m half-width  =>  Δ = 250 m

def run_pipeline_B():
    print("=== PIPELINE B: Buffer-only (w = 125 m; Δ = 250 m) ===")

    # Load inputs exactly as in Pipeline A (unit_id → home_id/stop_id/store_id)
    homes_ll = read_ll(IN_HOMES)
    stops_ll = read_ll(IN_STOPS)
    stores_ll = read_ll(IN_STORES)

    # Basic assertions for consistency with Pipeline A
    for df, nm in [(homes_ll, "homes"), (stops_ll, "stops"), (stores_ll, "stores")]:
        assert "unit_id" in df.columns, f"{nm} layer missing 'unit_id'"
        # node_id is not needed for the buffer-only variant, but we expect it in the data
        assert "node_id" in df.columns, f"{nm} layer missing 'node_id'"

    # Canonical ID columns
    homes_ll  = homes_ll.rename(columns={"unit_id": HOME_ID_COL})
    stops_ll  = stops_ll.rename(columns={"unit_id": STOP_ID_COL})
    stores_ll = stores_ll.rename(columns={"unit_id": STORE_ID_COL})

    # Stable integer indices for pairing
    homes_idx = stable_indexed(homes_ll, "_hid_idx")
    stops_idx = stable_indexed(stops_ll, "_sid_idx")

    # ------------------------------------------------------------------ B1 ---
    #  Air-distance (500 m) home–stop pairs and straight OD lines
    # -------------------------------------------------------------------------
    if os.path.exists(B1_PAIRS_AIR_GPKG) and os.path.exists(B1_LINES_AIR_GPKG):
        gdf_b1    = gpd.read_file(B1_PAIRS_AIR_GPKG)
        od_b      = gpd.read_file(B1_LINES_AIR_GPKG)
        pairs_air = gdf_b1.drop(columns="geometry").copy()
        print("B1: loaded from disk.")
    else:
        print("B1: Euclidean prefilter homes→stops ≤ 500 m ...")
        homes_m = to_metric(homes_idx)
        stops_m = to_metric(stops_idx)

        # like A1: for each home, find nearby stops (≤ 500 m air)
        pairs_air = idx_pairs_air(homes_m, stops_m, STOP_AIR_MAX_M)
        if pairs_air.empty:
            raise RuntimeError("No (home, stop) pairs within 500 m air distance (buffer-only).")

        # Attach IDs (no node_id needed here)
        pairs_air = (
            pairs_air
            .merge(homes_idx[["_hid_idx", HOME_ID_COL]], on="_hid_idx", how="left")
            .merge(stops_idx[["_sid_idx", STOP_ID_COL]], on="_sid_idx", how="left")
        )

        # Save a home-centric point layer (geometry = home) for reference
        gdf_b1 = (
            pairs_air
            .merge(homes_idx[["_hid_idx", "geometry"]].rename(columns={"geometry": "home_geom"}),
                   on="_hid_idx", how="left")
            .merge(stops_idx[["_sid_idx", "geometry"]].rename(columns={"geometry": "stop_geom"}),
                   on="_sid_idx", how="left")
        )
        gdf_b1 = gpd.GeoDataFrame(
            gdf_b1.drop(columns=["home_geom", "stop_geom"]).copy(),
            geometry=gdf_b1["home_geom"],
            crs=CRS_LL
        )
        save_gdf(B1_PAIRS_AIR_GPKG, gdf_b1)

        # Straight OD segment lines from stop → home (for buffering)
        od_b = od_lines_gdf(homes_idx, stops_idx, pairs_air[["_hid_idx", "_sid_idx"]]) \
                   .merge(pairs_air[[HOME_ID_COL, STOP_ID_COL]], left_index=True, right_index=True)
        save_gdf(B1_LINES_AIR_GPKG, od_b)

    # ------------------------------------------------------------------ B2 ---
    #  Corridor buffers (w = 125 m) and store hits (buffer-only)
    # -------------------------------------------------------------------------
    if os.path.exists(B2_BUFFERS_GPKG) and os.path.exists(B2_HITS_CSV):
        b_corr_buf = gpd.read_file(B2_BUFFERS_GPKG)
        b_hits     = pd.read_csv(B2_HITS_CSV)
        print("B2: loaded from disk.")
    else:
        print(f"B2: Corridor buffers (w = {BUF_W} m) & store sjoin ...")

        # Metric copies
        od_b_m    = to_metric(od_b)
        stores_m  = to_metric(stores_ll)
        stores_sx = stores_m.sindex

        buf_rows = []
        hit_rows = []

        for i, r in tqdm(list(od_b_m.iterrows()), total=len(od_b_m)):
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue

            # Single width: BUF_W (125 m half-width)
            w = float(BUF_W)
            buf = geom.buffer(w)
            buf_rows.append({"_pair_idx": i, "w": w, "geometry": buf})

            # prefilter via spatial index, then exact within(buffer)
            cand = list(stores_sx.query(buf))
            if cand:
                sub = stores_m.iloc[cand]
                sub = sub[sub.geometry.within(buf)]
                for _, srow in sub.iterrows():
                    hit_rows.append({
                        "_pair_idx": i,
                        "w": w,
                        STORE_ID_COL: srow[STORE_ID_COL]
                    })

        # 1) Corridor buffers → GeoPackage
        b_buf_df = pd.DataFrame(buf_rows)
        if "geometry" not in b_buf_df.columns:
            raise RuntimeError("B2: buf_rows missing 'geometry' key.")
        b_buf_gdf_m  = gpd.GeoDataFrame(b_buf_df, geometry="geometry", crs=CRS_M)
        b_buf_gdf_ll = b_buf_gdf_m.to_crs(CRS_LL)

        # 2) Pair metadata WITHOUT geometry (avoid *_x / *_y clashes)
        pair_meta_b = (
            od_b.reset_index()
                .rename(columns={"index": "_pair_idx"})
                [[ "_pair_idx",
                   "_hid_idx", "_sid_idx",
                   HOME_ID_COL, STOP_ID_COL ]]
        )

        # 3) Join & save (active geometry = corridor buffer)
        b_corr_buf = b_buf_gdf_ll.merge(pair_meta_b, on="_pair_idx", how="left")
        b_corr_buf = gpd.GeoDataFrame(b_corr_buf, geometry="geometry", crs=CRS_LL)
        save_gdf(B2_BUFFERS_GPKG, b_corr_buf)

        # Store hits: one row per (home, stop, store) for w = BUF_W
        b_hits = pd.DataFrame(hit_rows).merge(
            od_b.reset_index().rename(columns={"index": "_pair_idx"}),
            on="_pair_idx", how="left"
        )
        b_hits.to_csv(B2_HITS_CSV, index=False)

    # ------------------------------------------------------------------ B3 ---
    #  Building-level buffer-only exposure (E_d^buf at w = 125 m)
    # -------------------------------------------------------------------------
    if os.path.exists(B3_BUILDING_GPKG):
        print("B3: loaded from disk.")
    else:
        print("B3: Building-level exposure (buffer-only, w = 125 m) ...")

        # N_d_air: number of air-reachable stops (≤ 500 m) per home
        Nd_air = (
            pairs_air.groupby(HOME_ID_COL)[STOP_ID_COL]
                     .nunique()
                     .rename("N_d_air")
                     .reset_index()
        )

        out_home = (
            homes_ll[[HOME_ID_COL, "geometry"]]
            .merge(Nd_air, on=HOME_ID_COL, how="left")
            .fillna({"N_d_air": 0})
        )

        b_hits_local = pd.read_csv(B2_HITS_CSV)

        # Only the single width BUF_W (125)
        w = float(BUF_W)
        col_suffix = f"{int(w)}"  # "125"

        tmp = b_hits_local[b_hits_local["w"] == w]
        if len(tmp):
            # A stop is "risky" if its OD corridor buffer contains ≥ 1 store
            risky_counts = (
                tmp.groupby([HOME_ID_COL, STOP_ID_COL])
                   .size()
                   .rename("has_store")
                   .reset_index()
            )

            # n_risky_stops: number of risky stops per home
            risky_per_home = (
                risky_counts.groupby(HOME_ID_COL)["has_store"]
                            .count()
                            .rename(f"n_risky_stops_w{col_suffix}_buf")
                            .reset_index()
            )

            # n_stores: distinct stores per home across all its risky OD corridors
            nstores_per_home = (
                tmp.groupby(HOME_ID_COL)[STORE_ID_COL]
                   .nunique()
                   .rename(f"n_stores_w{col_suffix}_buf")
                   .reset_index()
            )

            out_home = (
                out_home
                .merge(risky_per_home,  on=HOME_ID_COL, how="left")
                .merge(nstores_per_home, on=HOME_ID_COL, how="left")
            )
        else:
            out_home[f"n_risky_stops_w{col_suffix}_buf"] = 0
            out_home[f"n_stores_w{col_suffix}_buf"]      = 0

        # Clean types and compute E_d^buf(w) = n_risky / N_d_air
        n_risky_col = f"n_risky_stops_w{col_suffix}_buf"
        n_store_col = f"n_stores_w{col_suffix}_buf"
        idx_col     = f"idx_risky_w{col_suffix}_buf"

        out_home[n_risky_col] = out_home[n_risky_col].fillna(0).astype(int)
        out_home[n_store_col] = out_home[n_store_col].fillna(0).astype(int)

        out_home[idx_col] = np.where(
            out_home["N_d_air"] > 0,
            out_home[n_risky_col] / out_home["N_d_air"],
            0.0
        )

        save_gdf(B3_BUILDING_GPKG, out_home)

    print("Pipeline B complete.")
# --------------------------- SECTION C: AGGREGATION ---------------------------

# Outputs for Section C
C2_HEXAGG_GPKG    = os.path.join(OUT_DIR, "C2_hex_agg_ratio_w125.gpkg")


def run_section_C():
    print("=== SECTION C: Hex aggregation (125 m ratio, weighted by area_m2 × levels) ===")
    if not os.path.exists(A5_BUILDING_GPKG):
        raise RuntimeError("Section C requires A5 (building-level exposure). Run Pipeline A first.")
    if os.path.exists(C2_HEXAGG_GPKG):
        print("C2: loaded from disk.")
        return

    # --- homes layer with IDs consistent with Pipeline A ---
    homes_ll = read_ll(IN_HOMES)
    if "unit_id" not in homes_ll.columns:
        raise RuntimeError("IN_HOMES must contain 'unit_id'.")
    homes_ll = homes_ll.rename(columns={"unit_id": HOME_ID_COL})

    # --- building-level network exposure (A5) ---
    a5 = gpd.read_file(A5_BUILDING_GPKG)
    needed_cols = ["idx_risky_w50", "idx_risky_w125", "idx_risky_w200", "N_d"]
    for c in needed_cols:
        if c not in a5.columns:
            raise RuntimeError(f"A5 must contain '{c}'.")

    # only buildings with at least one accessible stop (N_d > 0)
    keep = a5[a5["N_d"].fillna(0) > 0][
        [HOME_ID_COL, "geometry", "N_d", "idx_risky_w50", "idx_risky_w125", "idx_risky_w200"]
    ].copy()

    # join back area & floors from original homes layer (no geometry collision)
    base_attrs = homes_ll.drop(columns=["geometry"]).copy()
    keep = keep.merge(base_attrs, on=HOME_ID_COL, how="left")

    # weights = area_m2 × floors
    keep["weight_af"] = (_infer_area_series(keep) * _infer_floors_series(keep)).astype(float)
    keep = keep[keep["weight_af"] > 0]

    # hex grid (~500 m across flats)
    hexgrid = _ensure_hexgrid(homes_ll, C1_HEXGRID_GPKG, HEX_EDGE_LEN_M)

    # spatial join homes→hex
    homes_hex = gpd.sjoin(
        gpd.GeoDataFrame(keep, geometry="geometry", crs=CRS_LL),
        hexgrid[["hex_id", "geometry"]],
        how="inner",
        predicate="intersects"
    ).drop(columns=["index_right"])

    if homes_hex.empty:
        raise RuntimeError("Section C: No homes fell into any hexes—check inputs/CRS.")

    # weighted means per hex for all three w:
    # Eg(w) = Σ(E_d(w) * weight_af) / Σ(weight_af)
    exposure_cols = {
        "idx_risky_w50": "ratio_w50_weighted",
        "idx_risky_w125": "ratio_w125_weighted",
        "idx_risky_w200": "ratio_w200_weighted",
    }

    # denominator: Σ(weight_af) per hex
    den = homes_hex.groupby("hex_id")["weight_af"].sum()

    hex_means = []
    for col, out_name in exposure_cols.items():
        num = (homes_hex[col] * homes_hex["weight_af"]).groupby(homes_hex["hex_id"]).sum()
        hex_means.append((num / den).rename(out_name))

    hex_mean_df = pd.concat(hex_means, axis=1).reset_index()


    # merge all three ratios onto the grid
    out = hexgrid.merge(hex_mean_df, on="hex_id", how="left")
    # NaN = no homes with corridors; 0.0 = real weighted mean of 0

    save_gdf(C2_HEXAGG_GPKG, out)

    print("Section C complete.")


# -------------------------- SECTION D: POLICY ANALYSIS ------------------------

# Outputs for Section D
D1_HEXAGG_GPKG   = os.path.join(OUT_DIR, "D1_hex_agg_ratio_w125_liquor_only.gpkg")
D2_HEXAGG_GPKG   = os.path.join(OUT_DIR, "D2_hex_agg_ratio_w125_super_liquor.gpkg")
D3_HEXAGG_GPKG   = os.path.join(OUT_DIR, "D3_hex_agg_ratio_w125_top25_leverage_removed.gpkg")


def _store_type_kept(mid_type_value, policy):
    """
    mid_type: one of {'supermarket','kiosk','petrol','liquor'} (case-insensitive)
    policy:   'liquor_only' (keep only liquor) OR 'super_liquor' (keep supermarkets & liquor)
    """
    if pd.isna(mid_type_value):
        return False
    t = str(mid_type_value).strip().lower()
    if policy == "liquor_only":
        return t == "liquor"
    if policy == "super_liquor":
        return t in {"liquor", "supermarket"}
    return False

def _recompute_ratio_w125_under_policy(homes_ll, a5, triplets_df, policy_name):
    # use A4 triplets for w=125 to decide which (home, stop) pairs are "risky" under policy
    t = triplets_df[triplets_df["w"] == 125.0].copy()
    if t.empty:
        raise RuntimeError("Section D: No 125 m triplets found in A4. Cannot run policy analysis.")

    # Read stores in the SAME way as in Pipeline A (unit_id → store_id)
    stores = read_ll(IN_STORES)
    if "unit_id" in stores.columns and STORE_ID_COL not in stores.columns:
        stores = stores.rename(columns={"unit_id": STORE_ID_COL})

    if STORE_TYPE_COL not in stores.columns:
        raise RuntimeError(f"Section D requires '{STORE_TYPE_COL}' in store layer.")


    # attach store type and filter by policy
    t = t.merge(stores[[STORE_ID_COL, STORE_TYPE_COL]], on=STORE_ID_COL, how="left")
    t = t[t[STORE_TYPE_COL].apply(lambda v: _store_type_kept(v, policy_name))]

    # risky if any kept store passes for that (home, stop)
    if t.empty:
        risky_pairs = pd.DataFrame(columns=[HOME_ID_COL, STOP_ID_COL, "risky"])
    else:
        risky_pairs = (t.groupby([HOME_ID_COL, STOP_ID_COL]).size()
                         .rename("risky").reset_index())
        risky_pairs["risky"] = 1

    # counts per home under policy
    n_risky = (risky_pairs.groupby(HOME_ID_COL)["risky"].sum()
                          .rename("n_risky_policy").reset_index())

    # bring base N_d (reachable stops) from A5 and compute policy ratio
    # Base homes with at least one reachable stop (same as Section C)
    base = a5[[HOME_ID_COL, "N_d", "geometry"]].copy()
    base = base[base["N_d"].fillna(0) > 0]

    # Attach n_risky (0 if missing)
    base = base.merge(n_risky, on=HOME_ID_COL, how="left")
    base["n_risky_policy"] = base["n_risky_policy"].fillna(0)

    # Policy exposure E_d^policy = n_risky / N_d
    base["idx_risky_w125_policy"] = base["n_risky_policy"] / base["N_d"]

    # Attach area×floors weights (same logic as Section C)
    attrs = homes_ll.drop(columns=["geometry"]).copy()
    base = base.merge(attrs, on=HOME_ID_COL, how="left")
    base["weight_af"] = (_infer_area_series(base) * _infer_floors_series(base)).astype(float)
    base = base[base["weight_af"] > 0]

    return gpd.GeoDataFrame(base, geometry="geometry", crs=CRS_LL)


def _recompute_ratio_w125_under_leverage_cut(homes_ll, a5, triplets_df, lev_gdf, q=0.75):
    """
    Recompute idx_risky_w125 when we remove the top (1-q) fraction of stores
    by leverage (i.e. q=0.75 => remove worst 25% by leverage).

    Uses:
      - A4 triplets (w = 125) to know which (home, stop, store) were risky.
      - A6 leverage layer for Ls_net_delta_250.
    """
    # Only 125 m corridors
    t = triplets_df[triplets_df["w"] == 125.0].copy()
    if t.empty:
        raise RuntimeError("Section D: No 125 m triplets found in A4. Cannot run leverage-cut policy.")

    # Make sure IDs are comparable as strings
    t[STORE_ID_COL] = t[STORE_ID_COL].astype(str)

    # Leverage column from A6
    lev_col = f"Ls_net_delta_{LEVERAGE_DELTA}"
    if lev_col not in lev_gdf.columns:
        raise RuntimeError(f"A6 leverage layer must contain '{lev_col}'.")

    # Keep only stores with a valid leverage value
    lev = lev_gdf[[STORE_ID_COL, lev_col]].copy()
    lev[STORE_ID_COL] = lev[STORE_ID_COL].astype(str)
    lev = lev.dropna(subset=[lev_col])

    if lev.empty:
        raise RuntimeError("Leverage layer has no non-missing values.")

    # Quantile threshold: keep bottom q, remove top (1-q)
    cutoff = lev[lev_col].quantile(q)

    # We KEEP stores strictly below cutoff → top (1-q) fraction is removed
    keep_ids = set(lev.loc[lev[lev_col] < cutoff, STORE_ID_COL])

    # Filter A4 triplets to only those stores we keep
    t = t[t[STORE_ID_COL].isin(keep_ids)]

    # If everything got removed, ratios will all be zero
    if t.empty:
        risky_pairs = pd.DataFrame(columns=[HOME_ID_COL, STOP_ID_COL, "risky"])
    else:
        # A (home, stop) pair is risky if any kept store passes for it
        risky_pairs = (t.groupby([HOME_ID_COL, STOP_ID_COL])
                         .size()
                         .rename("risky")
                         .reset_index())
        risky_pairs["risky"] = 1

    # Counts per home under this leverage-cut policy
    n_risky = (risky_pairs.groupby(HOME_ID_COL)["risky"].sum()
                          .rename("n_risky_policy")
                          .reset_index())

    # Base N_d and geometry from A5
    base = a5[[HOME_ID_COL, "N_d", "geometry"]].copy()
    out = base.merge(n_risky, on=HOME_ID_COL, how="left").fillna({"n_risky_policy": 0})
    out = out[out["N_d"].fillna(0) > 0].copy()  # homes with at least one accessible stop

    out["idx_risky_w125_policy"] = np.where(
        out["N_d"] > 0,
        out["n_risky_policy"] / out["N_d"],
        0.0
    )

    # Attach weights (area × floors) from homes layer
    attrs = homes_ll.drop(columns=["geometry"]).copy()
    out = out.merge(attrs, on=HOME_ID_COL, how="left")
    out["weight_af"] = (_infer_area_series(out) * _infer_floors_series(out)).astype(float)
    out = out[out["weight_af"] > 0]

    return gpd.GeoDataFrame(out, geometry="geometry", crs=CRS_LL)

def run_section_D():
    print("=== SECTION D: Policy analysis on hexes (125 m) ===")
    if not (os.path.exists(A5_BUILDING_GPKG) and os.path.exists(A4_TRIPLETS_CSV)):
        print("Section D skipped: requires A5 and A4 outputs.")
        return

    # baseline hex exposures (from Section C)
    if not os.path.exists(C2_HEXAGG_GPKG):
        raise RuntimeError("Section D with deltas requires C2 (baseline hex exposure).")
    baseline_hex = gpd.read_file(C2_HEXAGG_GPKG)[["hex_id", "ratio_w125_weighted"]]

    # homes layer consistent with A5 / Section C
    homes_ll = read_ll(IN_HOMES)
    if "unit_id" in homes_ll.columns and HOME_ID_COL not in homes_ll.columns:
        homes_ll = homes_ll.rename(columns={"unit_id": HOME_ID_COL})

    a5 = gpd.read_file(A5_BUILDING_GPKG)
    triplets_df = pd.read_csv(A4_TRIPLETS_CSV)

    # same hexgrid as Section C
    hexgrid = _ensure_hexgrid(homes_ll, C1_HEXGRID_GPKG, HEX_EDGE_LEN_M)

    # --- D1: Liquor only ---
    if os.path.exists(D1_HEXAGG_GPKG):
        print("D1: loaded from disk.")
    else:
        homes_policy = _recompute_ratio_w125_under_policy(homes_ll, a5, triplets_df, "liquor_only")

        hp = gpd.sjoin(
            homes_policy,
            hexgrid[["hex_id", "geometry"]],
            how="inner",
            predicate="intersects"
        ).drop(columns=["index_right"])

        if hp.empty:
            raise RuntimeError("D1: No homes intersect any hexes after policy filtering.")

        num = (hp["idx_risky_w125_policy"] * hp["weight_af"]).groupby(hp["hex_id"]).sum()
        den = hp["weight_af"].groupby(hp["hex_id"]).sum()

        hex_mean = (num / den).rename("ratio_w125_liquor_only").reset_index()

        # merge with grid and baseline, then compute delta
        out = hexgrid.merge(hex_mean, on="hex_id", how="left")
        out = out.merge(baseline_hex, on="hex_id", how="left")
        out["delta_ratio_w125_liquor_only"] = (
            out["ratio_w125_liquor_only"] - out["ratio_w125_weighted"]
        )

        save_gdf(D1_HEXAGG_GPKG, out)
        print("D1 complete.")

    # --- D2: Supermarkets + Liquor (no petrol, no kiosk) ---
    if os.path.exists(D2_HEXAGG_GPKG):
        print("D2: loaded from disk.")
    else:
        homes_policy = _recompute_ratio_w125_under_policy(homes_ll, a5, triplets_df, "super_liquor")

        hp = gpd.sjoin(
            homes_policy,
            hexgrid[["hex_id", "geometry"]],
            how="inner",
            predicate="intersects"
        ).drop(columns=["index_right"])

        if hp.empty:
            raise RuntimeError("D2: No homes intersect any hexes after policy filtering.")

        num = (hp["idx_risky_w125_policy"] * hp["weight_af"]).groupby(hp["hex_id"]).sum()
        den = hp["weight_af"].groupby(hp["hex_id"]).sum()

        hex_mean = (num / den).rename("ratio_w125_super_and_liquor").reset_index()

        out = hexgrid.merge(hex_mean, on="hex_id", how="left")
        out = out.merge(baseline_hex, on="hex_id", how="left")
        out["delta_ratio_w125_super_and_liquor"] = (
            out["ratio_w125_super_and_liquor"] - out["ratio_w125_weighted"]
        )

        save_gdf(D2_HEXAGG_GPKG, out)
        print("D2 complete.")

    # --- D3: Remove top 25% highest-leverage stores (A6) ---
    if os.path.exists(D3_HEXAGG_GPKG):
        print("D3: loaded from disk.")
    else:
        if not os.path.exists(A6_LEVERAGE_GPKG):
            print("D3 skipped: requires A6 store leverage (A6_LEVERAGE_GPKG).")
        else:
            print("D3: Recomputing ratios after removing top 25% highest-leverage stores ...")
            lev_gdf = gpd.read_file(A6_LEVERAGE_GPKG)

            homes_policy = _recompute_ratio_w125_under_leverage_cut(
                homes_ll, a5, triplets_df, lev_gdf, q=0.75
            )

            hp = gpd.sjoin(
                homes_policy,
                hexgrid[["hex_id", "geometry"]],
                how="inner",
                predicate="intersects"
            ).drop(columns=["index_right"])

            if hp.empty:
                raise RuntimeError("D3: No homes intersect any hexes after leverage-cut filtering.")

            num = (hp["idx_risky_w125_policy"] * hp["weight_af"]).groupby(hp["hex_id"]).sum()
            den = hp["weight_af"].groupby(hp["hex_id"]).sum()

            hex_mean = (num / den).rename("ratio_w125_top25_leverage_removed").reset_index()

            out = hexgrid.merge(hex_mean, on="hex_id", how="left")
            out = out.merge(baseline_hex, on="hex_id", how="left")
            out["delta_ratio_w125_top25_leverage_removed"] = (
                out["ratio_w125_top25_leverage_removed"] - out["ratio_w125_weighted"]
            )

            save_gdf(D3_HEXAGG_GPKG, out)
            print("D3 complete.")

    print("Section D complete.")

# --------------------------- SECTION E: DENSITY VS PATH -----------------------

E1_HEX_DENSITY_GPKG = os.path.join(OUT_DIR, "E1_hex_store_density.gpkg")
E_HEX_CORR_CSV     = os.path.join(OUT_DIR, "E2_hex_corr_density_path.csv")

def run_section_E():
    """
    Section E: static store density S_g and its correlation with path-based Eg(w).
    """
    print("=== SECTION E: Hex store density and correlations ===")

    if not os.path.exists(C2_HEXAGG_GPKG):
        raise RuntimeError("Section E requires Section C (hex exposure). Run Section C first.")

    # --- 1. Load hex grid with Eg(w) ---
    hex_ll = gpd.read_file(C2_HEXAGG_GPKG)
    if "hex_id" not in hex_ll.columns:
        raise RuntimeError("Hex grid must contain 'hex_id'.")

    # exposure columns (from Section E)
    exp_cols = [c for c in ["ratio_w50_weighted",
                            "ratio_w125_weighted",
                            "ratio_w200_weighted"] if c in hex_ll.columns]
    if not exp_cols:
        raise RuntimeError("No exposure columns found in hex grid.")

    # compute hex area in km² (in metric CRS)
    hex_m = to_metric(hex_ll)
    hex_m["area_km2"] = hex_m.geometry.area / 1e6

    # --- 2. Load midpoints (stores) and spatially join to hexes ---
    # keep only alcohol-related midpoints: you can refine filter if needed
    mids_ll = read_ll(IN_STORES)[["unit_id", "geometry"]].copy()


    mids_m = to_metric(mids_ll)
    mids_m = gpd.sjoin(
        mids_m,
        hex_m[["hex_id", "geometry"]],
        how="inner",
        predicate="intersects"
    ).drop(columns=["index_right"])

    # count stores per hex: M_g
    Mg = (
        mids_m.groupby("hex_id")["unit_id"]
        .nunique()
        .rename("store_count")
        .reset_index()
    )

    # --- 3. Compute static density S_g = M_g / A_g ---
    hex_density = hex_m.merge(Mg, on="hex_id", how="left")
    hex_density["store_count"] = hex_density["store_count"].fillna(0)
    hex_density["S_g"] = hex_density["store_count"] / hex_density["area_km2"]

    # save density layer (LL CRS for mapping)
    out_ll = hex_density.to_crs(CRS_LL)
    save_gdf(E1_HEX_DENSITY_GPKG, out_ll)

    print("Section D complete")

# --------------------------- MAIN --------------------------------------------

if __name__ == "__main__":
    run_pipeline_A()
    run_pipeline_B()
    run_section_C()
    run_section_D()
    run_section_E()
    print("\nAll done. Outputs in:\n", OUT_DIR)
