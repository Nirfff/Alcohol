# Mannheim alcohol exposure pipeline

This project is split into **two Python files**:

- `input_layers.py` = **downloads + preprocessing** (creates the standardized GeoPackages you’ll use everywhere else)
- `metrics.py` = **analysis + outputs** (reads those GeoPackages and computes exposure/leverage/hex aggregations)

The intent is: **File 1 produces clean, consistent inputs; File 2 consumes them and produces results.**


## 1) `input_layers.py` (inputs → standardized layers)

### What it does
Creates the “ready-to-use” layers needed by the metrics pipeline, with **consistent IDs** and **network node linkage**.

### Outputs you should end up with (typical)
All in a “base folder” (or whatever you configure), commonly as `.gpkg`:

- `mannheim_residential_buildings_noded.gpkg`
  - building/household points (or centroids)
  - must contain: `unit_id`, `node_id`, `geometry`
  - optional: `area_m2`, `building:levels`
- `mannheim_midpoints_nodes.gpkg`
  - alcohol outlet midpoints
  - must contain: `unit_id`, `node_id`, `mid_type`, `geometry`
- `mannheim_transit_origins_nodes.gpkg`
  - transit stop/origin points
  - must contain: `unit_id`, `node_id`, `geometry`
- `mannheim_walk_nodes_cache.gpkg`
  - walk network nodes
  - must contain: `osmid`, `geometry`
- `mannheim_walk_edges.gpkg`
  - walk network edges
  - must contain: `u`, `v`, and either `length` or geometry (length can be derived)

### Key contract for file 2
Every feature layer must have:
- `unit_id` (stable unique identifier)
- `node_id` (walk-network node id)
- CRS in **WGS84 (EPSG:4326)** (metrics converts to EPSG:25832 when measuring meters)


## 2) `metrics.py` (standardized layers → exposure metrics)

### What it does
Computes **Transit → Home alcohol exposure** at building level, plus hex aggregations and policy scenarios.

### Required inputs (paths)
Configured at the top:

- `IN_HOMES = ...mannheim_residential_buildings_noded.gpkg`
- `IN_STORES = ...mannheim_midpoints_nodes.gpkg`
- `IN_STOPS = ...mannheim_transit_origins_nodes.gpkg`
- `IN_WALK_NODES = ...mannheim_walk_nodes_cache.gpkg`
- `IN_WALK_EDGES = ...mannheim_walk_edges.gpkg`

### Core assumptions
- Inputs are in **EPSG:4326**
- `node_id` values correspond to nodes in the walk graph
- Distances are computed in **EPSG:25832 (meters)**

### Pipelines inside `metrics.py`

#### Pipeline A: Hybrid network-detour (main)
1. **A1**: home→stop prefilter by **500 m air-distance** (buffer-based join)
2. **A2**: filter to home–stop pairs with **≤ 500 m on the walk network**, and save network paths
3. **A3**: candidate store prefilter using endpoint discs (based on `L + Δ`)
4. **A4**: exact **network detour** test (stop→store→home minus stop→home ≤ Δ)
5. **A5**: building-level exposure outputs
6. **A6**: store leverage at Δ=250, plus **hex-aggregated leverage** using the shared hex grid

#### Pipeline B: Buffer-only baseline (simpler comparator)
- Air prefilter (≤ 500 m) + corridor buffer (w=125 m) + store hits by spatial join

#### Section C: Hex aggregation
- Builds/loads a shared ~500 m hex grid (`C1_hexgrid_500m.gpkg`)
- Aggregates building exposure to hexes using **area × floors** weights

#### Section D: Policy analysis on hexes
Recomputes 125 m exposure under:
- liquor-only
- supermarket + liquor
- removing top 25% stores by leverage

#### Section E: Store density vs path exposure
Computes static store density per hex and saves a layer for comparison.

### Outputs (written to `OUT_DIR`)
You’ll see a set of GPKGs/CSVs like:
- `A5_building_exposure_network.gpkg`
- `B3_building_exposure_buffer.gpkg`
- `C1_hexgrid_500m.gpkg`
- `C2_hex_agg_ratio_w125.gpkg`
- `A6_store_leverage_net_delta_250.gpkg`
- `A6_store_leverage_hex.gpkg`
- `D1/D2/D3_*.gpkg`
- `E1_hex_store_density.gpkg`

### Parameters (as implemented)
- stop air cutoff: **500 m**
- stop–home network cutoff: **500 m**
- corridor widths `w`: **50, 125, 200 m**
- detours `Δ`: **100, 250, 400 m** (= 2*w)
- leverage computed for **Δ = 250 m** (w=125)


## Running order

1) Run `input_layers.py` to create/refresh the standardized GeoPackages.
2) Run `metrics.py` to compute exposure + aggregations:
- Pipeline A → Pipeline B → Section C → Section D → Section E

That’s it: **File 1 prepares the layers; File 2 produces the metrics.**
