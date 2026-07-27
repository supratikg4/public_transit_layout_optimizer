"""
real_city_optimizer.py

CODE PROVIDED BY CLAUDE SONNET 4.6
Prompt: "I want to migrate my program to use real data from Open Street Map and U.S. Census Bureau data.
        Please take my program and create a new program that keeps my exact logic and architecture the same, 
        but uses OSMnx data for the city input instead of my toy inputs. Ensure no logic in the genetic algorithm
        and helper functions are altered, unless required for data formatting purposes when migrating to OSMnx.
Alterations were made by myself to fix bugs and implement changes after generation.

Transit station placement GA using real OSMnx street + Census/OSM density data.

The genetic algorithm, fitness function, and all penalty/commute logic are
identical to the toy grid version. Only the data-loading and precomputation
layers are replaced.

Dependencies:
    pip install osmnx networkx numpy geopandas shapely matplotlib requests

Usage:
    python real_city_optimizer.py
    >>> Place name: Downtown Berkeley, California, USA

Coordinate system: all distances are in metres; all times are in seconds.
Speed constants below are calibrated to match (metres/second → seconds per metre).
"""

import math
import random
import warnings
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import networkx as nx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import osmnx as ox

import tempfile
# Create temp directory
ox.settings.cache_folder = tempfile.gettempdir()
ox.settings.use_cache = False

warnings.filterwarnings("ignore")
multiprocessing.set_start_method("fork")

# ─── PLACE CONFIG ─────────────────────────────────────────────────────────────
DEFAULT_PLACE   = "Piedmont, California, USA"
DEFAULT_SIZE = 2000

# Hard caps — increase for accuracy, decrease for speed
MAX_CANDIDATES  = 60    # candidate station nodes (R tiles → real intersections)
MAX_ORIGINS     = 50    # residential demand nodes
MAX_DESTS       = 30    # commercial / POI destination nodes

# ─── SOFT CONSTRAINTS ─────────────────────────────────────────────────────────
BUS_COST            = 200_000   # $ per bus stop
METRO_COST          = 500_000   # $ per metro station
BUDGET              = 10_000_000
BUS_MIN_DISTANCE    = 400       # metres — same-mode minimum separation
METRO_MIN_DISTANCE  = 800
TRANSFER_DISTANCE   = 300       # metres — bus must be ≤ this from each metro

# ─── TRAVEL MODEL ─────────────────────────────────────────────────────────────
# All times in SECONDS, distances in METRES.
TRAFFIC_LEVELS   = ["Low", "Normal", "Rush"]

# Traffic multipliers applied to OSMnx travel_time edge attribute
RUSH_MULT = {
    "motorway": 1.4,  "trunk": 1.4,
    "motorway_link": 1.4, "trunk_link": 1.4,
    "primary": 2.0,   "primary_link": 2.0,
    "secondary": 2.5, "secondary_link": 2.5,
    "tertiary": 3.0,  "residential": 3.2,
    "living_street": 3.5, "unclassified": 3.0,
}
RUSH_DEFAULT = 3.0
LOW_MULT     = 0.8

WALK_SPEED_MS   = 1.2    # m/s — pedestrian speed
BUS_SPEED_MS    = 5.0    # m/s — average bus speed including stops (~18 km/h)
METRO_SPEED_MS  = 17.0   # m/s — metro speed including stops (~43 km/h)
BUS_WAIT        = 120    # seconds — average wait at bus stop
METRO_WAIT      = 45     # seconds — average wait at metro station
TRANSFER_PENALTY= 75     # seconds — penalty for bus↔metro transfer

BUS   = 1
METRO = 2

# Motorway-class tags: nodes whose ALL edges are these are excluded as candidates
# (equivalent to 'H'-only nodes in toy version which couldn't host stations)
MOTORWAY_TAGS = {"motorway", "trunk", "motorway_link", "trunk_link"}

# ─── GA HYPERPARAMETERS ───────────────────────────────────────────────────────
POP_SIZE               = 200
W_BUDGET               = 30
W_MINDIST              = 100
W_TRANSFER             = 400
W_COMMUTE              = 0.001   # scaled for real travel times (seconds)
INIT_MUTATION_RATE     = 0.05
MUTATION_MIN           = 0.01
MUTATION_MAX           = 0.12
NUM_PARENTS            = 2
K_INIT                 = 4
K_MIN                  = 3
K_MAX                  = 7
ELITE_RATE             = 0.03
CONVERGENCE_THRESHOLD  = 0.05
MIN_GENERATIONS        = 50
CONVERGENCE_GENERATIONS= 20
DIVERSITY_THRESH       = 7.5


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (replaces cityStringTo2DArray / createGraph)
# ═══════════════════════════════════════════════════════════════════════════════

def load_network(place: str, size: float) -> nx.MultiDiGraph:
    """
    Downloads a network. If the input is a coordinate string like "35.779,-78.638", 
    it fetches a 2000m radius. Otherwise, it attempts a standard place name fetch.
    """
    print(f"[1/4] Downloading drive network for '{place}'…")
    
    # Check if the user entered coordinates (e.g., "35.779, -78.638")
    if "," in place and any(char.isdigit() for char in place.split(",")[0]):
        # Parse the coordinates
        lat, lng = map(float, place.split(","))
        # Fetch a 2000-meter radius around the point (Downtown Raleigh)
        G = ox.graph_from_point((lat, lng), dist=size, network_type="drive", simplify=True)
    else:
        # Fall back to the standard Nominatim string method
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        
    G = ox.project_graph(G)         # project to UTM so coords are in metres
    G = ox.add_edge_speeds(G)       # impute missing speed limits
    G = ox.add_edge_travel_times(G) # add travel_time (seconds) to every edge
    print(f"      {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")
    return G

def get_node_coords(G: nx.MultiDiGraph) -> dict:
    """Return {node_id: (x_m, y_m)} for every node."""
    return {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

def snap_features_to_nodes(
    place: str, size: float, G: nx.MultiDiGraph, node_coords: dict
) -> tuple[list, list]:
    """
    Fetch OSM building footprints, snap each to its nearest network node,
    and aggregate demand counts.

    Returns
    -------
    origins      [(node_id, demand_weight), …]  — residential nodes
    destinations [(node_id, demand_weight), …]  — commercial / POI nodes
    """
    print("[2/4] Fetching land-use / POI data …")
    crs = G.graph["crs"]

    def _fetch(tags):
        lat, lng = map(float, place.split(","))
        try:
            #gdf = ox.features_from_place(place, tags=tags)
            gdf = ox.features_from_point((lat, lng), dist=size, tags=tags)
            gdf = gdf[gdf.geometry.geom_type.isin(
                ["Point", "Polygon", "MultiPolygon"])].copy()
            gdf["centroid"] = gdf.geometry.centroid
            return gdf.set_geometry("centroid").to_crs(crs)
        except Exception as exc:
            print(f"      ⚠  OSM fetch failed ({exc})")
            return gpd.GeoDataFrame()

    # Fetch residential and commercial in parallel using threads
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_res = ex.submit(_fetch, {
            "building": ["residential", "apartments", "house", "detached", "yes"]
        })
        f_com = ex.submit(_fetch, {
            "building": ["commercial", "office", "retail", "supermarket"],
            "amenity":  ["school", "university", "hospital", "library", "workplace"],
        })
        res_gdf = f_res.result()
        com_gdf = f_com.result()

    # Snap to nearest node and accumulate demand
    def _snap(gdf, demand_dict):
        if gdf.empty:
            return
        for _, row in gdf.iterrows():
            node = ox.nearest_nodes(G, row.centroid.x, row.centroid.y)
            demand_dict[node] += 1

    origin_demand: dict = defaultdict(int)
    dest_demand:   dict = defaultdict(int)
    _snap(res_gdf, origin_demand)
    _snap(com_gdf, dest_demand)

    # Fallback: if OSM data is too sparse, use high-degree intersections
    if len(origin_demand) < 5:
        print("      Sparse residential data — using high-degree nodes as proxy origins")
        for n in sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:MAX_ORIGINS]:
            origin_demand[n] = max(1, origin_demand[n])

    if len(dest_demand) < 3:
        print("      Sparse commercial data — sampling nodes as proxy destinations")
        for n in random.sample(list(G.nodes()), min(MAX_DESTS, G.number_of_nodes())):
            dest_demand[n] = max(1, dest_demand[n])

    origins      = sorted(origin_demand.items(), key=lambda x: x[1], reverse=True)[:MAX_ORIGINS]
    destinations = sorted(dest_demand.items(),   key=lambda x: x[1], reverse=True)[:MAX_DESTS]

    # Remove any node that is both origin and destination
    dest_set = {d for d, _ in destinations}
    origins  = [(o, w) for o, w in origins if o not in dest_set]

    print(f"      Origins: {len(origins)}  Destinations: {len(destinations)}")
    return origins, destinations

def select_candidate_stations(G: nx.MultiDiGraph, node_coords: dict) -> list:
    """
    Candidate stations = intersections with ≥1 non-motorway edge.
    Equivalent to 'R' tiles in the toy version (R = local road, H = highway).
    Spatially sub-sampled to MAX_CANDIDATES.
    """
    print("[3/4] Selecting candidate stations …")
    candidates = []
    for node in G.nodes():
        edge_hws: set = set()
        for _, _, edata in G.edges(node, data=True):
            hw = edata.get("highway", "")
            if isinstance(hw, list):
                edge_hws.update(hw)
            else:
                edge_hws.add(hw)
        if edge_hws and not edge_hws.issubset(MOTORWAY_TAGS):
            candidates.append(node)

    # Uniform spatial sub-sample — keeps stations spread across the city
    if len(candidates) > MAX_CANDIDATES:
        step       = len(candidates) // MAX_CANDIDATES
        candidates = candidates[::step][:MAX_CANDIDATES]

    print(f"      Candidate stations: {len(candidates)}")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTATION  (same pipeline as toy version, real units)
# ═══════════════════════════════════════════════════════════════════════════════

def build_OD_pairs(origins: list, destinations: list) -> list:
    """
    Cross-join origins × destinations into (o_node, d_node, demand).
    Demand = origin_weight + dest_weight, mirroring toy's res + commercial sum.
    """
    return [
        (o, d, o_w + d_w)
        for (o, o_w) in origins
        for (d, d_w) in destinations
        if o != d
    ]

def spatialDistance(candidate_stations: list, node_coords: dict) -> np.ndarray:
    """
    Vectorised Euclidean distance matrix in metres between candidate stations.
    Replaces the toy's grid-cell Euclidean loop; uses numpy broadcasting (N,2)
    instead of a double Python loop — ~100× faster for large N.
    """
    coords = np.array([node_coords[s] for s in candidate_stations])  # (N, 2)
    diff   = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]     # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=2))                           # (N, N)

def transitDistance(
    G: nx.MultiDiGraph, candidate_stations: list
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two distance matrices — identical semantics to the toy version:

    bus_distance   Shortest-path travel time (seconds) along the road network
                   with a 0.5× speed bonus representing dedicated bus lanes /
                   signal priority.  Buses MUST follow roads.

    metro_distance Euclidean straight-line distance (metres) between stations,
                   representing metros that travel independently of roads.
                   Converted to time inside commuteTime via METRO_SPEED_MS.

    Speedup: one Dijkstra per station (not N²); symmetry used to fill both
    halves simultaneously.
    """
    def bus_weight(u, v, data):
        # OSMnx MultiDiGraph edge — data is a single edge's attribute dict
        return data.get("travel_time", 60) * 0.5

    N            = len(candidate_stations)
    bus_dist     = np.full((N, N), np.inf)
    metro_dist   = np.zeros((N, N))

    # Bus: road-network Dijkstra
    for i, station in enumerate(candidate_stations):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, station, weight=bus_weight
            )
        except nx.NetworkXError:
            lengths = {}
        for j in range(i, N):
            d = lengths.get(candidate_stations[j], np.inf)
            bus_dist[i][j] = d
            bus_dist[j][i] = d  # symmetric

    # Metro: fully vectorised Euclidean (numpy broadcasting, same as spatialDistance)
    coords     = np.array([node_coords[s] for s in candidate_stations])
    diff       = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    metro_dist = np.sqrt((diff ** 2).sum(axis=2))

    return bus_dist, metro_dist

def originDestinationTimes(
    G: nx.MultiDiGraph, OD_pairs: list
) -> dict:
    """
    Baseline car/walk travel time (seconds) per traffic level for each OD pair.
    Uses OSMnx travel_time edge attribute with per-highway-type traffic multipliers.

    Speedup: run only one Dijkstra per unique origin per traffic level (not one
    per OD pair), and batch-fill the baseline dict from those results.
    """
    print("      Computing baseline OD travel times …")

    def make_weight_fn(traffic_level: str):
        def weight_fn(u, v, data):
            base = data.get("travel_time", 60)
            if traffic_level == "Normal":
                return base
            if traffic_level == "Low":
                return base * LOW_MULT
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            return base * RUSH_MULT.get(hw, RUSH_DEFAULT)
        return weight_fn

    baseline        = {t: {} for t in TRAFFIC_LEVELS}
    unique_origins  = list({o for o, _, _ in OD_pairs})
    dest_set        = {d for _, d, _ in OD_pairs}

    for tl in TRAFFIC_LEVELS:
        wfn = make_weight_fn(tl)
        for origin in unique_origins:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, origin, weight=wfn)
            except nx.NetworkXError:
                lengths = {}
            for d in dest_set:
                if d in lengths:
                    baseline[tl][(origin, d)] = lengths[d]

    return baseline

def originToStationDistances(
    candidate_stations: list, node_coords: dict,
    origins: list, destinations: list
) -> tuple[dict, dict]:
    """
    Walking time (seconds) from each origin/destination to every candidate
    station, using straight-line distance ÷ WALK_SPEED_MS.

    This is a deliberate simplification matching the toy version's WALK_WEIGHT ×
    Euclidean heuristic (pedestrians cut across blocks, not just car roads).

    Speedup: fully vectorised numpy per origin/destination — no Python loops
    over stations.
    """
    station_coords = np.array([node_coords[s] for s in candidate_stations])  # (N, 2)

    origin_to_station: dict = {}
    for (o, _) in origins:
        o_xy  = np.array(node_coords[o])
        dists = np.linalg.norm(station_coords - o_xy, axis=1) / WALK_SPEED_MS
        origin_to_station[o] = {candidate_stations[i]: float(dists[i])
                                 for i in range(len(candidate_stations))}

    station_to_dest: dict = {}
    for (d, _) in destinations:
        d_xy  = np.array(node_coords[d])
        dists = np.linalg.norm(station_coords - d_xy, axis=1) / WALK_SPEED_MS
        station_to_dest[d] = {candidate_stations[i]: float(dists[i])
                               for i in range(len(candidate_stations))}

    return origin_to_station, station_to_dest

def precompute(G, origins, destinations, candidate_stations, node_coords):
    """Orchestrates all precomputation. Returns the same 7-tuple as toy version."""
    print("[4/4] Precomputing distance matrices …")
    OD_pairs                       = build_OD_pairs(origins, destinations)
    spatial_dist                   = spatialDistance(candidate_stations, node_coords)
    bus_dist, metro_dist           = transitDistance(G, candidate_stations)
    base_time                      = originDestinationTimes(G, OD_pairs)
    origin_to_station, station_to_dest = originToStationDistances(
        candidate_stations, node_coords, origins, destinations
    )
    print(f"      OD pairs: {len(OD_pairs):,}")
    return spatial_dist, base_time, bus_dist, metro_dist, OD_pairs, origin_to_station, station_to_dest


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION  (identical logic to toy version)
# ═══════════════════════════════════════════════════════════════════════════════

def commuteTime(layout, base_time, bus_distance, metro_distance,
                OD_pairs, candidate_stations, origin_to_station, station_to_dest):
    """
    Total weighted commute time (seconds) across all OD pairs and traffic levels.

    Transit leg times are computed exactly as in the toy version:
      bus  → bus   road-network distance / BUS_SPEED_MS  + BUS_WAIT
      metro→ metro Euclidean distance    / METRO_SPEED_MS + METRO_WAIT
      mixed        Euclidean             / METRO_SPEED_MS + both waits + TRANSFER_PENALTY
                   (metro leg is the enabling leg; bus provides the feeder)

    All units are seconds, consistent with base_time from OSMnx travel_time.
    """
    active_indices = [i for i, v in enumerate(layout) if v != 0]

    if not active_indices:
        return sum(
            demand * base_time[tl].get((o, d), np.inf)
            for tl in TRAFFIC_LEVELS for o, d, demand in OD_pairs
        )

    active_nodes = [candidate_stations[i] for i in active_indices]
    active_modes = np.array([layout[i] for i in active_indices])

    # Mode masks
    modes_i     = active_modes[:, None]
    modes_j     = active_modes[None, :]
    bus_bus     = (modes_i == BUS)   & (modes_j == BUS)
    metro_metro = (modes_i == METRO) & (modes_j == METRO)
    mixed       = ~bus_bus & ~metro_metro

    # Slice distance matrices to active stations only
    idx = np.ix_(active_indices, active_indices)
    B   = bus_distance[idx]     # road-network seconds (bus legs)
    M   = metro_distance[idx]   # Euclidean metres     (metro legs)

    # Convert distances → travel times
    T_time = np.full_like(B, np.inf)
    T_time[bus_bus]     = B[bus_bus]     / BUS_SPEED_MS   + BUS_WAIT
    T_time[metro_metro] = M[metro_metro] / METRO_SPEED_MS + METRO_WAIT
    T_time[mixed]       = (M[mixed] / METRO_SPEED_MS
                           + BUS_WAIT + METRO_WAIT + TRANSFER_PENALTY)
    T_time[B == np.inf] = np.inf

    # Walk-time arrays (seconds) to/from each active station
    unique_origins = {o for o, _, _ in OD_pairs}
    unique_dests   = {d for _, d, _ in OD_pairs}

    walk_o = {o: np.array([origin_to_station[o][s] for s in active_nodes])
              for o in unique_origins if o in origin_to_station}
    walk_d = {d: np.array([station_to_dest[d][s]   for s in active_nodes])
              for d in unique_dests   if d in station_to_dest}

    # cost_to_d[d][i] = cheapest time from entry-station i onward to destination d
    cost_to_d = {d: np.min(T_time + walk_d[d], axis=1) for d in unique_dests if d in walk_d}

    commute_time = 0.0
    for (o, d, demand) in OD_pairs:
        if o not in walk_o or d not in cost_to_d:
            continue
        best_transit = float(np.min(walk_o[o] + cost_to_d[d]))
        for tl in TRAFFIC_LEVELS:
            base = base_time[tl].get((o, d), np.inf)
            commute_time += demand * (best_transit if best_transit < base else base)

    return commute_time

def penaltyBudget(layout) -> float:
    """ total = sum(
        BUS_COST if v == BUS else METRO_COST if v == METRO else 0
        for v in layout
    ) """
    tot_cost = 0
    for station in layout:
        if station == BUS:
            tot_cost += BUS_COST
        elif station == METRO:
            tot_cost += METRO_COST
    return max(0, tot_cost - BUDGET)
    #return max(0.0, total - BUDGET)

def penaltyMinDistance(layout, spatial_distance) -> int:
    layout     = np.array(layout)
    bus_mask   = (layout == BUS)
    metro_mask = (layout == METRO)
    v  = int(np.sum((spatial_distance < BUS_MIN_DISTANCE)   & (bus_mask[:,None]   & bus_mask[None,:])))
    v += int(np.sum((spatial_distance < METRO_MIN_DISTANCE) & (metro_mask[:,None] & metro_mask[None,:])))
    return v // 2

def penaltyTransfer(layout, spatial_distance) -> int:
    layout = np.array(layout)
    metros = (layout == METRO)
    buses  = (layout == BUS)
    if not np.any(metros): return 0
    if not np.any(buses):  return int(np.sum(metros))
    transfers = spatial_distance[metros][:, buses]
    return int(np.sum(~np.any(transfers <= TRANSFER_DISTANCE, axis=1)))

def fitness(layout, spatial_distance, base_time, bus_distance, metro_distance,
            OD_pairs, candidate_stations, origin_to_station, station_to_dest) -> float:
    return (
          W_BUDGET   * penaltyBudget(layout)
        + W_MINDIST  * penaltyMinDistance(layout, spatial_distance)
        + W_TRANSFER * penaltyTransfer(layout, spatial_distance)
        + W_COMMUTE  * commuteTime(layout, base_time, bus_distance, metro_distance,
                                   OD_pairs, candidate_stations,
                                   origin_to_station, station_to_dest)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM OPERATORS  (identical to toy version)
# ═══════════════════════════════════════════════════════════════════════════════

def tournament_selection(population: list, k: int) -> tuple:
    parents = []
    for _ in range(NUM_PARENTS):
        members = [population[i] for i in random.sample(range(len(population)), k)]
        parents.append(min(members, key=lambda x: x[1]))
    return parents[0][0], parents[1][0]

def spatial_crossover(parent_1, parent_2, candidate_stations: list,
                      node_coords: dict, mutation_probability: float) -> np.ndarray:
    """
    Bounding-box spatial crossover adapted for projected UTM coordinates.
    Stations inside a random sub-rectangle inherit from parent_1; others from parent_2.
    Beta(2,2) distribution biases the box toward ~50/50 splits.
    Semantically identical to the toy grid version.
    """
    xs = [node_coords[s][0] for s in candidate_stations]
    ys = [node_coords[s][1] for s in candidate_stations]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    box_l = x_min + random.betavariate(2, 2) * x_range
    box_r = x_min + random.betavariate(2, 2) * x_range
    box_b = y_min + random.betavariate(2, 2) * y_range
    box_t = y_min + random.betavariate(2, 2) * y_range
    if box_l > box_r: box_l, box_r = box_r, box_l
    if box_b > box_t: box_b, box_t = box_t, box_b

    child = np.zeros(len(parent_1), dtype=np.int32)
    for i, s in enumerate(candidate_stations):
        x, y     = node_coords[s]
        inside   = box_l <= x <= box_r and box_b <= y <= box_t
        child[i] = parent_1[i] if inside else parent_2[i]
        if random.random() < mutation_probability:
            child[i] = random.randint(0, 2)
    return child

def pct_diff(a: float, b: float) -> float:
    mid = (a + b) / 2
    return 0.0 if mid == 0 else abs(a - b) / mid * 100

def diversity(scores: list) -> float:
    vals = [s[1] for s in scores]
    mean = np.mean(vals)
    return 0.0 if mean == 0 else (np.std(vals) / mean) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def computeEdgeFlows(G, layout, OD_pairs, candidate_stations, node_coords,
                     bus_distance, metro_distance,
                     origin_to_station, station_to_dest, k: int = 3) -> dict:
    stations   = [(i, candidate_stations[i]) for i, v in enumerate(layout) if v != 0]
    edge_flow  = {edge: 0 for edge in G.edges()}
    path_cache = {}

    def get_path(u, v):
        if (u, v) not in path_cache:
            try:    path_cache[(u, v)] = nx.shortest_path(G, u, v, weight="travel_time")
            except: path_cache[(u, v)] = None
        return path_cache[(u, v)]

    if not stations:
        return edge_flow

    for (o, d, demand) in OD_pairs:
        closest_o = sorted(stations, key=lambda x: origin_to_station[o].get(x[1], np.inf))[:k]
        closest_d = sorted(stations, key=lambda x: station_to_dest[d].get(x[1], np.inf))[:k]

        best_time, best_pair = np.inf, None
        for (i, s1) in closest_o:
            w1 = origin_to_station[o].get(s1, np.inf)
            if w1 == np.inf: continue
            for (j, s2) in closest_d:
                w2 = station_to_dest[d].get(s2, np.inf)
                if w2 == np.inf: continue
                mi, mj = layout[i], layout[j]
                if mi == BUS and mj == BUS:
                    t = bus_distance[i][j] / BUS_SPEED_MS + BUS_WAIT
                elif mi == METRO and mj == METRO:
                    t = metro_distance[i][j] / METRO_SPEED_MS + METRO_WAIT
                else:
                    t = metro_distance[i][j] / METRO_SPEED_MS + BUS_WAIT + METRO_WAIT + TRANSFER_PENALTY
                total = w1 + t + w2
                if total < best_time:
                    best_time, best_pair = total, (s1, s2)

        if best_pair is None: continue
        s1, s2 = best_pair
        p1, p2, p3 = get_path(o, s1), get_path(s1, s2), get_path(s2, d)
        if not all([p1, p2, p3]): continue
        full = p1[:-1] + p2[:-1] + p3
        for u, v in zip(full[:-1], full[1:]):
            if   (u, v) in edge_flow: edge_flow[(u, v)] += demand
            elif (v, u) in edge_flow: edge_flow[(v, u)] += demand

    return edge_flow

def computeMetrics(layout, spatial_distance, base_time, bus_distance, metro_distance,
                   OD_pairs, candidate_stations, origin_to_station, station_to_dest) -> dict:
    budget   = penaltyBudget(layout)
    min_dist = penaltyMinDistance(layout, spatial_distance)
    transfer = penaltyTransfer(layout, spatial_distance)
    commute  = commuteTime(layout, base_time, bus_distance, metro_distance,
                           OD_pairs, candidate_stations, origin_to_station, station_to_dest)
    total = (W_BUDGET * budget + W_MINDIST * min_dist +
             W_TRANSFER * transfer + W_COMMUTE * commute)
    return {
        "total":               total,
        "avg_commute_s":       commute / max(len(OD_pairs), 1),
        "budget_violation":    budget,
        "min_dist_violations": min_dist,
        "transfer_violations": transfer,
    }

def drawFlowGraph(G, edge_flow, candidate_stations, layout,
                  node_coords, metrics, place_name: str) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1])
    ax  = fig.add_subplot(gs[0, 0])
    tax = fig.add_subplot(gs[0, 1])
    tax.axis("off")

    # Draw base street map
    ox.plot_graph(G, ax=ax, show=False, close=False,
                  bgcolor="white", node_size=0,
                  edge_color="#e0e0e0", edge_linewidth=0.4)

    # Edge flows
    edges  = list(G.edges())
    pos    = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    flows  = np.array([edge_flow.get(e, 0) for e in edges], dtype=float)
    log_f  = np.log1p(flows)
    norm_f = log_f / log_f.max() if log_f.max() > 0 else log_f
    widths = 0.5 + 5 * norm_f
    mask0  = flows == 0

    G_draw = G.to_undirected()
    nx.draw_networkx_edges(G_draw, pos, edgelist=[e for i,e in enumerate(edges) if mask0[i]],
                           width=0.3, edge_color="gainsboro", alpha=0.4, ax=ax)
    nx.draw_networkx_edges(G_draw, pos, edgelist=[e for i,e in enumerate(edges) if not mask0[i]],
                           width=widths[~mask0], edge_color=norm_f[~mask0],
                           edge_cmap=plt.cm.plasma, ax=ax)

    # Station nodes
    station_idx = {s: i for i, s in enumerate(candidate_stations)}
    bus_xy   = [(node_coords[s][0], node_coords[s][1])
                for s in candidate_stations if layout[station_idx[s]] == BUS]
    metro_xy = [(node_coords[s][0], node_coords[s][1])
                for s in candidate_stations if layout[station_idx[s]] == METRO]

    if bus_xy:
        bx, by = zip(*bus_xy)
        ax.scatter(bx, by, c="dodgerblue", s=120, marker="s",
                   edgecolors="black", linewidths=1, zorder=5, label="Bus stop")
    if metro_xy:
        mx, my = zip(*metro_xy)
        ax.scatter(mx, my, c="darkviolet", s=180, marker="D",
                   edgecolors="black", linewidths=1, zorder=5, label="Metro station")

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=mpl.colors.Normalize(vmin=flows.min(), vmax=flows.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, label="Edge flow (demand·trips)")

    ax.set_title(f"Transit Flow — {place_name}", fontsize=11)

    # Metrics panel
    n_bus   = int(np.sum(np.array(layout) == BUS))
    n_metro = int(np.sum(np.array(layout) == METRO))
    text = (
        "METRICS\n"
        "─────────────────────\n"
        f"Total fitness   : {metrics['total']:.1f}\n"
        f"Avg commute     : {metrics['avg_commute_s']:.0f} s\n"
        f"Budget excess   : ${metrics['budget_violation']:,.0f}\n"
        f"MinDist viol.   : {metrics['min_dist_violations']}\n"
        f"Transfer viol.  : {metrics['transfer_violations']}\n"
        f"Bus stops       : {n_bus}\n"
        f"Metro stations  : {n_metro}"
    )
    tax.text(0.05, 0.75, text, fontsize=10, family="monospace", verticalalignment="top")
    legend_el = [
        mlines.Line2D([],[],marker="s",color="w",markerfacecolor="dodgerblue",
                      markersize=10, markeredgecolor="black", label="Bus stop"),
        mlines.Line2D([],[],marker="D",color="w",markerfacecolor="darkviolet",
                      markersize=10, markeredgecolor="black", label="Metro station"),
    ]
    tax.legend(handles=legend_el, loc="lower left", frameon=False, bbox_to_anchor=(0, 0.1))
    plt.tight_layout()
    plt.show()

def plotConvergence(fitness_history: list) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(fitness_history, linewidth=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (log)")
    plt.yscale("log")
    plt.title("GA Convergence")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Load city data ────────────────────────────────────────────────────
    place = input(f"Point: ").strip() or DEFAULT_PLACE
    size = float(input(f"Size: ").strip() or DEFAULT_SIZE)

    G           = load_network(place, size)
    node_coords = get_node_coords(G)

    origins, destinations  = snap_features_to_nodes(place, size, G, node_coords)
    candidate_stations     = select_candidate_stations(G, node_coords)

    # ── 2. Precompute ────────────────────────────────────────────────────────
    (spatial_distance, base_time, bus_distance, metro_distance,
     OD_pairs, origin_to_station, station_to_dest) = precompute(
        G, origins, destinations, candidate_stations, node_coords
    )

    if not OD_pairs:
        raise RuntimeError(
            "No OD pairs — the area may have too little OSM building data. "
            "Try a larger neighbourhood or lower MAX_ORIGINS / MAX_DESTS."
        )

    # ── 3. Initial population ────────────────────────────────────────────────
    N          = len(candidate_stations)
    population = [np.random.randint(0, 3, size=N, dtype=np.int32) for _ in range(POP_SIZE)]

    fitness_history: list = []
    generation            = 0
    k                     = K_INIT
    mutation_probability  = INIT_MUTATION_RATE

    # ── 4. Multiprocessing worker initialiser ────────────────────────────────
    def init_worker(sd, bt, bd, md, od, cs, ots, std):
        global _sd, _bt, _bd, _md, _od, _cs, _ots, _std
        _sd, _bt, _bd, _md = sd, bt, bd, md
        _od, _cs, _ots, _std = od, cs, ots, std

    def fitness_wrapper(layout):
        return (layout, fitness(layout, _sd, _bt, _bd, _md, _od, _cs, _ots, _std))

    # ── 5. GA loop ───────────────────────────────────────────────────────────
    print(f"\nStarting GA: {POP_SIZE} individuals · {N} candidates · {len(OD_pairs):,} OD pairs\n")

    with Pool(initializer=init_worker,
              initargs=(spatial_distance, base_time, bus_distance, metro_distance,
                        OD_pairs, candidate_stations, origin_to_station, station_to_dest)) as pool:

        while True:
            scores = pool.map(fitness_wrapper, population)
            scores.sort(key=lambda x: x[1])

            # Adaptive diversity control — identical to toy version
            div = diversity(scores)
            if div < DIVERSITY_THRESH:
                k                    = max(K_MIN, k - 1)
                mutation_probability = min(MUTATION_MAX, mutation_probability + 0.02)
            else:
                k                    = min(K_MAX, k + 1)
                mutation_probability = max(MUTATION_MIN, mutation_probability - 0.01)

            # Elitism
            best_layouts   = [x[0] for x in scores[:max(1, int(POP_SIZE * ELITE_RATE))]]
            new_population = best_layouts.copy()

            # Crossover + mutation
            while len(new_population) < POP_SIZE:
                p1, p2 = tournament_selection(scores, k)
                child  = spatial_crossover(p1, p2, candidate_stations,
                                           node_coords, mutation_probability)
                new_population.append(child)

            population = new_population

            best     = scores[0][0]
            best_score = scores[0][1]
            fitness_history.append(best_score)

            # Per-component breakdown
            budget_s   = W_BUDGET   * penaltyBudget(best)
            mindist_s  = W_MINDIST  * penaltyMinDistance(best, spatial_distance)
            transfer_s = W_TRANSFER * penaltyTransfer(best, spatial_distance)
            commute_s  = W_COMMUTE  * commuteTime(best, base_time, bus_distance, metro_distance,
                                                   OD_pairs, candidate_stations,
                                                   origin_to_station, station_to_dest)
            n_bus   = int(np.sum(best == BUS))
            n_metro = int(np.sum(best == METRO))

            print(
                f"Gen {generation:4d} | fit={best_score:12.1f} | "
                f"budget={budget_s:9.0f} | mindist={mindist_s:7.0f} | "
                f"transfer={transfer_s:7.0f} | commute={commute_s:8.1f} | "
                f"bus={n_bus:2d} metro={n_metro:2d} | "
                f"div={div:5.1f}% | k={k} | mut={mutation_probability:.3f}"
            )

            # Convergence check — identical to toy version
            window = fitness_history[-CONVERGENCE_GENERATIONS:]
            if (generation >= MIN_GENERATIONS
                    and len(window) == CONVERGENCE_GENERATIONS
                    and pct_diff(max(window), min(window)) < CONVERGENCE_THRESHOLD):
                print("\nConverged.")
                break

            generation += 1

    # ── 6. Results ───────────────────────────────────────────────────────────
    layout  = scores[0][0]
    metrics = computeMetrics(layout, spatial_distance, base_time, bus_distance,
                             metro_distance, OD_pairs, candidate_stations,
                             origin_to_station, station_to_dest)

    print("\n" + "═" * 60)
    print("  FINAL RESULTS")
    print("═" * 60)
    print(f"  Total fitness score     : {metrics['total']:.1f}")
    print(f"  Avg commute time        : {metrics['avg_commute_s']:.0f} s  "
          f"({metrics['avg_commute_s']/60:.1f} min)")
    print(f"  Budget excess           : ${metrics['budget_violation']:,.0f}")
    print(f"  Min-distance violations : {metrics['min_dist_violations']}")
    print(f"  Transfer violations     : {metrics['transfer_violations']}")
    print(f"  Bus stops placed        : {int(np.sum(layout == BUS))}")
    print(f"  Metro stations placed   : {int(np.sum(layout == METRO))}")
    print("═" * 60)

    plotConvergence(fitness_history)

    edge_flows = computeEdgeFlows(
        G, layout, OD_pairs, candidate_stations, node_coords,
        bus_distance, metro_distance, origin_to_station, station_to_dest
    )
    drawFlowGraph(G, edge_flows, candidate_stations, layout,
                  node_coords, metrics, place)