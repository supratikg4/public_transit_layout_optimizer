import numpy as np
from colorama import Back
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import multiprocessing
from multiprocessing import Pool

# Set multiprocessor default start method to fork for macOS terminal compatibility
multiprocessing.set_start_method('fork')

# CITY INPUTS [city string, width, height]
MANHATTAN = [
    "...HHHHHHHHHHHHHHHHHHHHHH........H.R99R88R7777R88R99R.H........HRRRRRRRRRRRRRRRRRRRRH........H.R99R..R....R..R88R.H........H.R88R..R....R..R77R.H........H.R88R..R....R..R77R.H........H.RRRRRRRRRRRRRRRRRR.H........H.R88R..R....R..R77R.H........H.R77R..R....R..R66R.H........H.R77R..R....R..R66R.H........HRRRRRRRRRRRRRRRRRRRRH........H.R66RaaRbbbbRccR66R.H........H.R66RbbRccccRbbR66R.H........H.R55RccReeeeRccR55R.H........H.RRRRRRRRRRRRRRRRRR.H........H.RccReeReeeeReeRccR.H........H.RddReeReeeeReeRddR.H........H.RccRddReeeeRddRccR.H........HRRRRRRRRRRRRRRRRRRRRH........H.R44RaaRbbbbRaaR44R.H........H.R44RccRccccRccR44R.H........H.R55RbbRaaaaRbbR55R.H........H.RRRRRRRRRRRRRRRRRR.H........H.R66R44R4444R44R66R.H........H.R77R55R5555R55R77R.H........H.R88R66R6666R66R88R.H........HRRRRRRRRRRRRRRRRRRRRH........H.R88R77R7777R77R88R.H........H.R99R88R8888R88R99R.H........H.R99R99R9999R99R99R.H........H.RRRRRRRRRRRRRRRRRRHH........H....RbbRccccRbbR...H.........H....RccRddddRccR...H.........H....RddReeeeRddR...H.........H....RRRRRRRRRRRR...H.........H.......RccddR......H.........H.......RddeeR......H.........H.......RRRRRR......H.........H..........ee.......H.........HHHHHHHHHHHHHHHHHHHHH......",
    30, 40
]
MONOCENTRIC = [
    "111R222R33111R222R33RRRRRRRRRR44RaabR55544RcdeR555RRRRRRRRRR66RbcaR77766RdeeR777RRRRRRRRRR888R999R99",
    10, 10
]
TWIN = [
    "111Raa.H..ccR666..111Raa.H..ccR666..RRRRRRRHRRRRRRRRRR22.Rbb.H..ddR777..33.Rbb.H..ddR777..RRRRRRRHRRRRRRRRRR444R...H..eeR88899555RHHHH..eeR88899",
    18, 8
]
DENSE = [
    "99999Reeeee99999Reeeee99999ReeeeeRRRRRRRRRRR88888Rddddd88888Rddddd88888RdddddRRRRRRRRRRR77777Rccccc77777Rccccc77777Rccccc",
    11, 11
]
LARGE = [
    "11.22R11.22R.....H11.22Raa.bbRaa.bb11.22R11.22R.....H11.22Raa.bbRaa.bb.....R.....R.....H.....R.....R.....33.11R33.11R.....H33.11Rbb.aaRbb.aa33.11R33.11R.....H33.11Rbb.aaRbb.aaRRRRRRRRRRRRRRRRRHRRRRRRRRRRRRRRRRR11.22R44.55R44.55Haa.bbRcc.ccRcc.cc11.22R44.55R44.55Haa.bbRcccccRccccc.....R..6..R..6..H.....R..c..R..c..33.11R55.44R55.44Hbb.aaRcccccRccccc33.11R55.44R55.44Hbb.aaRcc.ccRcc.ccRRRRRRRRRRRRRRRRRHRRRRRRRRRRRRRRRRR44.55R44.55R77889Hcc.ccRdddeeRdddee44.55R44.55R77889HcccccRdddeeRdddee..6..R..6..R88999H..c..ReeeeeReeeee55.44R55.44R99999HcccccReeeedReeeed55.44R55.44R99999Hcc.ccReedddReedddHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH44.55R77889R77889HdddeeReeeeeRdddee44.55R77889R77889HdddeeReeeeeRdddee..6..R88999R88999HeeeeeReeeeeReeeee55.44R99999R99999HeeeedReeeeeReeeed55.44R99999R99999HeedddReeeeeReedddRRRRRRRRRRRRRRRRRHRRRRRRRRRRRRRRRRR.....R77889R77889HdddeeRdddeeRcc.cc.....R77889R77889HdddeeRdddeeRccccc.....R88999R88999HeeeeeReeeeeR..c.......R99999R99999HeeeedReeeedRccccc.....R99999R99999HeedddReedddRcc.ccRRRRRRRRRRRRRRRRRHRRRRRRRRRRRRRRRRR44.55R44.55R44.55H.....Rcc.ccRaa.bb44.55R44.55R44.55H.....RcccccRaa.bb..6..R..6..R..6..H.....R..c..R.....55.44R55.44R55.44H.....RcccccRbb.aa55.44R55.44R55.44H.....Rcc.ccRbb.aa",
    35, 35
]

CITIES = [MONOCENTRIC, TWIN, DENSE, LARGE, MANHATTAN]

# SOFT CONSTRAINTS FOR FITNESS FUNCTION
BUS_COST = 200 # Cost of a bus station
METRO_COST = 500 # Cost of a metro station
BUDGET = 15000 # Total budget for all stations
BUS_MIN_DISTANCE = 2.5 # Minimum distance required between bus stations
METRO_MIN_DISTANCE = 4 # Minimum distance required between metro stations
TRANSFER_DISTANCE = 1.5 # Distance that a bus station must be placed next to a metro station

# TRAFFIC CONSTANTS
TRAFFIC_LEVELS = ["Low", "Normal", "Rush"] # Levels of traffic
BASE_HIGHWAY_WEIGHT = 1 # Base travel weight for highways
BASE_ROAD_WEIGHT = 2 # Base travel weight for roads
WALK_WEIGHT = 5 # Travel weight for walking
BUS = 1 # Value of bus
METRO = 2 # Value of metro
BUS_SPEED = 1.5 # Speed of buses
METRO_SPEED = 3 # Speed of metros
BUS_WAIT = 2 # Wait time for buses
METRO_WAIT = 1 # Wait time for metros
TRANSFER_PENALTY = 1 # Penalty for transferring modes

# GENETIC ALGORITHM HYPERPARAMETERS
POP_SIZE = 200 # Population size
W_BUDGET = 30 # Exceeded budget weight
W_MINDIST = 100 # Minimum distance violation weight
W_TRANSFER = 400 # Transfer violation weight
W_COMMUTE = 0.5 # Commute time weight
INIT_MUTATION_RATE = 0.05 # Initial mutation rate for GA
MUTATION_MIN = 0.01 # Minimum mutation rate for dynamic diversity adjustment
MUTATION_MAX = 0.12 # Maximum mutation rate for dynamic diversity adjustment
NUM_PARENTS = 2 # Number of parents for each child
K_INIT = 4 # Initial number of tournament members in each parent selection
K_MIN = 3 # Minimum k value for dynamic diversity adjustment
K_MAX = 7 # Maximum k value for dynamic diversity adjustment
ELITE_RATE = 0.03 # Elite individuals to copy to next generation
CONVERGENCE_THRESHOLD = 0.05 # Threshold required to converge on a solution
MIN_GENERATIONS = 50 # Minimum number of generations to run before converge possible
CONVERGENCE_GENERATIONS = 20 # Number of generations to consider for convergence determination
DIVERSITY_THRESH = 7.5 # Threshold to determine diversity of population

# CITY REPRESENTATION
def cityStringTo2DArray(city, width, height):
    """Converts input city string into 2D array representation

    Args:
        city (string): input city as string
        width (int): number of columns in city
        height (int): number of rows in city

    Returns:
        ndarray: 2D array of chars, representing city layout
    """
    input_map = []
    index = 0
    for _ in range(height):
        cur_row = []
        for _ in range(width):
            cur_row.append(city[index])
            index += 1
        input_map.append(cur_row)
    return input_map

def showMap(city):
    """Prints city map to terminal

    Args:
        city (string): input city as string
    """
    
    RESET = '\033[0m'
    # Print map
    for i in range(len(city)):
        for j in range(len(city[i])):
            cur_char = (city[i][j])
            if (cur_char >= '1' and cur_char <= '9'):
                print(f"{Back.RED}{cur_char}{RESET}", end="")
            elif (cur_char >= 'a' and cur_char <= 'e'):
                print(f"{Back.GREEN}{cur_char}{RESET}", end="")
            elif (cur_char == 'H'):
                print(f"{Back.LIGHTBLACK_EX}{cur_char}{RESET}", end="")
            elif (cur_char == 'R'):
                print(f"{Back.LIGHTWHITE_EX}{cur_char}{RESET}", end="")
            else:
                print(cur_char, end="")
        print("")

def createGraph(city):
    """Creates a NetworkX graph of the city

    Args:
        city (list): city map

    Returns:
        nx.Graph: NetworkX Graph of city
    """
    height = len(city)
    width = len(city[0])
    
    G = nx.Graph()
    candidate_stations = []
    
    ROAD_TILES = {"R", "H"}
    
    # Add nodes to graph (G)
    for i in range(height):
        for j in range(width):
            tile = city[i][j]
            
            if tile != '.':
                G.add_node((i, j), tile_type=tile) # Add node to graph
            
            if tile == "R":
                candidate_stations.append((i, j))
    
    # Left, right, up, and down directions for edge checks
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # Add edges to graph
    for i in range(height):
        for j in range(width):
            tile = city[i][j]
            
            for di, dj in directions:
                i_next, j_next = i + di, j + dj
                # Boundary check
                if 0 <= i_next < height and 0 <= j_next < width:
                    neighbor = city[i_next][j_next]
                    
                    if tile in ROAD_TILES and neighbor in ROAD_TILES:
                        # ROAD
                        if tile == 'H' and neighbor == 'H':
                            weight = BASE_HIGHWAY_WEIGHT # H-H
                        else:
                            weight = BASE_ROAD_WEIGHT # R-H OR H-R
                        
                        # Add edge to graph
                        G.add_edge((i, j), (i_next, j_next), base_weight=weight, edge_type='road')
                    elif tile != '.' and neighbor != '.' and tile != 'H' and neighbor != 'H':
                        # ACCESS
                        G.add_edge((i, j), (i_next, j_next), base_weight=WALK_WEIGHT, edge_type='access')
                        
    return G, candidate_stations

# PRECOMPUTE MATRICES
def originDestinationTimes(G, city):
    """Computes baseline walking/driving time from every residential to office/POI node

    Args:
        G (Graph): NetworkX Graph of city
        city (string): city map string

    Returns:
        ndarray: matrix of commute time for every level of traffic at every node
    """
    # Get origins and destinations
    origins = []
    destinations = []
    for i in range(len(city)):
        for j in range(len(city[0])):
            tile = city[i][j]
            
            if tile.isdigit(): # Residential 1-9
                origins.append(((i, j), int(tile)))
            elif tile in "abcde": # Commerical a-e
                weight = ord(tile) - ord('a') + 1
                destinations.append(((i, j), weight))
                
    # Create matrix of origin-destination pairs (routes)
    OD_pairs = []
    for (o_coord, o_val) in origins:
        for (d_coord, d_val) in destinations:
            OD_pairs.append((o_coord, d_coord, o_val + d_val))
    
    # Compute baseline travel times
    baseline = {t: {} for t in TRAFFIC_LEVELS}
    
    for traffic_level in TRAFFIC_LEVELS:
        # Iterate through all traffic levels
        def dynamic_weight(u, v, edge):
            """Dynamically adjusts weights separately for highways and roadways

            Args:
                edge (tuple): edge between u and v

            Returns:
                int: weight of edge
            """
            # H: base=1, R: base=2
            base = edge.get('base_weight', 1)
            edge_type = edge.get('edge_type', 'road')
            
            # Walking time does not fluctuate with traffic level
            if edge_type == 'access':
                return base
            
            if traffic_level == 'Rush':
                return base * 1.5 if base == 1 else base * 3.0 # highways less affected by rush hour traffic
            elif traffic_level == 'Low':
                return base * 0.8 # Everyone can go faster during low traffic
            else:
                return base
        
        for (origin, _, _) in OD_pairs:
            lengths = nx.single_source_dijkstra_path_length(G, origin, weight=dynamic_weight)
            
            for (o, d, _) in OD_pairs:
                if o == origin and d in lengths:
                    baseline[traffic_level][(o, d)] = lengths[d]
    
    return baseline, OD_pairs, origins, destinations
    
def spatialDistance(candidate_stations):
    """Computes matrix of euclidean distances between each candidate station

    Args:
        candidate_stations (list): list of candidate stations

    Returns:
        ndarray: 2D matrix of distances between stations
    """
    N = len(candidate_stations)
    
    # Spatial (Euclidean) distance
    spatial_distance = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            spatial_distance[i][j] = math.dist(candidate_stations[i], candidate_stations[j]) # euclidean distance
            
    return spatial_distance

def transitDistance(G, candidate_stations):
    """Computes travel time by taking transit for every candidate station-station pair

    Args:
        G (nx.Graph): NetworkX Graph of city
        candidate_stations (list): list of candidate stations
    """
    def transit_weight(u, v, edge):
        """Dynamically adjusts weights separately for highways and roadways

        Args:
            edge (tuple): edge between u and v

        Returns:
            int: weight of edge
        """
        base = edge.get('base_weight', 1)
        edge_type = edge.get('edge_type', 'road')

        # Walking stays the same
        if edge_type == 'access':
            return base

        # Transit ignores traffic and is faster than driving
        return base * 0.5
    
    N = len(candidate_stations)
    
    # Buses take roads - calculate Dijkstra length from graph
    bus_distance = np.zeros((N, N))
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, candidate_stations[i], weight=transit_weight)
        for j in range(N):
            d = lengths.get(candidate_stations[j], np.inf)
            bus_distance[i][j] = d
            bus_distance[j][i] = d
                
    # Metros independent of road network, use euclidean distance
    metro_distance = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            metro_distance[i][j] = math.dist(candidate_stations[i], candidate_stations[j])
            
    return bus_distance, metro_distance

def originToStationDistances(G, origins, destinations):
    """Calculate distances from every origin to destination

    Args:
        G (nx.Graph()): NetworkX graph
        origins (list): list of origin nodes
        destinations (list): list of destination nodes

    Returns:
        ndarray: matrix of distances between origins and destinations
    """
    origin_to_station = {}
    station_to_dest = {}
    
    # Calculate Dijkstra distance from origin to nearest station
    for (o, _) in origins:
        origin_to_station[o] = nx.single_source_dijkstra_path_length(G, o, weight='base_weight')
    # Calculate Dijkstra distance from destination to nearest station
    for (d, _) in destinations:
        station_to_dest[d] = nx.single_source_dijkstra_path_length(G, d, weight='base_weight')
            
    return origin_to_station, station_to_dest
    
def precompute(G, city, candidate_stations):
    """Precomputes matrices for O(1) matrix array lookups inside genetic algorithm loop

    Args:
        G (nx.Graph()): NetworkX Graph
        city (list): city map
        candidate_stations (list): list of candidate stations
    """
    # Compute spatial distances between each station
    spatial_distance = spatialDistance(candidate_stations)
    # Compute transit travel distances between each origin-destination pair (route)
    bus_distance, metro_distance = transitDistance(G, candidate_stations)
    # Compute baseline travel time between each origin-destination pair (route)
    base_time, OD_pairs, origins, destinations = originDestinationTimes(G, city)
    # Compute walking time between origin, station, and destination for all routes
    origin_to_station, station_to_dest = originToStationDistances(G, origins, destinations)
    
    return spatial_distance, base_time, bus_distance, metro_distance, OD_pairs, origin_to_station, station_to_dest

# FITNESS FUNCTION
def commuteTime(layout, base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest):
    """Calculate average commute time of transit layout

    Args:
        layout (list): station/stop layout
        base_time (ndarray): base transit time without using public transit
        bus_distance (ndarray): distances from each bus station to every other bus station
        metro_distance (ndarray): distances from each metro station to every other metro station
        OD_pairs (ndarray): pairs of origin and destination nodes
        candidate_stations (list): list of candidate station placements
        origin_to_station (ndarray): distances from every origin to its nearest station
        station_to_dest (ndarray): distances from every destination to its nearest station

    Returns:
        float: average commute time for every origin-destination pair using transit network
    """
    # Identify active stations
    active_indices = [i for i, v in enumerate(layout) if v != 0]
    
    # Return base time if no stations placed
    if not active_indices:
        return sum(demand * base_time[traffic_level].get((o, d), np.inf) for traffic_level in TRAFFIC_LEVELS for o, d, demand in OD_pairs) / len(OD_pairs)
    
    # Vectorize the Mode Penalties (Bus vs Metro vs Mixed)
    T_time = np.zeros((len(active_indices), len(active_indices)))

    for a, i in enumerate(active_indices):
        for b, j in enumerate(active_indices):
            # Find distance by each mode of transit
            if layout[i] == BUS and layout[j] == BUS:
                base = bus_distance[i][j]
            elif layout[i] == METRO and layout[j] == METRO:
                base = metro_distance[i][j]
            else:
                base = metro_distance[i][j]

            # Transit leg does not exist
            if base == np.inf:
                T_time[a][b] = np.inf
                continue

            # Calculate travel time based on speed and wait time of transit mode
            if layout[i] == BUS and layout[j] == BUS:
                T_time[a][b] = (base / BUS_SPEED) + BUS_WAIT
            elif layout[i] == METRO and layout[j] == METRO:
                T_time[a][b] = (base / METRO_SPEED) + METRO_WAIT
            else:
                T_time[a][b] = (base / METRO_SPEED + BUS_WAIT + METRO_WAIT + TRANSFER_PENALTY)
    
    # Extract walk times
    active_nodes = [candidate_stations[i] for i in active_indices]
    unique_origins = {o for o, d, _ in OD_pairs}
    unique_dests = {d for o, d, _ in OD_pairs}
    
    walk_o_cache = {o: np.array([origin_to_station[o].get(s, np.inf) for s in active_nodes]) for o in unique_origins}
    walk_d_cache = {d: np.array([station_to_dest[d].get(s, np.inf) for s in active_nodes]) for d in unique_dests}
    
    # Precompute best transit entry/exit routes
    cost_to_d_cache = {d: np.min(T_time + walk_d_cache[d], axis=1) for d in unique_dests}
    
    commute_time = 0
    
    # Evaluate OD Pairs with O(1) array additions
    for (o, d, demand) in OD_pairs:
        best_transit_route_time = np.min(walk_o_cache[o] + cost_to_d_cache[d])
        
        for traffic_level in TRAFFIC_LEVELS:
            base = base_time[traffic_level].get((o, d), np.inf)
            
            # commuters take faster route
            if best_transit_route_time < base:
                commute_time += demand * best_transit_route_time
            else:
                commute_time += demand * base
                
    return commute_time / len(OD_pairs)

def penaltyBudget(layout):
    """Computes the total cost incurred by placing stations in the layout

    Args:
        layout (list): list of stations

    Returns:
        int: 0 if total cost is within budget, amount over budget if total cost exceeds budget
    """
    total_cost = 0
    for station in layout:
        if station == BUS:
            total_cost += BUS_COST
        elif station == METRO:
            total_cost += METRO_COST
    return max(0, total_cost - BUDGET)

def penaltyMinDistance(layout, spatial_distance):
    """Calculates penalty from minimum stop/station placement violations

    Args:
        layout (list): transit network list
        spatial_distance (ndarray): distance between every candidate station

    Returns:
        int: number of minimum distance violations
    """
    violations = 0
    for i in range(len(layout)):
        for j in range(len(layout)):
            if i != j:
                # Check if stations exist
                if layout[i] != 0 and layout[j] != 0 and layout[i] == layout[j]:
                    # Calculate distance between stations
                    distance = spatial_distance[i][j]
                    
                    penalty = 0
                    if layout[i] == BUS:
                        # Bus stations
                        penalty = 1 if distance < BUS_MIN_DISTANCE else 0
                    else:
                        # Metro stations (layout[i] != 0 checked)
                        penalty = 1 if distance < METRO_MIN_DISTANCE else 0
                    violations += penalty
    return violations

def penaltyTransfer(layout, spatial_distance):
    """Calculates penalty incurred by transfer violations

    Args:
        layout (list): transit network
        spatial_distance (ndarray): distance between every candidate station

    Returns:
        int: number of metro stations without a nearby bus station
    """
    layout = np.array(layout)
    metros = (layout == METRO)
    buses = (layout == BUS)
    
    # There are no metros
    if not np.any(metros):
        return 0
       
    # There are no buses
    if not np.any(buses):
        return np.sum(metros)
    
    transfers = spatial_distance[metros][:, buses]
    valid_transfers = np.any(transfers <= TRANSFER_DISTANCE, axis=1)
    
    return np.sum(~valid_transfers)

# GENETIC ALGORITHM
def fitness(layout, spatial_distance, base_time, bus_distance, metro_distance, OD_matrix, candidate_stations, origin_to_station, station_to_dest):
    """Calculates fitness of layout

    Args:
        layout (list): station/stop layout
        spatial_distance (ndarray): distance between every candidate station
        base_time (ndarray): base transit time without using public transit
        bus_distance (ndarray): distances from each bus station to every other bus station
        metro_distance (ndarray): distances from each metro station to every other metro station
        OD_pairs (ndarray): pairs of origin and destination nodes
        candidate_stations (list): list of candidate station placements
        origin_to_station (ndarray): distances from every origin to its nearest station
        station_to_dest (ndarray): distances from every destination to its nearest station

    Returns:
        float: fitness value of transit layout
    """
    budget = W_BUDGET * penaltyBudget(layout)
    min_distance = W_MINDIST * penaltyMinDistance(layout, spatial_distance)
    transfer = W_TRANSFER * penaltyTransfer(layout, spatial_distance)
    commute_time = W_COMMUTE * commuteTime(layout, base_time, bus_distance, metro_distance, OD_matrix, candidate_stations, origin_to_station, station_to_dest)

    return commute_time + budget + min_distance + transfer

def tournament_selection(population, k):
    """Selects parents to cross over and create new child

    Args:
        population (list): list of transit layouts
        k (int): number of members to include in tournament

    Returns:
        list: list of parents selected to pass on genes
    """
    parent_pair = []
    
    for _ in range(NUM_PARENTS):
        # Randomly select tournament participants
        indices = random.sample(range(len(population)), k)
        
        # Find winning index
        members = [population[index] for index in indices]
        winning_member = min(members, key=lambda x: x[1])
        
        # Add to parent pair
        parent_pair.append(winning_member)
    
    return parent_pair[0][0], parent_pair[1][0]

def spatial_crossover(parent_1, parent_2, candidate_stations, height, width, mutation_probability):
    """Crosses over genes from each parent

    Args:
        parent_1 (list): transit layout of parent 1
        parent_2 (list): transit layout of parent 2
        candidate_stations (list): list of candidate stations
        height (int): height of city map
        width (int): width of city map
        mutation_probability (float): probability of mutation occurring in child

    Returns:
        list: newly evolved child from parents
    """
    child = np.zeros(len(parent_1), dtype=int)
    
    # Create bounding box from beta distribution - bias towards roughly equal splits
    top = int(random.betavariate(2, 2) * height)
    bottom = int(random.betavariate(2, 2) * height)
    left   = int(random.betavariate(2, 2) * width)
    right  = int(random.betavariate(2, 2) * width)
    
    # Switch if random sample made wrong variable larger
    if top > bottom: top, bottom = bottom, top
    if left > right: left, right = right, left
    
    for i, (r, c) in enumerate(candidate_stations):
        if top <= r <= bottom and left <= c <= right:
            # coord is inside bounding box
            child[i] = parent_1[i]
        else:
            # coord outside bounding box
            child[i] = parent_2[i]
        
        # Mutate child index if probability hit
        if random.random() < mutation_probability:
            child[i] = random.randint(0, 2)
            
    return child

def pct_diff(cur, old):
    """Calculates the percent difference between cur and old

    Args:
        cur (float): first number
        old (float): second number

    Returns:
        float: percent difference between cur and old
    """
    return (np.abs(cur - old) / ((cur + old) / 2)) * 100

def diversity(scores):
    """Calculates diversity of population using coefficient of variation

    Args:
        scores (list): population layouts and fitness values

    Returns:
        float: coefficient of variation of population (diversity measure)
    """
    scores = [score[1] for score in scores]
    mean_score = np.mean(scores)
    if mean_score == 0:
        return 0
    return (np.std(scores) / mean_score) * 100


# --- MAIN BLOCK ---
if __name__ == '__main__':
    # Ask for input city and create map
    city_index = int(input("Select Map Number [1: Monocentric (10x10), 2: Twin (18x8), 3: Dense (11x11), 4: Large (35x35), 5: Manhattan (30x40)]: "))
    city, height, width = CITIES[city_index - 1][0], CITIES[city_index - 1][1], CITIES[city_index - 1][2]
    
    city = cityStringTo2DArray(city, height, width)
    showMap(city)

    # Create NetworkX graph
    G, candidate_stations = createGraph(city)

    # Precompute distances for O(1) lookups inside GA loop
    spatial_distance, base_time, bus_distance, metro_distance, OD_pairs, origin_to_station, station_to_dest = precompute(G, city, candidate_stations)

    # --- GENETIC ALGORITHM ---

    # Create initial population of randomized stations
    population = [np.random.randint(0, 3, size=len(candidate_stations), dtype=np.int32) for _ in range(POP_SIZE)]

    fitness_history = []
    best_layout = []
    generation = 0
    
    def init_worker(spatial_distance, base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest):
        """Initialize worker for use in multiprocessing

        Args:
            spatial_distance (ndarray): distance between every candidate station
            base_time (ndarray): base transit time without using public transit
            bus_distance (ndarray): distances from each bus station to every other bus station
            metro_distance (ndarray): distances from each metro station to every other metro station
            OD_pairs (ndarray): pairs of origin and destination nodes
            candidate_stations (list): list of candidate station placements
            origin_to_station (ndarray): distances from every origin to its nearest station
            station_to_dest (ndarray): distances from every destination to its nearest station
        """
        global global_spatial_distance, global_base_time, global_bus_distance, global_metro_distance, global_OD_pairs, global_candidate_stations, global_origin_to_station, global_station_to_dest
        
        global_spatial_distance = spatial_distance
        global_base_time = base_time
        global_OD_pairs = OD_pairs
        global_candidate_stations = candidate_stations
        global_bus_distance = bus_distance
        global_metro_distance = metro_distance
        global_origin_to_station = origin_to_station
        global_station_to_dest = station_to_dest

    def fitness_wrapper(layout):
        """Wraps fitness function for use in pool

        Args:
            layout (list): transit network

        Returns:
            tuple: tuple of transit layout and fitness score
        """
        return (layout, fitness(layout, global_spatial_distance, global_base_time, global_bus_distance, global_metro_distance, global_OD_pairs, global_candidate_stations, global_origin_to_station, global_station_to_dest))

    # Initial k & mutation rate values
    k = K_INIT
    mutation_probability = INIT_MUTATION_RATE

    # Create multiprocessor workers
    with Pool(initializer=init_worker, initargs=(spatial_distance, base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest)) as pool:
        while True:
            # Calculate fitness of every population member
            scores = pool.map(fitness_wrapper, population)
            # Sort by fitness value
            scores.sort(key=lambda x: x[1])
            
            # Determine diversity of population to dynamically adjust parameters
            div = diversity(scores)
            if div < DIVERSITY_THRESH:
                # Population is not diverse, increase exploration rate
                k = max(K_MIN, k - 1) # Lower selection pressure, less fit individuals can reproduce
                mutation_probability = min(MUTATION_MAX, mutation_probability + 0.02) # Increase random mutations
            else:
                # Population is diverse, lower exploration rate
                k = min(K_MAX, k + 1) # Increase selection pressure, only the best win
                mutation_probability = max(MUTATION_MIN, mutation_probability - 0.01) # Decrease random mutations
            
            # Select elite members to copy to next generation
            best_layouts = [x[0] for x in scores[:int(POP_SIZE * ELITE_RATE)]]
            
            # Crossover
            new_population = best_layouts.copy()
            
            # Add children until population size is reached
            while len(new_population) < POP_SIZE:
                # Select two random parents from best layouts
                parent_1, parent_2 = tournament_selection(scores, k)
                
                # 2D bounding box crossover
                child = spatial_crossover(parent_1, parent_2, candidate_stations, height, width, mutation_probability)
                
                # Add child to population
                new_population.append(child)
            
            population = new_population
            
            best_score = scores[0][1]
            fitness_history.append(best_score)
            #print(f"Generation {generation}: Best = {best_score:.3f}; Layout = {scores[0][0]}")
            budget = W_BUDGET * penaltyBudget(scores[0][0])
            min_distance = W_MINDIST * penaltyMinDistance(scores[0][0], spatial_distance)
            transfer = W_TRANSFER * penaltyTransfer(scores[0][0], spatial_distance)
            commute_time = W_COMMUTE * commuteTime(scores[0][0], base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest)
            
            # Calculate number of bus and metro stations placed in network
            num_bus = 0
            num_metro = 0
            for station in scores[0][0]:
                if station == BUS:
                    num_bus += 1
                elif station == METRO:
                    num_metro += 1
            
            print(
                f"Gen {generation:4d} | fitness={best_score:10.2f} | budget={budget:7d} | mindist={min_distance:5d} | transfer={transfer:4d} | commute_time={commute_time:5.2f} | "
                f"div={div:5.1f}% | k={k} | mut={mutation_probability:.3f} | "
                f"bus={num_bus} metro={num_metro}"
            )
            
            # Convergence check
            fitness_max = np.max(fitness_history[len(fitness_history)-CONVERGENCE_GENERATIONS:]) # Gets maximum fitness of layouts in range for convergence check
            fitness_min = np.min(fitness_history[len(fitness_history)-CONVERGENCE_GENERATIONS:]) # Gets minimum fitness of layouts in range for convergence check
            if generation >= MIN_GENERATIONS and pct_diff(fitness_max, fitness_min) < CONVERGENCE_THRESHOLD:
                break
            
            # Increment generation
            generation += 1
        
    # Plot graphs and visualization
    layout = scores[0][0]
    
    # Plot fitness history
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.yscale('log')
    plt.title("GA Convergence")
    plt.show()
    
    """
    VISUALIZATION CODE PROVIDED BY CLAUDE SONNET 4.6
    Prompt: "Provide implementation for a visualization of the NetworkX Graph that shows traffic flow rates throughout the city.
            Edges should display thicker if more commuters use it, and residential and commercial density nodes should be scaled
            to represent their density. Provide evaluation metrics such as average commute time (from my function) and penalty
            violation information.
    Alterations were made by myself to fix bugs and improve visualization.
    """
    def computeBaseTrafficFlows(G, OD_pairs):
        """Computes edges thickness for base city with no transit layout

        Args:
            G (nx.Graph()): NetworkX graph
            OD_pairs (ndarray): matrix of origin-destination pairs

        Returns:
            dict: edge flows based on usage
        """
        edge_flow = {edge: 0 for edge in G.edges()}
        path_cache = {}

        def get_path(u, v):
            key = (u, v)
            if key not in path_cache:
                try:
                    path_cache[key] = nx.shortest_path(G, u, v, weight='base_weight')
                except:
                    path_cache[key] = None
            return path_cache[key]

        for (o, d, demand) in OD_pairs:
            path = get_path(o, d)
            if not path:
                continue

            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in edge_flow:
                    edge_flow[(u, v)] += demand
                else:
                    edge_flow[(v, u)] += demand

        return edge_flow

    def computeEdgeFlows(G, layout, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest, k=3):
        """Calculates edge flows given transit layout

        Args:
            G (nx.Graph()): NetworkX Graph
            layout (list): station/stop layout
            OD_pairs (ndarray): pairs of origin and destination nodes
            candidate_stations (list): list of candidate station placements
            bus_distance (ndarray): distances from each bus station to every other bus station
            metro_distance (ndarray): distances from each metro station to every other metro station
            origin_to_station (ndarray): distances from every origin to its nearest station
            station_to_dest (ndarray): distances from every destination to its nearest station
            k (int, optional): _description_. Defaults to 3.

        Returns:
            dict: edge flows based on usage
        """
        # Precompute station index lookup
        station_index = {s: i for i, s in enumerate(candidate_stations)}
        
        # Extract active stations
        stations = [(i, candidate_stations[i]) for i, v in enumerate(layout) if v != 0]
        
        if not stations:
            return computeBaseTrafficFlows(G, OD_pairs)
            #return {edge: 0 for edge in G.edges()}
        
        # Initialize edge flow
        edge_flow = {edge: 0 for edge in G.edges()}
        
        # Cache paths to avoid recomputation
        path_cache = {}
        
        def get_path(u, v):
            key = (u, v)
            if key not in path_cache:
                try:
                    path_cache[key] = nx.shortest_path(G, u, v, weight='base_weight')
                except:
                    path_cache[key] = None
            return path_cache[key]
        
        for (o, d, demand) in OD_pairs:
            
            # ---- STEP 1: pick top-k closest stations ----
            closest_to_o = sorted(
                stations,
                key=lambda x: origin_to_station[o].get(x[1], np.inf)
            )[:k]
            
            closest_to_d = sorted(
                stations,
                key=lambda x: station_to_dest[d].get(x[1], np.inf)
            )[:k]
            
            # ---- STEP 2: find best station pair (distance only) ----
            best_time = np.inf
            best_pair = None
            
            for (i, s1) in closest_to_o:
                walk1 = origin_to_station[o].get(s1, np.inf)
                if walk1 == np.inf:
                    continue
                    
                for (j, s2) in closest_to_d:
                    walk2 = station_to_dest[d].get(s2, np.inf)
                    if walk2 == np.inf:
                        continue
                    
                    mode_i = layout[i]
                    mode_j = layout[j]

                    if mode_i == BUS and mode_j == BUS:
                        transit = bus_distance[i][j] / BUS_SPEED + BUS_WAIT
                    elif mode_i == METRO and mode_j == METRO:
                        transit = metro_distance[i][j] / METRO_SPEED + METRO_WAIT
                    else:
                        base = min(bus_distance[i][j], metro_distance[i][j])
                        transit = base / ((BUS_SPEED + METRO_SPEED) / 2) + BUS_WAIT + METRO_WAIT + TRANSFER_PENALTY
                        
                    total = walk1 + transit + walk2
                    
                    if total < best_time:
                        best_time = total
                        best_pair = (s1, s2)
            
            if best_pair is None:
                continue
            
            s1, s2 = best_pair
            
            # ---- STEP 3: compute actual paths ONLY once ----
            path1 = get_path(o, s1)
            path2 = get_path(s1, s2)
            path3 = get_path(s2, d)
            
            if not path1 or not path2 or not path3:
                continue
            
            full_path = path1[:-1] + path2[:-1] + path3
            
            # ---- STEP 4: accumulate flow ----
            for u, v in zip(full_path[:-1], full_path[1:]):
                if (u, v) in edge_flow:
                    edge_flow[(u, v)] += demand
                else:
                    edge_flow[(v, u)] += demand
        
        return edge_flow
    
    def computeMetrics(layout, spatial_distance, base_time, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest):
        """Computes metrics for display in visualization

        Args:
            layout (list): station/stop layout
            spatial_distance (list): distances between candidate stations
            base_time (ndarray): travel time without public transit from each origin-destination pair
            OD_pairs (ndarray): pairs of origin and destination nodes
            candidate_stations (list): list of candidate station placements
            bus_distance (ndarray): distances from each bus station to every other bus station
            metro_distance (ndarray): distances from each metro station to every other metro station
            origin_to_station (ndarray): distances from every origin to its nearest station
            station_to_dest (ndarray): distances from every destination to its nearest station

        Returns:
            dict: dictionary of metrics given string name
        """
        # Calculate fitness function terms
        budget = penaltyBudget(layout)
        min_distance = penaltyMinDistance(layout, spatial_distance)
        transfer = penaltyTransfer(layout, spatial_distance)
        commute_time = commuteTime(layout, base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest)
        
        # Get base commute times for city without public transit
        empty_layout = np.zeros(len(layout))
        empty_commute_time = commuteTime(empty_layout, base_time, bus_distance, metro_distance, OD_pairs, candidate_stations, origin_to_station, station_to_dest)

        # Calculates number of buses and metros in layout
        num_bus = 0
        num_metro = 0
        for station in layout:
            if station == BUS:
                num_bus += 1
            elif station == METRO:
                num_metro += 1

        # Calculate total fitness function value
        total = (W_BUDGET * budget +
                W_MINDIST * min_distance +
                W_TRANSFER * transfer +
                W_COMMUTE * commute_time)

        return {
            "total": total,
            "commute_time": commute_time,
            "budget_violation": budget,
            "min_dist_violations": min_distance,
            "transfer_violations": transfer,
            "num_bus": num_bus,
            "num_metro": num_metro,
            "empty_commute_time": empty_commute_time
        }
    
    def drawFlowGraph(G, edge_flow, candidate_stations, layout, metrics):
        """Draw graph with edge flows

        Args:
            G (nx.Graph()): NetworkX Graph
            edge_flow (ndarray): Thickness of edges
            candidate_stations (list): list of candiate stations
            layout (list): transit network layout
            metrics (dict): dictionary of metrics given string name
        """
        import matplotlib as mpl
        
        # 1. Setup GridSpec to fix the overlap bug
        # This splits the window: 80% for the map, 20% for the metrics/legend
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
        ax = fig.add_subplot(gs[0, 0])
        text_ax = fig.add_subplot(gs[0, 1])
        text_ax.axis('off') # Hide axes for the text panel

        pos = {node: (node[1], -node[0]) for node in G.nodes()}
        edges = list(G.edges())

        # ---- Edge widths & Flow ----
        flows = np.array([edge_flow.get(edge, 0) for edge in edges], dtype=float)
        flows_log = np.log1p(flows)
        
        if flows_log.max() > 0:
            norm_flows = flows_log / flows_log.max()
        else:
            norm_flows = flows_log

        edge_widths = 1 + 6 * norm_flows
        zero_mask = flows == 0

        # Draw Zero-flow edges (pushed to background using zorder)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[e for i, e in enumerate(edges) if zero_mask[i]],
            width=0.5, edge_color="gainsboro", alpha=0.5, ax=ax#, zorder=1
        )

        # Draw Commute flow edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[e for i, e in enumerate(edges) if not zero_mask[i]],
            width=edge_widths[~zero_mask],
            edge_color=norm_flows[~zero_mask],
            edge_cmap=plt.cm.plasma,
            ax=ax#, zorder=2
        )

        # ---- Node Categorization (For Shapes and Clutter Reduction) ----
        bus_nodes, metro_nodes = [], []
        res_nodes, com_nodes, base_nodes = [], [], []
        
        bus_sizes, metro_sizes = [], []
        res_sizes, com_sizes = [], []
        
        station_index = {s: i for i, s in enumerate(candidate_stations)}
        tile_types = nx.get_node_attributes(G, 'tile_type')
        
        base_size = 60 # Reduced base size

        for node in G.nodes():
            if node in station_index and layout[station_index[node]] == 1:
                bus_nodes.append(node)
                bus_sizes.append(base_size * 4)
            elif node in station_index and layout[station_index[node]] == 2:
                metro_nodes.append(node)
                metro_sizes.append(base_size * 5)
            else:
                t = tile_types.get(node, None)
                if t is None or t in ['R', 'H']:
                    base_nodes.append(node)
                elif t.isdigit():
                    res_nodes.append(node)
                    res_sizes.append(base_size + int(t) * 40)
                elif t.isalpha():
                    com_nodes.append(node)
                    com_sizes.append(base_size + (ord(t) - ord('a')) * 40)
                else:
                    base_nodes.append(node)

        # Draw Background/Base Nodes
        nx.draw_networkx_nodes(G, pos, nodelist=base_nodes, node_color='whitesmoke', node_size=base_size, ax=ax)#, zorder=3)
        # Draw Residential Nodes
        nx.draw_networkx_nodes(G, pos, nodelist=res_nodes, node_color='lightcoral', node_size=res_sizes, alpha=0.75, ax=ax)#, zorder=3)
        # Draw Commercial Nodes
        nx.draw_networkx_nodes(G, pos, nodelist=com_nodes, node_color='mediumseagreen', node_size=com_sizes, alpha=0.75, ax=ax)#, zorder=3)
        
        # Draw Stations (Distinct Shapes, Highest Z-Order)
        nx.draw_networkx_nodes(G, pos, nodelist=bus_nodes, node_color='dodgerblue', node_size=bus_sizes, node_shape='s', edgecolors='black', linewidths=1.5, ax=ax)#, zorder=4)
        nx.draw_networkx_nodes(G, pos, nodelist=metro_nodes, node_color='darkviolet', node_size=metro_sizes, node_shape='D', edgecolors='black', linewidths=1.5, ax=ax)#, zorder=4)

        # ---- Labels ----
        # ONLY label the stations to eliminate text clutter
        station_labels = {n: "B" for n in bus_nodes}
        station_labels.update({n: "M" for n in metro_nodes})
        nx.draw_networkx_labels(G, pos, labels=station_labels, font_size=8, font_color='white', font_weight='bold', ax=ax)

        # ---- Colorbar ----
        norm = mpl.colors.Normalize(vmin=flows.min(), vmax=flows.max())
        sm = mpl.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Edge Flow (Traffic Density)")

        # ---- Metrics Panel (Fixed location) ----
        text = (
            f"METRICS REPORT\n"
            f"------------------------\n"
            f"Total Fitness : {metrics['total']:.2f}\n"
            f"Avg. Commute  : {(metrics['commute_time'] / 60):.0f}m {(((metrics['commute_time'] / 60) % 1) * 60):.0f}s\n"
            f"No Transit    : {(metrics['empty_commute_time'] / 60):.0f}m {(((metrics['empty_commute_time'] / 60) % 1) * 60):.0f}s\n"
            f"Budget Pen.   : {metrics['budget_violation']}\n"
            f"Min Dist Pen. : {metrics['min_dist_violations']}\n"
            f"Transfer Pen. : {metrics['transfer_violations']}\n"
            f"Bus Stations  : {metrics['num_bus']}\n"
            f"Metro Stations: {metrics['num_metro']}"
        )
        
        text_ax.text(0.05, 0.7, text, fontsize=11, family='monospace', verticalalignment='top')
        
        # Create a clean, readable legend in the text panel
        legend_elements = [
            mpl.lines.Line2D([0], [0], marker='s', color='w', markerfacecolor='dodgerblue', markersize=10, markeredgecolor='black', label='Bus Station'),
            mpl.lines.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkviolet', markersize=10, markeredgecolor='black', label='Metro Station'),
            mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, alpha=0.5, label='Residential (Scale=Demand)'),
            mpl.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumseagreen', markersize=10, alpha=0.5, label='Commercial (Scale=Demand)')
        ]
        text_ax.legend(handles=legend_elements, loc='lower left', title="Map Legend", frameon=False, bbox_to_anchor=(0.0, 0.2))

        ax.set_title("Transit Flow & Density Visualization")
        plt.tight_layout()
        plt.show()
    
    # Visualize empty layout (no transit)
    empty_layout = np.zeros(len(layout))
    edge_flows_empty = computeEdgeFlows(G, empty_layout, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest)
    metrics_empty = computeMetrics(empty_layout, spatial_distance, base_time, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest)
    drawFlowGraph(G, edge_flows_empty, candidate_stations, empty_layout, metrics_empty)
    
    # Visualize transit layout
    edge_flows = computeEdgeFlows(G, layout, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest)
    metrics = computeMetrics(layout, spatial_distance, base_time, OD_pairs, candidate_stations, bus_distance, metro_distance, origin_to_station, station_to_dest)
    drawFlowGraph(G, edge_flows, candidate_stations, layout, metrics)
