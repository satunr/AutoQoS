# -*- coding: utf-8 -*-
"""
Wireless Body Area Network (WBAN) Simulation with Adaptive QoS Modulation

This simulation implements an autonomous QoS modulation system for Wireless Body Area Networks
as described in the AutoQoS paper. The system adaptively adjusts data transmission thresholds
to maintain target packet delivery rates while optimizing energy consumption.

Key Components:
- Sensor nodes (E-): Mobile sensor nodes that sense events and forward data
- Base station (B-): Central node that receives and processes all data
- Event generator (G-): Generates events in the simulation environment
- Adaptive threshold (tau): Dynamically adjusted based on PDR and energy metrics
- Isolation Forest: Used for anomaly detection at the base station
- Redundancy checking: Reduces redundant data transmission using Minkowski distance
"""

import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import random
import simpy
import pickle
import pandas as pd
from copy import deepcopy
from scipy.spatial.distance import euclidean
from sklearn.ensemble import IsolationForest
import networkx as nx

np.set_printoptions(precision=2, suppress=True)

# ============================================================================
# GLOBAL METRICS TRACKING
# ============================================================================
# Lists to store performance metrics across simulation runs
PRECISION = []
RECALL = []
F1score = []
ACCURACY = []
PDR11 = []
TIMES = []

# Per-run metrics
dpre = []  # Precision over time
dacc = []  # Accuracy over time
df1 = []   # F1 score over time
drec = []  # Recall over time

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def accuracy2(list1, list2, T):
    """
    Calculate classification metrics (Precision, Recall, F1, Accuracy).
    
    Args:
        list1: Ground truth labels (actual)
        list2: Predicted labels
        T: Current time step
    
    Updates global lists: dpre, drec, df1, dacc
    """
    TP = TN = FP = FN = 0
    length1 = len(list2)
    
    for i in range(length1):
        if list1[i] == list2[i] and list1[i] == 1:
            TP += 1
        elif list1[i] == list2[i] and list1[i] == 0:
            TN += 1
        elif list1[i] == 1:
            FN += 1
        else:
            FP += 1
    
    if (TP + FP) != 0:
        dpre.append(TP / (TP + FP))
    
    if (TP + FN) != 0:
        drec.append(TP / (TP + FN))
    
    if (TP + FP) != 0 and (TP + FN) != 0:
        Prec = TP / (TP + FP)
        Reca = TP / (TP + FN)
        if Prec != 0 and Reca != 0:
            df1.append((2 * Prec * Reca) / (Prec + Reca))
    
    if (TP + TN) != 0 and (TP + TN + FN + FP) != 0:
        dacc.append((TP + TN) / (TP + TN + FN + FP))


def sensor_placement(N):
    """
    Generate sensor node locations using Gaussian distribution around base station.
    
    Args:
        N: Number of sensors to place
    
    Returns:
        List of (x, y) coordinates for sensor nodes
    """
    mean = [1.0, 0.5]  # Base station at (1, 0.5)
    cov = [[0.2, 0], [0, 0.1]]  # Covariance matrix (adjust spread)
    sensors = []
    
    while len(sensors) < N:
        x, y = np.random.multivariate_normal(mean, cov)
        if 0 <= x <= 2 and 0 <= y <= 1:  # Ensure within 2×1 area
            sensors.append((x, y))
    return sensors


# ============================================================================
# NETWORK GRAPH
# ============================================================================
G = nx.Graph()  # Global network topology graph

# ============================================================================
# TRANSMISSION PROBABILITIES
# ============================================================================
psch = 0.85  # Probability of sending to cluster head
psbs = 0.6   # Probability of sending to base station

def send_ch():
    """Check if node should send to cluster head."""
    return random.random() < psch

def send_bs():
    """Check if node should send to base station."""
    return random.random() < psbs

# ============================================================================
# ENERGY AND PERFORMANCE TRACKING
# ============================================================================
BENRG = []  # Base energy over time
ENRG = []   # Normalized energy consumption over time
PDR = []    # Packet Delivery Ratio over time
LTY = []   # Latency over time
LATENCY = []  # Detailed latency tracking

S_E = 0   # Total sensing energy consumed
SD_E = 0  # Total sending energy consumed

def energy():
    """Calculate average residual energy across all sensor nodes."""
    sum_energy = 0
    for i in range(1, U + 1):
        sum_energy += entities[i].rE
    return sum_energy / U


def latency():
    """Calculate average latency for delivered packets."""
    sum_latency = 0
    for events in LATENCY:
        sum_latency += abs(events[1] - events[0][2])
    if len(LATENCY) == 0:
        return 1000000  # Return large value if no packets delivered
    return sum_latency / len(LATENCY)


def pdr():
    """
    Calculate Packet Delivery Ratio (PDR).
    
    PDR = (Number of delivered packets) / (Number of generated packets)
    """
    global U, num, den
    
    for u in range(1, U + 1):
        for event in entities[u].generated:
            if event not in den:
                den.append(event)
    
    for event in entities[0].delivered:
        if event not in num:
            num.append(event)
    
    if len(den) == 0:
        return 0
    
    return float(len(num)) / float(len(den))

# ============================================================================
# ROBUST STATISTICS FOR DATA NORMALIZATION
# ============================================================================

def robust_z_score(data):
    """
    Calculate robust z-scores using Median Absolute Deviation (MAD).
    More robust to outliers than standard z-scores.
    
    Args:
        data: Input data array
    
    Returns:
        Robust z-scores
    """
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data)
    robust_z = (data - median) / mad
    return robust_z


def robust(median, data):
    """
    Calculate robust z-score for incoming data point when median is known.
    
    Args:
        median: Pre-computed median value
        data: New data point(s)
    
    Returns:
        Robust z-score(s) as list
    """
    data = np.array(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data).tolist()
    robust_z = (data - median) / mad
    return robust_z.tolist()


def robust_list(data):
    """
    Apply robust normalization to event data.
    Extracts 5 features from each event and normalizes first 4 using robust z-scores.
    
    Args:
        data: List of events, each with format [id, location, time, features]
              where features is [f1, f2, f3, f4, f5]
    
    Returns:
        List of normalized events
    """
    l1, l2, l3, l4, l5 = [], [], [], [], []
    A, B, C = [], [], []
    
    for events in data:
        l1.append(events[3][0])
        l2.append(events[3][1])
        l3.append(events[3][2])
        l4.append(events[3][3])
        l5.append(events[3][4])
        A.append(events[0])
        B.append(events[1])
        C.append(events[2])
    
    # Normalize first 4 features using pre-computed medians
    l1 = robust(median_list[0], l1)
    l2 = robust(median_list[1], l2)
    l3 = robust(median_list[2], l3)
    l4 = robust(median_list[3], l4)
    
    # Reconstruct events with normalized features
    ans = []
    for i in range(len(l1)):
        ans.append([A[i], B[i], C[i], [l1[i], l2[i], l3[i], l4[i], l5[i]]])
    return ans

# ============================================================================
# ADAPTIVE THRESHOLD CONTROL (Core QoS Modulation)
# ============================================================================
tau = 0.7  # Initial redundancy threshold (controls data transmission)
target_delivery = 0.7  # Target Packet Delivery Ratio
target_energy = 0.7  # Target normalized energy consumption
alpha = 0.4  # Learning rate for threshold adjustment
eng_max = 55.73000000004795  # Maximum energy consumption (for normalization)
tau_updates = []  # Track tau changes over time

prev_BENRG = 10000.0  # Previous base energy (for energy delta calculation)


def nredundancy(l11, l22):
    """
    Check if two data points are redundant using Minkowski distance.
    
    Args:
        l11: First data point (feature vector)
        l22: Second data point (feature vector)
    
    Returns:
        True if redundant (distance < tau), False otherwise
    """
    from scipy.spatial import distance
    if distance.minkowski(l11, l22) >= tau:
        return True
    return False


def adjustment1(O, T, O1, O2, E):
    """
    Adaptive threshold adjustment based on PDR and energy consumption.
    
    This is the core QoS modulation mechanism that adjusts tau (redundancy threshold)
    to maintain target PDR while managing energy consumption.
    
    Update rule: tau = tau + alpha * (PDR - target_delivery)
    
    Args:
        O: Current Packet Delivery Ratio
        T: Current time step
        O1: Ground truth labels (not used in current implementation)
        O2: Predicted labels (not used in current implementation)
        E: Energy consumed since last update
    """
    global tau, alpha, target_delivery, target_energy
    
    # Calculate adjustment based on PDR deviation from target
    pdr_res = alpha * (O - target_delivery)
    
    # Energy-based adjustment (currently commented out)
    joule_res = alpha * ((E / eng_max) - target_energy)
    
    # Update tau (only PDR component active)
    tau = tau + pdr_res  # + joule_res  # Energy component disabled
    
    tau_updates.append(tau)
    print(f"tau={tau:.3f}, alpha={alpha}, PDR={O:.3f}, target={target_delivery:.3f}, "
          f"E/E_max={E/eng_max:.3f}")
    
    # Track normalized energy consumption
    ENRG.append([E / eng_max, T])

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
# Load median values for robust normalization
df = pd.read_excel('FINAL113.xlsx')
median_list = df.median()

# Load historical data patterns
X1 = []
for index, row in df.iterrows():
    row_list = row.tolist()
    int_list = list(map(int, row_list))
    X1.append(int_list)


def med():
    """Return a random historical data pattern."""
    return X1[random.randint(0, len(X1) - 1)]

# ============================================================================
# NETWORK TOPOLOGY AND MOBILITY
# ============================================================================

def normal(A):
    """
    Normalize transition matrix so each row sums to 1.0.
    
    Args:
        A: Transition count matrix
    
    Returns:
        Normalized probability matrix
    """
    N = np.shape(A)[0]
    M = np.zeros((N, np.shape(A)[1]))
    
    for i in range(N):
        M[i, :] = A[i, :] / np.sum(A[i, :])
    
    return M


def place_node_in_zone(D, f, lat_dist=0.0124274, long_dist=0.0124274):
    """
    Place node in a random location within the simulation area.
    
    Args:
        D: Zone dictionary (not used in current implementation)
        f: Zone index (not used in current implementation)
        lat_dist: Latitude distance (not used)
        long_dist: Longitude distance (not used)
    
    Returns:
        Random (x, y) coordinates within bounds
    """
    return (random.uniform(X[0], X[1]), random.uniform(Y[0], Y[1]))


def calculate_period(P, N, I, M, u):
    """
    Calculate periodic movement pattern for a node.
    
    Args:
        P: Maximum periodicity
        N: Number of locations
        I: Importance vector for locations
        M: Transition probability matrix
        u: Node ID (0 for base station)
    
    Returns:
        List of location indices representing movement pattern
    """
    if u == 0:
        # Base station: single location based on importance
        start = np.random.choice([i for i in range(N)], p=I, size=1)[0]
        return [start]
    
    # Generate periodic movement pattern
    p = random.randint(2, P)
    L = []
    
    for i in range(p):
        if i == 0:
            # First location based on importance
            start = np.random.choice([i for i in range(N)], p=I, size=1)
            L.append(start[0])
        else:
            # Next location based on transition matrix
            last_location = L[-1]
            next_loc = np.random.choice([i for i in range(N)], 
                                       p=list(M[last_location, :]), size=1)
            L.append(next_loc[0])
    
    return L

# ============================================================================
# NODE CLASS DEFINITION
# ============================================================================

class Node(object):
    """
    Represents a node in the WBAN simulation.
    
    Three types of nodes:
    - 'E-': Edge/Sensor nodes (mobile, sense events, forward data)
    - 'B-': Base station (receives and processes all data)
    - 'G-': Event generator (creates events in the environment)
    """
    
    def __init__(self, env, ID, waypoints, my_coor):
        global T, Periodicity_List, G, tau, alpha, target_delivery, prev_BENRG, target_energy, S_E, SD_E
        
        self.ID = ID
        self.env = env
        self.my_coor = my_coor
        
        # Network topology
        self.nlist = []  # Neighbor list
        self.next_hop = []  # Next hop in routing path
        
        # Node state
        self.rE = 10000.0 if 'E-' in self.ID else None  # Residual energy (sensor nodes only)
        self.events = []  # Detected events
        self.generated = []  # Generated event IDs (for PDR calculation)
        self.forward = []  # Events to forward to next hop
        
        # Base station specific
        if 'B-' in self.ID:
            self.delivered = []  # Delivered event IDs
            self.out1 = []  # Ground truth labels
            self.out2 = []  # Predicted labels (from Isolation Forest)
            self.num = 0
            self.assign = True
            self.ch_dis = True
            self.ch_bc = True
            self.OUT1 = []  # Redundant events (for analysis)
            self.OUT2 = []
            self.OUT3 = []
            self.env.process(self.neighbors())
            self.env.process(self.compute())
        
        # Sensor node specific
        elif 'E-' in self.ID:
            self.dict = {i: [] for i in range(tp)}  # Time-period based event storage
            self.my_waypoint = Periodicity_List[int(self.ID[2:])][0]
            self.ch_bc = False
            self.env.process(self.neighbors())
            self.env.process(self.update_graph())
            self.env.process(self.sense())
            self.env.process(self.send())
        
        # Event generator
        elif 'G-' in self.ID:
            self.nlist = None
            self.env.process(self.genEvent())
            self.env.process(self.time_increment())
    
    def move(self):
        """Update node position based on periodic movement pattern."""
        global X, Y, Periodicity_List, mho, D
        
        while True:
            if T % mho == 0:
                tP = deepcopy(Periodicity_List[int(self.ID[2:])])
                self.my_waypoint = tP[int(T / mho) % len(tP)]
                self.my_coor = place_node_in_zone(D, self.my_waypoint)
            yield self.env.timeout(minimumWaitingTime)
    
    def compute(self):
        """
        Base station computation: anomaly detection and QoS adjustment.
        
        Every 10 time steps:
        1. Receives forwarded events
        2. Applies Isolation Forest for anomaly detection
        3. Calculates classification metrics
        4. Adjusts tau threshold based on PDR
        """
        global target_delivery, tau, prev_BENRG, target_energy
        
        while True:
            if T % 10 == 0:
                # Receive forwarded events
                if self.forward:
                    for events in self.forward:
                        self.events.append(events[0])
                    self.forward = []
                
                if self.events:
                    # Prepare data for Isolation Forest
                    Yi = []  # Ground truth (last feature is anomaly label)
                    Yj = []  # Predictions
                    temp = []
                    
                    for j in self.events:
                        temp.append(j[3])  # Feature vector
                        self.out1.append(j[3][4])  # Last feature is ground truth label
                        Yi.append(j[3][4])
                    
                    # Train Isolation Forest and predict
                    clf = IsolationForest(random_state=0).fit(temp)
                    LABEL = clf.predict(temp).tolist()
                    
                    # Convert Isolation Forest labels (-1=anomaly, 1=normal) to binary
                    for i in LABEL:
                        if i == 1:
                            self.out2.append(0)  # Normal
                            Yj.append(0)
                        else:
                            self.out2.append(1)  # Anomaly
                            Yj.append(1)
                    
                    # Calculate metrics and adjust QoS
                    accuracy2(self.out1, self.out2, T)
                    ENG = energy()
                    BENRG.append([ENG, T])
                    pd = pdr()
                    
                    if T != 0:
                        # Adjust threshold based on current PDR
                        adjustment1(pd, T, self.out1, self.out2, prev_BENRG - ENG)
                        prev_BENRG = ENG
                    
                    PDR.append([pd, T])
                    LTY.append([latency(), T])
                    self.events = []
            
            yield self.env.timeout(minimumWaitingTime)
    
    def neighbors(self):
        """
        Discover neighbors based on sensing range.
        Updates network graph with discovered links.
        """
        global sensing_range, U, c_p, G
        
        def convert(G):
            """Convert undirected graph to directed graph with random edge directions."""
            H = nx.DiGraph()
            H.add_nodes_from(list(G.nodes()))
            
            for (u, v) in G.edges():
                if random.choice([0, 1]) == 0:
                    H.add_edge(u, v)
                else:
                    H.add_edge(v, u)
            return H
        
        while True:
            self.nlist = []
            for u in range(U):
                if u == int(self.ID[2:]):
                    continue
                
                # Add randomness to sensing range
                add = random.random() * 0.05 * random.randint(-1, 1)
                if euclidean(list(entities[u].my_coor), list(self.my_coor)) < sensing_range[c_p] + add:
                    self.nlist.append(u)
                    G.add_edge(u, int(self.ID[2:]))
            
            G = convert(G)
            yield self.env.timeout(minimumWaitingTime)
    
    def update_graph(self):
        """
        Update routing paths using shortest path algorithm.
        Runs periodically to adapt to network topology changes.
        """
        global G, period
        
        while True:
            if T % period == 1:
                if nx.has_path(G, source=0, target=int(self.ID[2:])):
                    self.next_hop = nx.shortest_path(G, source=0, target=int(self.ID[2:]))
                    self.next_hop.pop()  # Remove self from path
            
            yield self.env.timeout(minimumWaitingTime)
    
    def time_increment(self):
        """Increment global simulation time."""
        global T
        
        while True:
            T = T + 1
            print(f"Time is {T}")
            yield self.env.timeout(minimumWaitingTime)
    
    def genEvent(self):
        """
        Generate events in the simulation environment.
        Events are placed randomly in the simulation area with data from historical patterns.
        """
        global T, frequencyEvent, globalEventCounter, X, Y, Event_Time_Dict, EV
        
        while True:
            if T % frequencyEvent == 0:
                self.events = []
                ev = EV[T]
                
                for i in range(how_many_events):
                    D = med()  # Get historical data pattern
                    new_event = [
                        globalEventCounter,
                        [random.uniform(X[0], X[1]), random.uniform(Y[0], Y[1])],
                        T,
                        D
                    ]
                    
                    if globalEventCounter not in Event_Time_Dict.keys():
                        Event_Time_Dict[globalEventCounter] = T
                    
                    self.events.append(new_event)
                    globalEventCounter += 1
                    ev.append(new_event)
                
                EV[T].extend(ev)
            
            yield self.env.timeout(minimumWaitingTime)
    
    def sense(self):
        """
        Sensor node sensing: detect events within sensing range.
        Consumes energy for each sensed event.
        """
        global baseE, frequencyEvent, c_p, senseE, sensing_range, U, S_E
        
        while self.rE > baseE:
            if T >= 0 and T % frequencyEvent == 0:
                # Check all events from generator
                for each in entities[U + 1].events:
                    add = random.random() * 0.05 * random.randint(-1, 1)
                    
                    # Check if event is within sensing range
                    if euclidean([each[1][0], each[1][1]], 
                                [self.my_coor[0], self.my_coor[1]]) <= sensing_range[c_p] + add:
                        self.rE = self.rE - senseE[c_p]
                        S_E = senseE[c_p] + S_E
                        self.events.append(each)
                        self.generated.append(each[0])
                
                self.move()
            
            yield self.env.timeout(minimumWaitingTime)
    
    def send(self):
        """
        Sensor node transmission: process events, check redundancy, and forward to base station.
        
        Process:
        1. Normalize events using robust z-scores
        2. Check redundancy against stored events in time-period buckets
        3. Store non-redundant events
        4. Periodically transmit stored events to base station
        """
        global T, baseE, SD_E
        
        while True:
            if 'E-' in self.ID and self.rE > baseE and T > 1:
                # Forward events from other nodes
                if self.forward:
                    for events in self.forward:
                        temp1 = events[0]  # Event
                        temp2 = events[1]  # Remaining path
                        NEXT = temp2.pop()
                        entities[NEXT].forward.append([temp1, temp2])
                    self.forward = []
                
                # Process locally sensed events
                if self.events:
                    TEMP = []
                    for events in self.events:
                        TEMP.append(events)
                    
                    # Normalize using robust z-scores
                    robust_z = robust_list(TEMP)
                    TEMP = robust_z
                    
                    # Check redundancy and store events
                    for dat in TEMP:
                        che = True  # Assume non-redundant
                        time_bucket = dat[2] % tp
                        
                        if self.dict[time_bucket]:
                            # Check against stored events in same time bucket
                            for dat1 in self.dict[time_bucket]:
                                # Check if redundant (Minkowski distance < tau)
                                if nredundancy(dat1[3][:-1], dat[3][:-1]) == False:
                                    che = False  # Found similar event, mark as redundant
                                    break
                            
                            if che:
                                # Non-redundant, store it
                                self.dict[time_bucket].append(dat)
                            else:
                                # Redundant, send to base station for analysis
                                entities[0].OUT1.append(dat)
                        else:
                            # Empty bucket, store event
                            self.dict[time_bucket].append(dat)
                    
                    self.events = []
                
                # Periodically transmit stored events
                if T % tp == 0 and self.next_hop:
                    for i in range(tp):
                        temp = self.next_hop[-1]  # Next hop in path
                        
                        for event in self.dict[i]:
                            if send_bs():  # Probabilistic transmission
                                # Calculate distance-based energy consumption
                                dist = euclidean([event[1][0], event[1][1]], 
                                                [self.my_coor[0], self.my_coor[1]])
                                K = 1024 * 6  # Packet size
                                send_energy = 0.0001 * K + 0.0001 * K * (dist ** 2)
                                self.rE = self.rE - send_energy
                                SD_E = send_energy + SD_E
                                
                                # Forward to next hop
                                entities[temp].forward.append([event, self.next_hop[:-1]])
                                LATENCY.append([event, T])
                                
                                if event not in entities[0].delivered:
                                    entities[0].delivered.append(event[0])
                        
                        self.dict[i] = []  # Clear bucket after transmission
            
            yield self.env.timeout(minimumWaitingTime)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
N = 18  # Number of locations
rP = 0.1  # Reset probability
P = 10  # Maximum periodicity
U = 20  # Number of sensor nodes
period = 30  # Network update period
Time = 2000  # Total simulation time
W = 20  # Learning window
th = 0.8  # Threshold (unused)
recur = 10  # Event recurrence (unused)
minimumWaitingTime = 1  # Minimum simulation time step
frequencyEvent = 2  # Event generation frequency
disc = 50  # Discovery period (unused)
dis = 100  # Cluster head determination period (unused)
globalEventCounter = 0  # Global event counter
how_many_events = 50  # Events per generation cycle
mho = 2  # Movement frequency
baseE = 100.0  # Base energy threshold (nodes die below this)
c_p = 0  # Current power mode index
senseE = [0.25, 2]  # Sensing energy consumption [low, high]
sendE = [0.6, 0.3]  # Sending energy (unused, replaced by distance-based)
X = [0, 2]  # X-axis bounds
Y = [0, 1]  # Y-axis bounds
tp = 5  # Time period for event batching
F = 5  # Feature size
sensing_range = [0.5, 20.0]  # Sensing range [min, max]

# ============================================================================
# INITIALIZATION
# ============================================================================
Event_Time_Dict = {}
T = 0
EV = {t: [] for t in range(Time)}
delay, num, den = [], [], []  # For PDR calculation
pdr_vec, lat_vec = [], []  # Unused
DataTrace = {u: [] for u in range(U + 1)}  # Unused
Parameters = {u: None for u in range(U + 1)}  # Unused
Accuracy = {u: None for u in range(U + 1)}  # Unused

# Location dictionary (unused in current implementation, kept for compatibility)
D = {1: [41.94168151131379, 12.545585304992104], 2: [41.92476673886478, 12.50625936430367],
     3: [41.92950124742932, 12.661287494647931], 4: [41.822109626135784, 12.576870129753795],
     5: [41.83066386982749, 12.506864462513247], 6: [41.904871135537455, 12.479667539836901],
     7: [41.81934831528825, 12.625561783954264], 8: [41.920731147413576, 12.465920041301466],
     9: [41.95396867995905, 12.432229440191799], 10: [41.87265183449205, 12.497255236012197],
     11: [41.91491919568486, 12.490692719043263], 12: [41.940673116610704, 12.497835832579032],
     13: [41.9372946306869, 12.559855361373568], 14: [41.90240523191011, 12.664630598981782],
     15: [41.84173148244936, 12.42826198946532], 16: [41.820430192801155, 12.536223191022177],
     17: [41.84737655362369, 12.621178002524523], 18: [41.914213175459714, 12.685784308489556],
     19: [41.93938795534091, 12.600234284322479], 20: [41.912181330418235, 12.628113488539494],
     21: [41.91275406625548, 12.61946347541366], 22: [41.85513275250908, 12.646441932103091],
     23: [41.8350393085066, 12.417148144390815], 24: [41.81682466341483, 12.68252006353243],
     25: [41.94845230615545, 12.582148030207213], 26: [41.95419306661668, 12.646914840781903],
     27: [41.94280859674365, 12.432050040311404], 28: [41.89489527539156, 12.65985352246119],
     29: [41.85462450607504, 12.580347929633108], 30: [41.81243185797423, 12.501498471075136],
     31: [41.970398038654125, 12.65142088823363], 32: [41.94999731811065, 12.45703544671786],
     33: [41.810353494774226, 12.52463082525551], 34: [41.877056174047, 12.51663126179204],
     35: [41.8803584542982, 12.445165016125012], 36: [41.848415983392066, 12.637884192670661],
     37: [41.88824852907222, 12.43786429347144], 38: [41.84479136967586, 12.673644613463955],
     39: [41.933223699775745, 12.597822743042908], 40: [41.84514310846181, 12.436637553162587],
     41: [41.79867567851633, 12.464755382018021], 42: [41.8961401942645, 12.560931559488518],
     43: [41.90903809951578, 12.529956548703863], 44: [41.89661615995606, 12.651061194428001],
     45: [41.947301190005945, 12.680597267991113], 46: [41.85090073735968, 12.478686451063043],
     47: [41.79960710869491, 12.679802525069812], 48: [41.92042893448671, 12.436489644478973],
     49: [41.83521664543882, 12.621758865198718], 50: [41.83604076835692, 12.41961740948081]}

D[0] = np.mean([tuple(val) for val in D.values()], axis=0)

recBufferCapacity = 1000  # Receive buffer capacity
rB = 5.0  # Neighborhood radius (unused)

# Generate transition matrix for mobility
A = np.random.randint(5, size=(N, N))
M = normal(A)  # Normalized transition probability matrix

# Calculate location importance
I = [np.sum(A[i, :]) + np.sum(A[:, i]) - A[i, i] for i in range(N)]
I = [float(I[i]) / float(sum(I)) for i in range(N)]

# Generate movement patterns for all nodes
Periodicity_List = {}
for u in range(U + 1):
    Periodicity_List[u] = calculate_period(P, N, I, M, u)

# Generate sensor locations
sensor_locations = sensor_placement(U)

# ============================================================================
# SIMULATION SETUP AND EXECUTION
# ============================================================================
env = simpy.Environment()
entities = []

# Create nodes
Coor = [Periodicity_List[i] for i in range(U + 1)]
print(Coor)

for i in range(U + 2):
    if i == 0:
        # Base station
        entities.append(Node(env, 'B-' + str(i), Coor, (1, 0.5)))
        G.add_node(i)
    elif i <= U:
        # Sensor node
        entities.append(Node(env, 'E-' + str(i), Coor, sensor_locations[i - 1]))
        G.add_node(i)
    else:
        # Event generator
        entities.append(Node(env, 'G-' + str(i), None, None))

# Run simulation
env.run(until=Time)

# ============================================================================
# RESULTS PROCESSING
# ============================================================================
event_eg = entities[U + 1].events

PRECISION.append(dpre)
RECALL.append(drec)
F1score.append(df1)
ACCURACY.append(dacc)
values = [item[0] for item in PDR]
PDR11.append(PDR)
TIMES = [item[1] for item in PDR]

# Calculate final metrics
final_prec = PRECISION[0][len(PRECISION[0]) - 1] if PRECISION[0] else 0
final_rec = RECALL[0][len(RECALL[0]) - 1] if RECALL[0] else 0
final_f1 = F1score[0][len(F1score[0]) - 1] if F1score[0] else 0
final_acc = ACCURACY[0][len(ACCURACY[0]) - 1] if ACCURACY[0] else 0

print(f"final_prec: {final_prec}")
print(f"final_rec: {final_rec}")
print(f"final_f1: {final_f1}")
print(f"final_acc: {final_acc}")

# Save results
all_data_packet = []#pickle.load(open("10_data_packets_w_acc.p", "rb"))
all_data_packet.append((LTY, PDR, ENRG, tau_updates, PRECISION, RECALL, F1score, ACCURACY))
pickle.dump(all_data_packet, open("test_w_new_code.p", "wb"))
print(f"Saved results, total runs: {len(all_data_packet)}")
