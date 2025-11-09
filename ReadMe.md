<div align="justify">

### __init__(self, env, ID, waypoints, my_coor)
Initializes the simulation entity (base station, sensor node, or event generator) with its environment, ID, coordinates, and role-specific attributes.
Sets up energy, routing, and event-handling parameters, and starts appropriate SimPy processes based on node type (B-, E-, or G-).


### move(self)
Updates the spatial position of mobile nodes based on periodic waypoint lists.
Simulates controlled mobility patterns by repositioning nodes at regular time intervals to mimic realistic network topology changes.

### compute(self)
Executed by the base station to perform anomaly detection and QoS control.
Aggregates incoming events, applies Isolation Forest for anomaly detection, computes accuracy metrics, and adjusts parameters (e.g., tau) using QoS feedback (PDR, latency, energy).


### neighbors(self)
Discovers neighboring nodes within sensing range and dynamically updates the network graph.
Randomizes link directions to form directed network structures, emulating real-world wireless communication uncertainty.


### update_graph(self)
Runs periodically to recalculate routing paths based on the latest network topology.
Employs shortest path algorithms to ensure adaptive and efficient data forwarding toward the base station.


### time_increment(self)
Maintains and increments the global simulation time.
Provides a synchronized timing mechanism across all entities and triggers event-driven updates.


### genEvent(self)
Executed by the event generator node to simulate environmental or sensor events.
Randomly spawns new events in space and time using historical data patterns, populating the simulation with realistic stimuli for sensor nodes.


### sense(self)
Handles local sensing activity at sensor nodes by detecting nearby events within range.
Consumes energy per detection and logs sensed events for later transmission, reflecting real-world sensing workloads.


## send(self)
Processes and transmits collected events toward the base station through multi-hop paths.
Applies redundancy filtering, robust z-score normalization, and energy-based transmission cost models, ensuring efficient and low-latency data forwarding.
</div>
