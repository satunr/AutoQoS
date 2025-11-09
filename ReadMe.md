
## AutoQoS: Autonomous QoS Modulation in Wireless Body Area Networks

<div align="justify">

**Abstract**

Wireless body area networks (WBANs) have emerged as a critical technology for continuous health monitoring, where transmission of physiological signals is essential. However, the heterogeneous nature of biomedical applications necessitates adaptive modulations of quality-of-service (QoS) parameters in user- or clinician-derived priorities. To address this challenge, we propose an adaptive framework, named **AutoQoS**, that leverages a differential equation-based dynamical system to regulate the reporting sensitivity thresholds of the biosensors in real-time. By determining whether successive physiological measurements provide clinically distinct information, AutoQoS meets the user-defined, individual, and joint throughput and energy consumption requirements, enabling trade-offs between conserving communication resources and preserving clinically significant events. The framework is validated on the MIMIC-III biomedical dataset across vital signs, showing its ability to balance QoS control with competitive accuracy in predicting important health events in comparison with an existing baseline. Overall, AutoQoS is demonstrably capable of personalizing operations to patient-specific monitoring needs, bridging the gap between clinical relevance and resource utilization.
</div>

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


### send(self)
Processes and transmits collected events toward the base station through multi-hop paths.
Applies redundancy filtering, robust z-score normalization, and energy-based transmission cost models, ensuring efficient and low-latency data forwarding.
</div>
