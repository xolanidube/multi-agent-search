"""
Explore Use Cases for Multi-Agent Search Framework.

Demonstrates the framework's applicability across different domains:
- Search and Rescue (SAR)
- Network Security Threat Hunting
- Warehouse Inventory Search
- Distributed Code Debugging
- Drone Swarm Surveillance
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from src.core.grid import Grid2D, GridAgent, GridCellStatus, MovementType
from src.strategies.grid_strategies import (
    GreedyGridStrategy, SpiralStrategy, QuadrantStrategy,
    SwarmStrategy, WavefrontStrategy, ProbabilisticGridStrategy
)
from src.visualization.grid_animator import Grid2DAnimator, run_grid_simulation


# =============================================================================
# FIXED LARGE SCALE - Better strategy and configuration
# =============================================================================

def create_large_scale_success():
    """Large scale that actually finds the target."""
    print("\n" + "=" * 60)
    print("LARGE SCALE GRID (40x40) - OPTIMIZED FOR SUCCESS")
    print("=" * 60)

    # Less obstacles, 8-way movement, target closer to center
    grid = Grid2D(40, 40, MovementType.EIGHT_WAY)
    grid.add_obstacles(density=0.05)  # Reduced from 0.1

    # Place target in a more accessible location
    grid.place_target((30, 30))

    # Create hotzone around target area to help probabilistic search
    grid.set_hotzone(center=(30, 30), radius=10, probability_boost=5.0)

    # 8 agents with mixed strategies for better coverage
    agents = [
        GridAgent(0, start_position=(0, 0)),
        GridAgent(1, start_position=(39, 0)),
        GridAgent(2, start_position=(0, 39)),
        GridAgent(3, start_position=(39, 39)),
        GridAgent(4, start_position=(20, 0)),
        GridAgent(5, start_position=(20, 39)),
        GridAgent(6, start_position=(0, 20)),
        GridAgent(7, start_position=(39, 20)),
    ]

    # Mix of strategies - some greedy, some probabilistic
    strategies = [
        ProbabilisticGridStrategy(exploitation_factor=0.9),  # High exploitation
        ProbabilisticGridStrategy(exploitation_factor=0.9),
        GreedyGridStrategy(),
        GreedyGridStrategy(),
        QuadrantStrategy(num_agents=8, agent_idx=4),
        QuadrantStrategy(num_agents=8, agent_idx=5),
        SwarmStrategy(repulsion_strength=1.5),
        SwarmStrategy(repulsion_strength=1.5),
    ]

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=200.0
    )

    print(f"Result: {'FOUND' if found else 'NOT FOUND'} in {elapsed:.1f}s")
    print(f"Cells explored: {grid.count_explored()}")

    animator = Grid2DAnimator(grid, agents, show_trails=True, figsize=(14, 12))
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    os.makedirs(output_dir, exist_ok=True)

    animator.create_animation(
        os.path.join(output_dir, "grid_large_scale_success.gif"),
        fps=15,
        title="Large Scale Search (40x40, 8 Agents, Optimized)"
    )

    return found, elapsed


# =============================================================================
# USE CASE 1: SEARCH AND RESCUE
# =============================================================================

class TerrainType(Enum):
    """SAR terrain types affecting search speed."""
    OPEN_FIELD = "open"
    FOREST = "forest"
    WATER = "water"
    CLIFF = "cliff"
    BUILDING = "building"


def create_sar_scenario():
    """
    Search and Rescue Scenario.

    Simulates:
    - Wilderness area with varied terrain
    - Missing person (target)
    - Multiple SAR teams with different capabilities
    - Golden hour time constraint
    """
    print("\n" + "=" * 60)
    print("USE CASE: SEARCH AND RESCUE (SAR)")
    print("=" * 60)
    print("Scenario: Missing hiker in wilderness area")
    print("Time constraint: Golden hour (simulated)")

    # Create wilderness terrain (30x30 = 900 cells = ~900 acres)
    grid = Grid2D(30, 30, MovementType.EIGHT_WAY)

    # Add terrain features
    # Dense forest areas (slower to search)
    for x in range(5, 15):
        for y in range(5, 15):
            if random.random() < 0.3:
                grid.cells[(x, y)].search_cost = 0.3  # Slower
                grid.cells[(x, y)].metadata['terrain'] = 'forest'

    # River (impassable)
    for y in range(0, 30):
        if grid.is_valid(15, y):
            grid.cells[(15, y)].is_obstacle = True
            grid.status[(15, y)] = GridCellStatus.OBSTACLE
    # Bridge
    grid.cells[(15, 12)].is_obstacle = False
    grid.status[(15, 12)] = GridCellStatus.UNEXPLORED
    grid.cells[(15, 13)].is_obstacle = False
    grid.status[(15, 13)] = GridCellStatus.UNEXPLORED

    # Cliff area (impassable)
    for x in range(20, 25):
        for y in range(0, 5):
            if grid.is_valid(x, y):
                grid.cells[(x, y)].is_obstacle = True
                grid.status[(x, y)] = GridCellStatus.OBSTACLE

    # Last known position creates hotzone
    last_seen = (10, 10)
    grid.set_hotzone(center=last_seen, radius=8, probability_boost=4.0)

    # Place missing person (unknown to searchers)
    grid.place_target((22, 18))

    # SAR Teams
    teams = [
        GridAgent(0, start_position=(0, 0)),      # Ground team 1
        GridAgent(1, start_position=(0, 29)),     # Ground team 2
        GridAgent(2, start_position=(29, 29)),    # K9 unit (faster)
        GridAgent(3, start_position=last_seen),   # Helicopter (starts at LKP)
    ]
    teams[2].speed = 1.5  # K9 faster
    teams[3].speed = 2.0  # Helicopter fastest

    # Strategies
    strategies = [
        WavefrontStrategy(),                          # Systematic ground search
        QuadrantStrategy(num_agents=4, agent_idx=1),  # Cover assigned zone
        GreedyGridStrategy(),                         # K9 follows scent (greedy)
        ProbabilisticGridStrategy(exploitation_factor=0.95),  # Heli checks hotspots
    ]

    print(f"\nDeploying {len(teams)} SAR teams:")
    print("  - Ground Team Alpha (systematic wavefront)")
    print("  - Ground Team Bravo (zone coverage)")
    print("  - K9 Unit (fast greedy search)")
    print("  - Helicopter (probability-based)")

    frames, found, elapsed = run_grid_simulation(
        grid, teams, strategies, time_budget=60.0  # Golden hour
    )

    print(f"\nResult: {'PERSON FOUND!' if found else 'Search continues...'}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Area covered: {grid.count_explored()}/{len(grid) - sum(1 for c in grid.cells.values() if c.is_obstacle)} cells")

    animator = Grid2DAnimator(grid, teams, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_sar.gif"),
        fps=10,
        title="Search and Rescue: Missing Hiker"
    )

    return found, elapsed


# =============================================================================
# USE CASE 2: NETWORK SECURITY - THREAT HUNTING
# =============================================================================

class NodeType(Enum):
    """Network node types."""
    SERVER = "server"
    WORKSTATION = "workstation"
    ROUTER = "router"
    FIREWALL = "firewall"
    DATABASE = "database"


def create_threat_hunting_scenario():
    """
    Network Security Threat Hunting Scenario.

    Simulates:
    - Network topology as a grid
    - Compromised node (target)
    - Security scanning agents
    - Dwell time constraint
    """
    print("\n" + "=" * 60)
    print("USE CASE: NETWORK SECURITY - THREAT HUNTING")
    print("=" * 60)
    print("Scenario: Detecting compromised node in corporate network")
    print("Time constraint: Minimize dwell time")

    # Network topology (25x25 = 625 nodes)
    grid = Grid2D(25, 25, MovementType.FOUR_WAY)

    # Network segments (some harder to scan)
    # DMZ (top rows) - well monitored
    for x in range(25):
        for y in range(0, 5):
            grid.cells[(x, y)].search_cost = 0.05  # Fast scan
            grid.cells[(x, y)].metadata['segment'] = 'DMZ'

    # Internal network - slower to scan
    for x in range(25):
        for y in range(5, 20):
            grid.cells[(x, y)].search_cost = 0.15
            grid.cells[(x, y)].metadata['segment'] = 'internal'

    # Database zone - slowest (deep inspection needed)
    for x in range(10, 15):
        for y in range(20, 25):
            grid.cells[(x, y)].search_cost = 0.3
            grid.cells[(x, y)].metadata['segment'] = 'database'

    # Air-gapped segments (obstacles)
    for x in range(0, 5):
        grid.cells[(x, 22)].is_obstacle = True
        grid.status[(x, 22)] = GridCellStatus.OBSTACLE

    # IOC (Indicator of Compromise) suggests internal network
    grid.set_hotzone(center=(15, 12), radius=6, probability_boost=3.0)

    # Compromised node
    grid.place_target((18, 14))

    # Security agents
    agents = [
        GridAgent(0, start_position=(0, 0)),      # Vulnerability scanner
        GridAgent(1, start_position=(24, 0)),     # Network monitor
        GridAgent(2, start_position=(12, 12)),    # EDR agent
        GridAgent(3, start_position=(12, 24)),    # SIEM analyzer
    ]

    strategies = [
        WavefrontStrategy(),                          # Systematic scan
        SwarmStrategy(repulsion_strength=2.0),        # Spread out
        ProbabilisticGridStrategy(exploitation_factor=0.9),  # Follow IOCs
        GreedyGridStrategy(),                         # Quick coverage
    ]

    print(f"\nDeploying {len(agents)} security agents:")
    print("  - Vulnerability Scanner (wavefront)")
    print("  - Network Monitor (swarm)")
    print("  - EDR Agent (probabilistic)")
    print("  - SIEM Analyzer (greedy)")

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=40.0
    )

    print(f"\nResult: {'THREAT IDENTIFIED!' if found else 'Scanning continues...'}")
    print(f"Time to detection: {elapsed:.1f}s")
    print(f"Nodes scanned: {grid.count_explored()}")

    animator = Grid2DAnimator(grid, agents, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_threat_hunting.gif"),
        fps=10,
        title="Network Security: Threat Hunting"
    )

    return found, elapsed


# =============================================================================
# USE CASE 3: WAREHOUSE INVENTORY SEARCH
# =============================================================================

def create_warehouse_scenario():
    """
    Warehouse Inventory Search Scenario.

    Simulates:
    - Warehouse with aisles and racks
    - Misplaced item (target)
    - Warehouse robots/workers
    - SLA time constraint
    """
    print("\n" + "=" * 60)
    print("USE CASE: WAREHOUSE INVENTORY SEARCH")
    print("=" * 60)
    print("Scenario: Finding misplaced high-value item")
    print("Time constraint: Order fulfillment SLA")

    # Warehouse layout (30x20)
    grid = Grid2D(30, 20, MovementType.FOUR_WAY)

    # Create aisle structure
    for aisle in range(3, 28, 5):  # Aisles every 5 units
        for y in range(0, 20):
            if y % 8 != 0:  # Leave cross-aisles
                grid.cells[(aisle, y)].is_obstacle = True
                grid.status[(aisle, y)] = GridCellStatus.OBSTACLE

    # Receiving area (high probability - last scanned location)
    grid.set_hotzone(center=(2, 10), radius=4, probability_boost=3.0)

    # Shipping area (also possible)
    grid.set_hotzone(center=(27, 10), radius=3, probability_boost=2.0)

    # Misplaced item
    grid.place_target((22, 15))

    # Warehouse robots
    robots = [
        GridAgent(0, start_position=(0, 0)),
        GridAgent(1, start_position=(0, 19)),
        GridAgent(2, start_position=(29, 0)),
        GridAgent(3, start_position=(29, 19)),
    ]

    strategies = [
        ProbabilisticGridStrategy(exploitation_factor=0.85),  # Check receiving
        QuadrantStrategy(num_agents=4, agent_idx=1),
        QuadrantStrategy(num_agents=4, agent_idx=2),
        ProbabilisticGridStrategy(exploitation_factor=0.85),  # Check shipping
    ]

    print(f"\nDeploying {len(robots)} warehouse robots:")
    print("  - Robot 1: Priority scan receiving area")
    print("  - Robot 2: Zone B coverage")
    print("  - Robot 3: Zone C coverage")
    print("  - Robot 4: Priority scan shipping area")

    frames, found, elapsed = run_grid_simulation(
        grid, robots, strategies, time_budget=50.0
    )

    print(f"\nResult: {'ITEM LOCATED!' if found else 'Search continues...'}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Locations scanned: {grid.count_explored()}")

    animator = Grid2DAnimator(grid, robots, show_trails=True, figsize=(14, 10))
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_warehouse.gif"),
        fps=10,
        title="Warehouse: Misplaced Inventory Search"
    )

    return found, elapsed


# =============================================================================
# USE CASE 4: DRONE SWARM SURVEILLANCE
# =============================================================================

def create_drone_swarm_scenario():
    """
    Drone Swarm Surveillance Scenario.

    Simulates:
    - Geographic area to monitor
    - Target (intruder/anomaly)
    - Heterogeneous drone swarm
    - Battery/daylight constraint
    """
    print("\n" + "=" * 60)
    print("USE CASE: DRONE SWARM SURVEILLANCE")
    print("=" * 60)
    print("Scenario: Agricultural field anomaly detection")
    print("Time constraint: Battery life")

    # Field area (35x35)
    grid = Grid2D(35, 35, MovementType.EIGHT_WAY)

    # Some areas with obstacles (buildings, trees)
    for _ in range(20):
        x, y = random.randint(0, 34), random.randint(0, 34)
        grid.cells[(x, y)].is_obstacle = True
        grid.status[(x, y)] = GridCellStatus.OBSTACLE

    # Previous anomaly hotspot
    grid.set_hotzone(center=(25, 25), radius=8, probability_boost=4.0)

    # Current anomaly location
    grid.place_target((28, 22))

    # Drone swarm (6 drones with different capabilities)
    drones = [
        GridAgent(0, start_position=(0, 0)),      # Visual drone
        GridAgent(1, start_position=(34, 0)),     # Thermal drone
        GridAgent(2, start_position=(0, 34)),     # Multispectral
        GridAgent(3, start_position=(34, 34)),    # LIDAR
        GridAgent(4, start_position=(17, 0)),     # Scout (fast)
        GridAgent(5, start_position=(17, 34)),    # Scout (fast)
    ]
    drones[4].speed = 1.5  # Scouts faster
    drones[5].speed = 1.5

    strategies = [
        SpiralStrategy(),                         # Visual - spiral pattern
        SwarmStrategy(repulsion_strength=2.5),    # Thermal - spread out
        QuadrantStrategy(num_agents=6, agent_idx=2),  # Multispectral - zone
        ProbabilisticGridStrategy(exploitation_factor=0.9),  # LIDAR - hotspot
        GreedyGridStrategy(),                     # Scout 1 - fast coverage
        GreedyGridStrategy(),                     # Scout 2 - fast coverage
    ]

    print(f"\nDeploying {len(drones)} drones:")
    print("  - Visual Drone (spiral pattern)")
    print("  - Thermal Drone (swarm coordination)")
    print("  - Multispectral (zone coverage)")
    print("  - LIDAR Drone (probability-based)")
    print("  - Scout Drones x2 (fast greedy)")

    frames, found, elapsed = run_grid_simulation(
        grid, drones, strategies, time_budget=80.0
    )

    print(f"\nResult: {'ANOMALY DETECTED!' if found else 'Continuing surveillance...'}")
    print(f"Flight time: {elapsed:.1f}s")
    print(f"Area covered: {grid.count_explored()} cells")

    animator = Grid2DAnimator(grid, drones, show_trails=True, figsize=(12, 12))
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_drone_swarm.gif"),
        fps=12,
        title="Drone Swarm: Agricultural Surveillance"
    )

    return found, elapsed


# =============================================================================
# USE CASE 5: DISTRIBUTED CODE DEBUGGING
# =============================================================================

def create_debugging_scenario():
    """
    Distributed Code Debugging Scenario.

    Simulates:
    - Codebase/execution space as grid
    - Bug location (target)
    - Diagnostic agents
    - SLA resolution time
    """
    print("\n" + "=" * 60)
    print("USE CASE: DISTRIBUTED CODE DEBUGGING")
    print("=" * 60)
    print("Scenario: Finding root cause in distributed system")
    print("Time constraint: Production SLA")

    # Codebase "topology" (20x20 = modules/services)
    grid = Grid2D(20, 20, MovementType.FOUR_WAY)

    # Some modules harder to analyze
    for x in range(0, 8):
        for y in range(0, 8):
            grid.cells[(x, y)].search_cost = 0.25  # Legacy code
            grid.cells[(x, y)].metadata['type'] = 'legacy'

    # Microservices (faster to analyze)
    for x in range(12, 20):
        for y in range(12, 20):
            grid.cells[(x, y)].search_cost = 0.08
            grid.cells[(x, y)].metadata['type'] = 'microservice'

    # Error logs suggest this area
    grid.set_hotzone(center=(10, 10), radius=5, probability_boost=3.0)

    # Actual bug location
    grid.place_target((8, 12))

    # Diagnostic agents
    agents = [
        GridAgent(0, start_position=(0, 0)),      # Log analyzer
        GridAgent(1, start_position=(19, 19)),    # Profiler
        GridAgent(2, start_position=(10, 10)),    # Tracer (starts at error)
        GridAgent(3, start_position=(0, 19)),     # Static analyzer
    ]

    strategies = [
        WavefrontStrategy(),                          # Log analyzer - systematic
        SwarmStrategy(repulsion_strength=1.5),        # Profiler - spread
        ProbabilisticGridStrategy(exploitation_factor=0.95),  # Tracer - follow logs
        GreedyGridStrategy(),                         # Static - quick scan
    ]

    print(f"\nDeploying {len(agents)} diagnostic agents:")
    print("  - Log Analyzer (wavefront)")
    print("  - Profiler (swarm)")
    print("  - Distributed Tracer (probabilistic)")
    print("  - Static Analyzer (greedy)")

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=35.0
    )

    print(f"\nResult: {'BUG FOUND!' if found else 'Investigation continues...'}")
    print(f"Debug time: {elapsed:.1f}s")
    print(f"Modules analyzed: {grid.count_explored()}")

    animator = Grid2DAnimator(grid, agents, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_debugging.gif"),
        fps=10,
        title="Distributed Debugging: Root Cause Analysis"
    )

    return found, elapsed


# =============================================================================
# USE CASE 6: MEDICAL DIAGNOSIS
# =============================================================================

def create_medical_diagnosis_scenario():
    """
    Medical Diagnosis Under Time Pressure.

    Simulates:
    - Symptom/test space as grid
    - Correct diagnosis (target)
    - Diagnostic tests as agents
    - Patient deterioration window
    """
    print("\n" + "=" * 60)
    print("USE CASE: MEDICAL DIAGNOSIS")
    print("=" * 60)
    print("Scenario: Emergency differential diagnosis")
    print("Time constraint: Patient window")

    # Diagnosis space (15x15 = possible conditions)
    grid = Grid2D(15, 15, MovementType.EIGHT_WAY)

    # Some conditions require more tests
    for x in range(10, 15):
        for y in range(10, 15):
            grid.cells[(x, y)].search_cost = 0.3  # Rare conditions
            grid.cells[(x, y)].metadata['type'] = 'rare'

    # Initial symptoms suggest this area
    grid.set_hotzone(center=(7, 7), radius=4, probability_boost=4.0)

    # Actual condition
    grid.place_target((9, 6))

    # Diagnostic "tests"
    tests = [
        GridAgent(0, start_position=(0, 0)),      # Blood panel
        GridAgent(1, start_position=(14, 14)),    # Imaging
        GridAgent(2, start_position=(7, 7)),      # Physical exam
        GridAgent(3, start_position=(0, 14)),     # Specialist consult
    ]

    strategies = [
        WavefrontStrategy(),                          # Blood panel - systematic
        GreedyGridStrategy(),                         # Imaging - targeted
        ProbabilisticGridStrategy(exploitation_factor=0.9),  # Physical - symptoms
        SwarmStrategy(repulsion_strength=1.5),        # Specialist - coverage
    ]

    print(f"\nRunning {len(tests)} diagnostic approaches:")
    print("  - Blood Panel (systematic)")
    print("  - Imaging Studies (targeted)")
    print("  - Physical Examination (symptom-guided)")
    print("  - Specialist Consultation (broad)")

    frames, found, elapsed = run_grid_simulation(
        grid, tests, strategies, time_budget=25.0
    )

    print(f"\nResult: {'DIAGNOSIS CONFIRMED!' if found else 'More tests needed...'}")
    print(f"Time to diagnosis: {elapsed:.1f}s")
    print(f"Conditions ruled out: {grid.count_explored()}")

    animator = Grid2DAnimator(grid, tests, show_trails=True, figsize=(10, 10))
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "usecase_medical.gif"),
        fps=8,
        title="Medical Diagnosis: Emergency Differential"
    )

    return found, elapsed


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all use case demonstrations."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MULTI-AGENT SEARCH FRAMEWORK - USE CASE EXPLORER")
    print("=" * 60)

    results = []

    # Fixed large scale
    results.append(("Large Scale (Fixed)", *create_large_scale_success()))

    # Use cases
    results.append(("Search & Rescue", *create_sar_scenario()))
    results.append(("Threat Hunting", *create_threat_hunting_scenario()))
    results.append(("Warehouse Search", *create_warehouse_scenario()))
    results.append(("Drone Swarm", *create_drone_swarm_scenario()))
    results.append(("Code Debugging", *create_debugging_scenario()))
    results.append(("Medical Diagnosis", *create_medical_diagnosis_scenario()))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, found, elapsed in results:
        status = "SUCCESS" if found else "TIMEOUT"
        print(f"  {name:25} - {status:8} - {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"All animations saved to: {output_dir}/")
    print("\nGenerated use case files:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('usecase_') or f.startswith('grid_large'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
