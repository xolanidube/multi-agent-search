"""
Create 2D Grid Search Animations.

Generates animated visualizations of multi-agent search on 2D grids
with various scenarios:
- Simple grid exploration
- Maze navigation
- Warehouse layout
- Multi-agent coordination
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.grid import Grid2D, GridAgent, GridCellStatus, MovementType
from src.strategies.grid_strategies import (
    GreedyGridStrategy, SpiralStrategy, QuadrantStrategy,
    SwarmStrategy, WavefrontStrategy, RandomGridStrategy,
    ProbabilisticGridStrategy
)
from src.visualization.grid_animator import Grid2DAnimator, run_grid_simulation


def create_simple_grid_animation():
    """Simple 15x15 grid with single agent."""
    print("1. Simple Grid - Single Agent (Greedy)...")

    grid = Grid2D(15, 15, MovementType.FOUR_WAY)
    grid.place_target((12, 10))

    agent = GridAgent(0, start_position=(0, 0))
    strategy = GreedyGridStrategy()

    frames, found, elapsed = run_grid_simulation(
        grid, [agent], [strategy], time_budget=50.0
    )

    animator = Grid2DAnimator(grid, [agent], show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    os.makedirs(output_dir, exist_ok=True)

    animator.create_animation(
        os.path.join(output_dir, "grid_simple.gif"),
        fps=8,
        title="Simple Grid Search (Greedy Strategy)"
    )

    return found, elapsed


def create_spiral_animation():
    """Spiral search pattern from center."""
    print("2. Spiral Search Pattern...")

    grid = Grid2D(20, 20, MovementType.FOUR_WAY)
    grid.place_target((5, 15))  # Hidden in corner

    # Start from center
    agent = GridAgent(0, start_position=(10, 10))
    strategy = SpiralStrategy()

    frames, found, elapsed = run_grid_simulation(
        grid, [agent], [strategy], time_budget=100.0
    )

    animator = Grid2DAnimator(grid, [agent], show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_spiral.gif"),
        fps=10,
        title="Spiral Search Pattern"
    )

    return found, elapsed


def create_maze_animation():
    """Navigation through a maze."""
    print("3. Maze Navigation...")

    grid = Grid2D.generate_maze(20, 20, complexity=0.15)
    grid.place_target((18, 18))

    agent = GridAgent(0, start_position=(1, 1))
    strategy = WavefrontStrategy()

    frames, found, elapsed = run_grid_simulation(
        grid, [agent], [strategy], time_budget=100.0
    )

    animator = Grid2DAnimator(grid, [agent], show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_maze.gif"),
        fps=10,
        title="Maze Navigation (Wavefront Strategy)"
    )

    return found, elapsed


def create_multi_agent_animation():
    """Multiple agents with quadrant strategy."""
    print("4. Multi-Agent Quadrant Search (4 agents)...")

    grid = Grid2D(20, 20, MovementType.FOUR_WAY)
    grid.place_target((15, 15))

    # Four agents, each starting in a corner
    agents = [
        GridAgent(0, start_position=(0, 0)),
        GridAgent(1, start_position=(19, 0)),
        GridAgent(2, start_position=(0, 19)),
        GridAgent(3, start_position=(19, 19)),
    ]

    strategies = [
        QuadrantStrategy(num_agents=4, agent_idx=i) for i in range(4)
    ]

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=50.0
    )

    animator = Grid2DAnimator(grid, agents, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_multi_agent.gif"),
        fps=10,
        title="Multi-Agent Quadrant Search (4 Agents)"
    )

    return found, elapsed


def create_swarm_animation():
    """Swarm coordination with repulsion."""
    print("5. Swarm Coordination (3 agents)...")

    grid = Grid2D(25, 25, MovementType.EIGHT_WAY)
    grid.place_target((20, 5))

    agents = [
        GridAgent(0, start_position=(0, 0)),
        GridAgent(1, start_position=(0, 12)),
        GridAgent(2, start_position=(0, 24)),
    ]

    strategies = [SwarmStrategy(repulsion_strength=2.0) for _ in range(3)]

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=80.0
    )

    animator = Grid2DAnimator(grid, agents, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_swarm.gif"),
        fps=10,
        title="Swarm Coordination (Repulsion-based)"
    )

    return found, elapsed


def create_warehouse_animation():
    """Warehouse-like room structure."""
    print("6. Warehouse Search (Room Structure)...")

    grid = Grid2D.generate_rooms(24, 24, room_count=9)
    grid.place_target((20, 20))

    agents = [
        GridAgent(0, start_position=(1, 1)),
        GridAgent(1, start_position=(12, 1)),
    ]

    strategies = [
        GreedyGridStrategy(),
        WavefrontStrategy(),
    ]

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=100.0
    )

    animator = Grid2DAnimator(grid, agents, show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_warehouse.gif"),
        fps=10,
        title="Warehouse Search (2 Agents, Room Layout)"
    )

    return found, elapsed


def create_probabilistic_animation():
    """Probabilistic search with hotzone."""
    print("7. Probabilistic Search (Hotzone)...")

    grid = Grid2D(20, 20, MovementType.FOUR_WAY)

    # Create hotzone in upper-right
    grid.set_hotzone(center=(15, 5), radius=5, probability_boost=5.0)
    grid.place_target((16, 4))  # Place target in hotzone

    agent = GridAgent(0, start_position=(0, 10))
    strategy = ProbabilisticGridStrategy(exploitation_factor=0.85)

    frames, found, elapsed = run_grid_simulation(
        grid, [agent], [strategy], time_budget=60.0
    )

    animator = Grid2DAnimator(grid, [agent], show_trails=True)
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_probabilistic.gif"),
        fps=10,
        title="Probabilistic Search (Hotzone Priority)"
    )

    return found, elapsed


def create_large_scale_animation():
    """Large grid with many agents."""
    print("8. Large Scale (40x40, 8 agents)...")

    grid = Grid2D(40, 40, MovementType.EIGHT_WAY)
    grid.add_obstacles(density=0.1)
    grid.place_target((35, 35))

    # 8 agents distributed around edges
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

    strategies = [SwarmStrategy(repulsion_strength=3.0) for _ in range(8)]

    frames, found, elapsed = run_grid_simulation(
        grid, agents, strategies, time_budget=150.0
    )

    animator = Grid2DAnimator(grid, agents, show_trails=True, figsize=(14, 12))
    animator.frames = frames

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    animator.create_animation(
        os.path.join(output_dir, "grid_large_scale.gif"),
        fps=12,
        title="Large Scale Search (40x40, 8 Agents, 10% Obstacles)"
    )

    return found, elapsed


def main():
    """Generate all 2D grid animations."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'animations')
    os.makedirs(output_dir, exist_ok=True)

    print("Creating 2D Grid Search Animations...")
    print("=" * 60)

    results = []

    # Run all animations
    results.append(("Simple Grid", *create_simple_grid_animation()))
    results.append(("Spiral", *create_spiral_animation()))
    results.append(("Maze", *create_maze_animation()))
    results.append(("Multi-Agent", *create_multi_agent_animation()))
    results.append(("Swarm", *create_swarm_animation()))
    results.append(("Warehouse", *create_warehouse_animation()))
    results.append(("Probabilistic", *create_probabilistic_animation()))
    results.append(("Large Scale", *create_large_scale_animation()))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("-" * 60)
    for name, found, elapsed in results:
        status = "FOUND" if found else "NOT FOUND"
        print(f"  {name:20} - {status:10} - Time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print(f"All 2D grid animations saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('grid_') and f.endswith('.gif'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
