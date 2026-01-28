#!/usr/bin/env python3
"""
Demo script for the Multi-Agent Search Simulation.

This script demonstrates the complete pipeline:
1. Generate a heterogeneous forest path
2. Benchmark different search strategies
3. Select optimal strategy based on benchmarks
4. Execute timed multi-agent search
5. Visualize results

Usage:
    python run_simulation.py [--no-viz] [--searchers N] [--time T] [--path-length L]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.orchestrator import SearchOrchestrator

# Try to import visualization (optional)
try:
    from src.visualization.animator import SearchAnimator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    SearchAnimator = None
    MATPLOTLIB_AVAILABLE = False


def run_demo(
    path_length: int = 50,
    num_searchers: int = 3,
    time_budget: float = 30.0,
    visualize: bool = True,
    seed: int = 42
):
    """
    Run a demonstration of the multi-agent search system.

    Args:
        path_length: Number of cells in the path
        num_searchers: Number of search agents
        time_budget: Total time in seconds
        visualize: Whether to show visualization
        seed: Random seed for reproducibility
    """
    print("\n" + "=" * 70)
    print("  TIME-CONSTRAINED MULTI-AGENT SEARCH DEMO")
    print("  with Online Strategy Selection via Stigmergic Coordination")
    print("=" * 70)

    # Create orchestrator
    orchestrator = SearchOrchestrator(
        path_length=path_length,
        num_searchers=num_searchers,
        time_budget=time_budget,
        heterogeneous_terrain=True,
        seed=seed
    )

    # Set up visualization if enabled
    animator = None
    if visualize:
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Running without visualization.")
            print("Install with: pip install matplotlib")
            visualize = False
        else:
            animator = SearchAnimator(
                orchestrator.get_path(),
                orchestrator.get_searchers(),
                orchestrator.get_marks()
            )

            # Record initial state
            animator.record_frame(0, time_budget, False)

            # Hook to record frames during search
            def on_step(searcher_id, cell_id, found):
                elapsed = orchestrator.time_manager.elapsed()
                animator.record_frame(elapsed, time_budget, found)

            orchestrator.on_search_step = on_step

    # Run the search
    results = orchestrator.run(sample_ratio=0.15, verbose=True)

    # Show visualization
    if visualize and animator:
        print("\nShowing final state visualization...")
        animator.show_final_state(results)

    return results


def run_comparison(
    num_runs: int = 10,
    path_length: int = 50,
    num_searchers: int = 3,
    time_budget: float = 30.0
):
    """
    Run multiple simulations to compare strategies.

    Args:
        num_runs: Number of runs per configuration
        path_length: Path length
        num_searchers: Number of searchers
        time_budget: Time budget
    """
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"  Runs per config: {num_runs}")
    print(f"  Path length: {path_length}")
    print(f"  Searchers: {num_searchers}")
    print(f"  Time budget: {time_budget}s")
    print("=" * 70)

    # Track results
    results_summary = {
        'with_benchmarking': {'successes': 0, 'total_time': 0},
        'greedy_only': {'successes': 0, 'total_time': 0},
        'random_only': {'successes': 0, 'total_time': 0},
    }

    for run in range(num_runs):
        seed = 100 + run

        # With benchmarking (full pipeline)
        orch = SearchOrchestrator(
            path_length=path_length,
            num_searchers=num_searchers,
            time_budget=time_budget,
            seed=seed
        )
        result = orch.run(sample_ratio=0.15, verbose=False)
        if result['success']:
            results_summary['with_benchmarking']['successes'] += 1
            results_summary['with_benchmarking']['total_time'] += result['time_used']

        # Greedy only (no benchmarking)
        from src.strategies.greedy_nearest import GreedyNearestStrategy
        from src.engine.simulation import SimulationRunner
        runner = SimulationRunner(path_length, seed)
        metrics = runner.run_single(
            num_searchers, time_budget,
            [GreedyNearestStrategy()]
        )
        if metrics.success:
            results_summary['greedy_only']['successes'] += 1
            results_summary['greedy_only']['total_time'] += metrics.time_used

        # Random only
        from src.strategies.random_walk import RandomWalkStrategy
        metrics = runner.run_single(
            num_searchers, time_budget,
            [RandomWalkStrategy()]
        )
        if metrics.success:
            results_summary['random_only']['successes'] += 1
            results_summary['random_only']['total_time'] += metrics.time_used

    # Print results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)

    for approach, data in results_summary.items():
        success_rate = data['successes'] / num_runs * 100
        avg_time = data['total_time'] / max(1, data['successes'])
        print(f"{approach:20}: {success_rate:5.1f}% success, "
              f"{avg_time:5.1f}s avg time when successful")

    print("-" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent Search Simulation Demo'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--searchers', '-s', type=int, default=3,
        help='Number of searchers (default: 3)'
    )
    parser.add_argument(
        '--time', '-t', type=float, default=30.0,
        help='Time budget in seconds (default: 30)'
    )
    parser.add_argument(
        '--path-length', '-l', type=int, default=50,
        help='Path length in cells (default: 50)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Run strategy comparison experiment'
    )
    parser.add_argument(
        '--runs', type=int, default=10,
        help='Number of runs for comparison (default: 10)'
    )

    args = parser.parse_args()

    if args.compare:
        run_comparison(
            num_runs=args.runs,
            path_length=args.path_length,
            num_searchers=args.searchers,
            time_budget=args.time
        )
    else:
        run_demo(
            path_length=args.path_length,
            num_searchers=args.searchers,
            time_budget=args.time,
            visualize=not args.no_viz,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
