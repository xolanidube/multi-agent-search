#!/usr/bin/env python3
"""
Research Experiment Runner.

Runs comprehensive experiments for the research paper:
1. Strategy selection effectiveness study
2. Optimal sampling ratio study
3. Scaling analysis study
4. Terrain heterogeneity study

Usage:
    python run_experiments.py --study all
    python run_experiments.py --study sampling --runs 50
    python run_experiments.py --study scaling --runs 30
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.experiments import ExperimentRunner, ExperimentConfig, ParameterSweep
from src.research.metrics import MetricsCollector


def run_sampling_ratio_study(runner: ExperimentRunner, runs: int = 50):
    """Study the effect of sampling ratio on performance."""
    print("\n" + "=" * 60)
    print("STUDY 1: Sampling Ratio Analysis")
    print("=" * 60)

    ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    results = runner.run_study_sampling_ratio(
        ratios=ratios,
        runs_per_ratio=runs,
        base_config=ExperimentConfig(
            path_length=50,
            num_searchers=3,
            time_budget=30.0,
            heterogeneous_terrain=True
        )
    )

    print("\n--- Sampling Ratio Results ---")
    print(f"{'Ratio':>8} | {'Success':>8} | {'Coverage':>10} | {'Efficiency':>10}")
    print("-" * 45)
    for ratio, data in sorted(results.items()):
        print(f"{ratio:>7.0%} | {data['success_rate']:>7.1%} | "
              f"{data['avg_coverage']:>9.1%} | {data['avg_time_efficiency']:>9.1%}")

    return results


def run_scaling_study(runner: ExperimentRunner, runs: int = 30):
    """Study how performance scales with number of searchers."""
    print("\n" + "=" * 60)
    print("STUDY 2: Scaling Analysis")
    print("=" * 60)

    searcher_counts = [1, 2, 3, 5, 7, 10]

    results = runner.run_study_scaling(
        searcher_counts=searcher_counts,
        runs_per_count=runs,
        base_config=ExperimentConfig(
            path_length=50,
            time_budget=30.0,
            sample_ratio=0.15,
            heterogeneous_terrain=True
        )
    )

    print("\n--- Scaling Results ---")
    print(f"{'Searchers':>10} | {'Success':>8} | {'Coverage':>10} | {'Avg Time':>10} | {'Cells/Agent':>12}")
    print("-" * 60)
    for count, data in sorted(results.items()):
        print(f"{count:>10} | {data['success_rate']:>7.1%} | "
              f"{data['avg_coverage']:>9.1%} | {data['avg_time']:>9.2f}s | "
              f"{data['cells_per_searcher']:>11.1f}")

    return results


def run_terrain_study(runner: ExperimentRunner, runs: int = 50):
    """Study effect of terrain heterogeneity."""
    print("\n" + "=" * 60)
    print("STUDY 3: Terrain Heterogeneity Analysis")
    print("=" * 60)

    results = {'homogeneous': [], 'heterogeneous': []}

    for hetero in [False, True]:
        terrain_type = 'heterogeneous' if hetero else 'homogeneous'
        print(f"\nTesting {terrain_type} terrain...")

        for run in range(runs):
            config = ExperimentConfig(
                path_length=50,
                num_searchers=3,
                time_budget=30.0,
                sample_ratio=0.15,
                heterogeneous_terrain=hetero,
                seed=3000 + run
            )
            result = runner.run_single(config)
            results[terrain_type].append(result)

    print("\n--- Terrain Results ---")
    print(f"{'Terrain':>15} | {'Success':>8} | {'Coverage':>10} | {'Avg Time':>10}")
    print("-" * 50)

    for terrain_type, terrain_results in results.items():
        successes = [r for r in terrain_results if r.success]
        success_rate = len(successes) / len(terrain_results)
        avg_coverage = sum(r.coverage_rate for r in terrain_results) / len(terrain_results)
        avg_time = sum(r.total_time_used for r in successes) / len(successes) if successes else 0

        print(f"{terrain_type:>15} | {success_rate:>7.1%} | "
              f"{avg_coverage:>9.1%} | {avg_time:>9.2f}s")

    return results


def run_strategy_comparison(runner: ExperimentRunner, runs: int = 50):
    """Compare performance with vs without strategy selection."""
    print("\n" + "=" * 60)
    print("STUDY 4: Strategy Selection Effectiveness")
    print("=" * 60)

    results = {
        'with_selection': [],
        'greedy_only': [],
        'random_only': []
    }

    from src.strategies.greedy_nearest import GreedyNearestStrategy
    from src.strategies.random_walk import RandomWalkStrategy
    from src.engine.simulation import SimulationRunner

    for run in range(runs):
        seed = 4000 + run

        # With strategy selection (full pipeline)
        config = ExperimentConfig(
            path_length=50,
            num_searchers=3,
            time_budget=30.0,
            sample_ratio=0.15,
            heterogeneous_terrain=True,
            seed=seed
        )
        result = runner.run_single(config)
        results['with_selection'].append(result)

        # Greedy only (no benchmarking)
        sim_runner = SimulationRunner(50, seed)
        metrics = sim_runner.run_single(
            num_searchers=3,
            time_budget=30.0,
            strategies=[GreedyNearestStrategy()],
            heterogeneous_terrain=True
        )
        results['greedy_only'].append(metrics)

        # Random only
        metrics = sim_runner.run_single(
            num_searchers=3,
            time_budget=30.0,
            strategies=[RandomWalkStrategy()],
            heterogeneous_terrain=True
        )
        results['random_only'].append(metrics)

    print("\n--- Strategy Comparison Results ---")
    print(f"{'Approach':>20} | {'Success':>8} | {'Coverage':>10}")
    print("-" * 45)

    for approach, approach_results in results.items():
        if approach == 'with_selection':
            successes = [r for r in approach_results if r.success]
            success_rate = len(successes) / len(approach_results)
            avg_coverage = sum(r.coverage_rate for r in approach_results) / len(approach_results)
        else:
            successes = [r for r in approach_results if r.success]
            success_rate = len(successes) / len(approach_results)
            avg_coverage = sum(r.coverage_rate for r in approach_results) / len(approach_results)

        print(f"{approach:>20} | {success_rate:>7.1%} | {avg_coverage:>9.1%}")

    return results


def run_full_sweep(runner: ExperimentRunner):
    """Run a comprehensive parameter sweep."""
    print("\n" + "=" * 60)
    print("FULL PARAMETER SWEEP")
    print("=" * 60)

    sweep = ParameterSweep(
        path_lengths=[30, 50, 100],
        num_searchers_range=[1, 2, 3, 5],
        time_budgets=[20.0, 30.0, 60.0],
        sample_ratios=[0.0, 0.10, 0.15, 0.20],
        heterogeneous_options=[True],
        runs_per_config=10
    )

    print(f"\nTotal experiments: {sweep.total_experiments()}")
    runner.run_parameter_sweep(sweep)


def main():
    parser = argparse.ArgumentParser(description='Run research experiments')
    parser.add_argument(
        '--study', '-s',
        choices=['sampling', 'scaling', 'terrain', 'strategy', 'sweep', 'all'],
        default='all',
        help='Which study to run'
    )
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=30,
        help='Number of runs per configuration'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create timestamp for this experiment run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"mas_study_{timestamp}"

    runner = ExperimentRunner(
        experiment_name=experiment_name,
        output_dir=args.output,
        verbose=True
    )

    print(f"\nExperiment: {experiment_name}")
    print(f"Output directory: {args.output}")
    print(f"Runs per config: {args.runs}")

    if args.study in ['sampling', 'all']:
        run_sampling_ratio_study(runner, args.runs)

    if args.study in ['scaling', 'all']:
        run_scaling_study(runner, args.runs)

    if args.study in ['terrain', 'all']:
        run_terrain_study(runner, args.runs)

    if args.study in ['strategy', 'all']:
        run_strategy_comparison(runner, args.runs)

    if args.study == 'sweep':
        run_full_sweep(runner)

    # Print final summary
    runner.get_metrics().print_summary()

    print(f"\nResults saved to {args.output}/")


if __name__ == '__main__':
    main()
