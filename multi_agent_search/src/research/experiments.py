"""
Experiment Runner for Research Studies.

Provides infrastructure for running controlled experiments:
- Configuration management
- Batch execution with parameter sweeps
- Statistical validity (multiple runs per configuration)
- Progress tracking and resumption
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from itertools import product
import time
import json
from pathlib import Path

from .metrics import MetricsCollector, ExperimentResult
from ..engine.orchestrator import SearchOrchestrator


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    path_length: int = 50
    num_searchers: int = 3
    time_budget: float = 30.0
    sample_ratio: float = 0.15
    heterogeneous_terrain: bool = True
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path_length': self.path_length,
            'num_searchers': self.num_searchers,
            'time_budget': self.time_budget,
            'sample_ratio': self.sample_ratio,
            'heterogeneous': self.heterogeneous_terrain,
            'seed': self.seed
        }


@dataclass
class ParameterSweep:
    """
    Defines a parameter sweep for experiments.

    Allows specifying ranges of values for each parameter
    to test systematically.
    """
    path_lengths: List[int] = field(default_factory=lambda: [50])
    num_searchers_range: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    time_budgets: List[float] = field(default_factory=lambda: [30.0])
    sample_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.15, 0.2])
    heterogeneous_options: List[bool] = field(default_factory=lambda: [True, False])
    runs_per_config: int = 10

    def generate_configs(self, base_seed: int = 1000) -> Iterator[ExperimentConfig]:
        """
        Generate all experiment configurations.

        Args:
            base_seed: Starting seed for reproducibility

        Yields:
            ExperimentConfig for each combination
        """
        config_id = 0
        for path_len, num_searchers, time_budget, sample_ratio, hetero in product(
            self.path_lengths,
            self.num_searchers_range,
            self.time_budgets,
            self.sample_ratios,
            self.heterogeneous_options
        ):
            for run in range(self.runs_per_config):
                yield ExperimentConfig(
                    path_length=path_len,
                    num_searchers=num_searchers,
                    time_budget=time_budget,
                    sample_ratio=sample_ratio,
                    heterogeneous_terrain=hetero,
                    seed=base_seed + config_id * self.runs_per_config + run
                )
                config_id += 1

    def total_experiments(self) -> int:
        """Calculate total number of experiments."""
        return (
            len(self.path_lengths) *
            len(self.num_searchers_range) *
            len(self.time_budgets) *
            len(self.sample_ratios) *
            len(self.heterogeneous_options) *
            self.runs_per_config
        )


class ExperimentRunner:
    """
    Runs experiments and collects metrics.

    Supports:
    - Single experiment execution
    - Batch execution with parameter sweeps
    - Progress tracking and reporting
    - Result aggregation and analysis
    """

    def __init__(
        self,
        experiment_name: str = "mas_experiment",
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the experiment runner.

        Args:
            experiment_name: Name for this experiment series
            output_dir: Directory for output files
            verbose: Whether to print progress
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        self.verbose = verbose
        self.metrics = MetricsCollector(experiment_name)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        config: ExperimentConfig,
        collect_detailed_metrics: bool = True
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            collect_detailed_metrics: Whether to collect detailed metrics

        Returns:
            ExperimentResult with all metrics
        """
        # Create and run orchestrator
        orchestrator = SearchOrchestrator(
            path_length=config.path_length,
            num_searchers=config.num_searchers,
            time_budget=config.time_budget,
            heterogeneous_terrain=config.heterogeneous_terrain,
            seed=config.seed
        )

        results = orchestrator.run(
            sample_ratio=config.sample_ratio,
            verbose=False
        )

        # Build configuration dict
        config_dict = config.to_dict()
        config_dict['key_position'] = orchestrator.path.key_position

        # Build outcome dict
        outcome = {
            'success': results['success'],
            'key_found_by': results.get('key_found_by'),
            'time_used': results['time_used'],
            'sampling_time': results.get('sample_time', 0),
            'execution_time': results.get('search_time', 0),
            'time_remaining': results['time_remaining'],
            'cells_searched': results['cells_searched'],
            'coverage_rate': results['coverage_rate'],
            'allocation': results.get('allocation', {})
        }

        # Build benchmark data
        benchmark_data = None
        if results.get('benchmark_scores'):
            benchmark_data = {
                'rankings': results['benchmark_scores'],
                'throughputs': results.get('benchmark_scores', {}),
                'selected': results.get('selected_strategy', ''),
                'confidence': 1.0,  # Could track from selector
                'method': 'optimal_selection',
                'time_used': results.get('sample_time', 0),
                'time_budget': config.time_budget * config.sample_ratio
            }

        # Build searcher data
        searcher_data = []
        for sid, stats in results.get('searcher_stats', {}).items():
            searcher_data.append({
                'id': sid,
                'strategy': stats.get('strategy', ''),
                'cells_searched': stats.get('cells_searched', 0),
                'distance_traveled': stats.get('distance_traveled', 0),
                'time_spent': stats.get('time_spent', 0),
                'throughput': stats.get('throughput', 0)
            })

        # Create result
        result = self.metrics.create_result(
            config=config_dict,
            outcome=outcome,
            benchmark_data=benchmark_data,
            searcher_data=searcher_data
        )

        return result

    def run_batch(
        self,
        configs: Iterator[ExperimentConfig],
        total: Optional[int] = None,
        checkpoint_interval: int = 100
    ) -> None:
        """
        Run a batch of experiments.

        Args:
            configs: Iterator of experiment configurations
            total: Total number of experiments (for progress)
            checkpoint_interval: Save results every N experiments
        """
        start_time = time.time()
        completed = 0

        if self.verbose and total:
            print(f"\nRunning {total} experiments...")
            print("=" * 50)

        for config in configs:
            self.run_single(config)
            completed += 1

            # Progress reporting
            if self.verbose and total and completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total} "
                      f"({completed/total*100:.1f}%) - "
                      f"ETA: {remaining:.0f}s")

            # Checkpoint
            if completed % checkpoint_interval == 0:
                self._save_checkpoint(completed)

        # Final save
        self._save_results()

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"\nCompleted {completed} experiments in {elapsed:.1f}s")
            self.metrics.print_summary()

    def run_parameter_sweep(
        self,
        sweep: ParameterSweep,
        base_seed: int = 1000
    ) -> None:
        """
        Run a full parameter sweep.

        Args:
            sweep: Parameter sweep configuration
            base_seed: Starting seed for reproducibility
        """
        total = sweep.total_experiments()
        configs = sweep.generate_configs(base_seed)
        self.run_batch(configs, total=total)

    def run_study_sampling_ratio(
        self,
        ratios: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        runs_per_ratio: int = 50,
        base_config: Optional[ExperimentConfig] = None
    ) -> Dict[float, Dict]:
        """
        Study the effect of sampling ratio on performance.

        Args:
            ratios: Sampling ratios to test
            runs_per_ratio: Number of runs per ratio
            base_config: Base configuration (uses defaults if None)

        Returns:
            Dictionary mapping ratio to aggregated results
        """
        if base_config is None:
            base_config = ExperimentConfig()

        results_by_ratio = {}

        for ratio in ratios:
            if self.verbose:
                print(f"\nTesting sampling ratio: {ratio:.1%}")

            ratio_results = []
            for run in range(runs_per_ratio):
                config = ExperimentConfig(
                    path_length=base_config.path_length,
                    num_searchers=base_config.num_searchers,
                    time_budget=base_config.time_budget,
                    sample_ratio=ratio,
                    heterogeneous_terrain=base_config.heterogeneous_terrain,
                    seed=1000 + run
                )
                result = self.run_single(config)
                ratio_results.append(result)

            # Aggregate
            successes = [r for r in ratio_results if r.success]
            results_by_ratio[ratio] = {
                'success_rate': len(successes) / len(ratio_results),
                'avg_coverage': sum(r.coverage_rate for r in ratio_results) / len(ratio_results),
                'avg_time_efficiency': (
                    sum(r.time_efficiency for r in successes) / len(successes)
                    if successes else 0
                )
            }

        return results_by_ratio

    def run_study_scaling(
        self,
        searcher_counts: List[int] = [1, 2, 3, 5, 7, 10],
        runs_per_count: int = 30,
        base_config: Optional[ExperimentConfig] = None
    ) -> Dict[int, Dict]:
        """
        Study how performance scales with number of searchers.

        Args:
            searcher_counts: Number of searchers to test
            runs_per_count: Number of runs per configuration
            base_config: Base configuration

        Returns:
            Dictionary mapping searcher count to aggregated results
        """
        if base_config is None:
            base_config = ExperimentConfig()

        results_by_count = {}

        for count in searcher_counts:
            if self.verbose:
                print(f"\nTesting {count} searcher(s)...")

            count_results = []
            for run in range(runs_per_count):
                config = ExperimentConfig(
                    path_length=base_config.path_length,
                    num_searchers=count,
                    time_budget=base_config.time_budget,
                    sample_ratio=base_config.sample_ratio,
                    heterogeneous_terrain=base_config.heterogeneous_terrain,
                    seed=2000 + count * 100 + run
                )
                result = self.run_single(config)
                count_results.append(result)

            # Aggregate
            successes = [r for r in count_results if r.success]
            results_by_count[count] = {
                'success_rate': len(successes) / len(count_results),
                'avg_coverage': sum(r.coverage_rate for r in count_results) / len(count_results),
                'avg_time': (
                    sum(r.total_time_used for r in successes) / len(successes)
                    if successes else 0
                ),
                'cells_per_searcher': (
                    sum(r.cells_searched for r in count_results) /
                    (len(count_results) * count)
                )
            }

        return results_by_count

    def _save_checkpoint(self, completed: int) -> None:
        """Save intermediate results."""
        checkpoint_path = self.output_dir / f"{self.experiment_name}_checkpoint_{completed}.json"
        self.metrics.export_json(str(checkpoint_path))

    def _save_results(self) -> None:
        """Save final results."""
        # Save JSON
        json_path = self.output_dir / f"{self.experiment_name}_results.json"
        self.metrics.export_json(str(json_path))

        # Save CSV
        csv_path = self.output_dir / f"{self.experiment_name}_results.csv"
        self.metrics.export_csv(str(csv_path))

        if self.verbose:
            print(f"\nResults saved to:")
            print(f"  JSON: {json_path}")
            print(f"  CSV: {csv_path}")

    def get_metrics(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self.metrics
