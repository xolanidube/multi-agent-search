"""
Comprehensive Metrics Collection for Research.

Tracks all relevant metrics for analyzing multi-agent search performance:
- Success rates across different configurations
- Time efficiency (sampling vs execution tradeoffs)
- Coverage patterns and redundancy
- Strategy performance comparisons
- Coordination overhead measurements
- Statistical analysis utilities
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import csv
import statistics
from pathlib import Path


@dataclass
class SearcherMetrics:
    """Metrics for a single searcher's performance."""
    searcher_id: int
    strategy_name: str
    cells_searched: int
    distance_traveled: float
    time_spent: float
    throughput: float  # cells per second

    @property
    def efficiency(self) -> float:
        """Distance efficiency: cells searched per unit distance."""
        if self.distance_traveled == 0:
            return 0.0
        return self.cells_searched / self.distance_traveled


@dataclass
class BenchmarkMetrics:
    """Metrics from the strategy benchmarking phase."""
    strategy_rankings: Dict[str, float]  # strategy -> score
    throughput_predictions: Dict[str, float]  # strategy -> predicted throughput
    selected_strategy: str
    selection_confidence: float
    selection_method: str  # 'optimal_selection' or 'fallback_fastest'
    sampling_time_used: float
    sampling_time_budget: float


@dataclass
class CoordinationMetrics:
    """Metrics measuring coordination effectiveness."""
    total_searches: int
    redundant_searches: int  # cells searched more than once
    mark_read_operations: int
    mark_write_operations: int
    collision_events: int  # times multiple searchers targeted same cell

    @property
    def redundancy_rate(self) -> float:
        """Fraction of searches that were redundant."""
        if self.total_searches == 0:
            return 0.0
        return self.redundant_searches / self.total_searches

    @property
    def coordination_overhead(self) -> float:
        """Ratio of coordination operations to search operations."""
        if self.total_searches == 0:
            return 0.0
        return (self.mark_read_operations + self.mark_write_operations) / self.total_searches


@dataclass
class ExperimentResult:
    """Complete result from a single experiment run."""
    # Identification
    experiment_id: str
    timestamp: str
    seed: int

    # Configuration
    path_length: int
    num_searchers: int
    time_budget: float
    sample_ratio: float
    heterogeneous_terrain: bool
    key_position: int

    # Outcome
    success: bool
    key_found_by: Optional[int]

    # Time metrics
    total_time_used: float
    sampling_time: float
    execution_time: float
    time_remaining: float

    # Coverage metrics
    cells_searched: int
    coverage_rate: float

    # Strategy metrics
    benchmark_metrics: Optional[BenchmarkMetrics] = None
    strategy_allocation: Dict[str, int] = field(default_factory=dict)

    # Coordination metrics
    coordination_metrics: Optional[CoordinationMetrics] = None

    # Per-searcher metrics
    searcher_metrics: List[SearcherMetrics] = field(default_factory=list)

    # Derived metrics
    @property
    def time_efficiency(self) -> float:
        """Fraction of time remaining when key found (0 if failed)."""
        if not self.success:
            return 0.0
        return self.time_remaining / self.time_budget

    @property
    def search_efficiency(self) -> float:
        """Cells searched per second during execution."""
        if self.execution_time == 0:
            return 0.0
        return self.cells_searched / self.execution_time

    @property
    def sampling_roi(self) -> float:
        """
        Return on investment for sampling time.
        Compares actual performance to expected random performance.
        """
        # Theoretical: random strategy would find key at ~50% coverage on average
        # Better strategies find it faster
        if not self.success:
            return -1.0  # Sampling didn't help enough

        expected_coverage_random = 0.5
        if self.coverage_rate < expected_coverage_random:
            return (expected_coverage_random - self.coverage_rate) / self.sample_ratio
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Handle nested dataclasses
        if self.benchmark_metrics:
            data['benchmark_metrics'] = asdict(self.benchmark_metrics)
        if self.coordination_metrics:
            data['coordination_metrics'] = asdict(self.coordination_metrics)
        data['searcher_metrics'] = [asdict(s) for s in self.searcher_metrics]
        return data


class MetricsCollector:
    """
    Collects and aggregates metrics across multiple experiment runs.

    Provides:
    - Real-time metrics collection during simulation
    - Aggregation across multiple runs
    - Statistical analysis (mean, std, confidence intervals)
    - Export to CSV/JSON for further analysis
    """

    def __init__(self, experiment_name: str = "multi_agent_search"):
        """
        Initialize the metrics collector.

        Args:
            experiment_name: Name for this experiment series
        """
        self.experiment_name = experiment_name
        self.results: List[ExperimentResult] = []
        self.run_counter = 0

    def create_result(
        self,
        config: Dict[str, Any],
        outcome: Dict[str, Any],
        benchmark_data: Optional[Dict] = None,
        coordination_data: Optional[Dict] = None,
        searcher_data: Optional[List[Dict]] = None
    ) -> ExperimentResult:
        """
        Create an ExperimentResult from simulation data.

        Args:
            config: Configuration parameters
            outcome: Simulation outcome data
            benchmark_data: Benchmarking phase data
            coordination_data: Coordination metrics
            searcher_data: Per-searcher metrics

        Returns:
            ExperimentResult instance
        """
        self.run_counter += 1

        # Build benchmark metrics if available
        benchmark_metrics = None
        if benchmark_data:
            benchmark_metrics = BenchmarkMetrics(
                strategy_rankings=benchmark_data.get('rankings', {}),
                throughput_predictions=benchmark_data.get('throughputs', {}),
                selected_strategy=benchmark_data.get('selected', ''),
                selection_confidence=benchmark_data.get('confidence', 0.0),
                selection_method=benchmark_data.get('method', ''),
                sampling_time_used=benchmark_data.get('time_used', 0.0),
                sampling_time_budget=benchmark_data.get('time_budget', 0.0)
            )

        # Build coordination metrics if available
        coordination_metrics = None
        if coordination_data:
            coordination_metrics = CoordinationMetrics(
                total_searches=coordination_data.get('total_searches', 0),
                redundant_searches=coordination_data.get('redundant', 0),
                mark_read_operations=coordination_data.get('mark_reads', 0),
                mark_write_operations=coordination_data.get('mark_writes', 0),
                collision_events=coordination_data.get('collisions', 0)
            )

        # Build searcher metrics
        searcher_metrics = []
        if searcher_data:
            for s in searcher_data:
                searcher_metrics.append(SearcherMetrics(
                    searcher_id=s.get('id', 0),
                    strategy_name=s.get('strategy', ''),
                    cells_searched=s.get('cells_searched', 0),
                    distance_traveled=s.get('distance_traveled', 0.0),
                    time_spent=s.get('time_spent', 0.0),
                    throughput=s.get('throughput', 0.0)
                ))

        result = ExperimentResult(
            experiment_id=f"{self.experiment_name}_{self.run_counter:04d}",
            timestamp=datetime.now().isoformat(),
            seed=config.get('seed', 0),
            path_length=config.get('path_length', 0),
            num_searchers=config.get('num_searchers', 0),
            time_budget=config.get('time_budget', 0.0),
            sample_ratio=config.get('sample_ratio', 0.0),
            heterogeneous_terrain=config.get('heterogeneous', False),
            key_position=config.get('key_position', 0),
            success=outcome.get('success', False),
            key_found_by=outcome.get('key_found_by'),
            total_time_used=outcome.get('time_used', 0.0),
            sampling_time=outcome.get('sampling_time', 0.0),
            execution_time=outcome.get('execution_time', 0.0),
            time_remaining=outcome.get('time_remaining', 0.0),
            cells_searched=outcome.get('cells_searched', 0),
            coverage_rate=outcome.get('coverage_rate', 0.0),
            benchmark_metrics=benchmark_metrics,
            strategy_allocation=outcome.get('allocation', {}),
            coordination_metrics=coordination_metrics,
            searcher_metrics=searcher_metrics
        )

        self.results.append(result)
        return result

    def add_result(self, result: ExperimentResult) -> None:
        """Add a pre-built result to the collection."""
        self.results.append(result)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all results.

        Returns:
            Dictionary with statistical summaries
        """
        if not self.results:
            return {}

        successes = [r for r in self.results if r.success]
        failures = [r for r in self.results if not r.success]

        stats = {
            'total_runs': len(self.results),
            'successes': len(successes),
            'failures': len(failures),
            'success_rate': len(successes) / len(self.results),
        }

        # Time metrics
        if successes:
            times = [r.total_time_used for r in successes]
            stats['avg_time_to_success'] = statistics.mean(times)
            stats['std_time_to_success'] = statistics.stdev(times) if len(times) > 1 else 0

            efficiencies = [r.time_efficiency for r in successes]
            stats['avg_time_efficiency'] = statistics.mean(efficiencies)

        # Coverage metrics
        coverages = [r.coverage_rate for r in self.results]
        stats['avg_coverage'] = statistics.mean(coverages)
        stats['std_coverage'] = statistics.stdev(coverages) if len(coverages) > 1 else 0

        # Search efficiency
        search_effs = [r.search_efficiency for r in self.results if r.execution_time > 0]
        if search_effs:
            stats['avg_search_efficiency'] = statistics.mean(search_effs)

        return stats

    def get_strategy_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across different strategy selections.

        Returns:
            Dictionary mapping strategy names to their performance metrics
        """
        strategy_results: Dict[str, List[ExperimentResult]] = {}

        for result in self.results:
            if result.benchmark_metrics:
                strategy = result.benchmark_metrics.selected_strategy
                if strategy not in strategy_results:
                    strategy_results[strategy] = []
                strategy_results[strategy].append(result)

        comparison = {}
        for strategy, results in strategy_results.items():
            successes = [r for r in results if r.success]
            comparison[strategy] = {
                'runs': len(results),
                'success_rate': len(successes) / len(results) if results else 0,
                'avg_coverage': statistics.mean([r.coverage_rate for r in results]),
                'avg_time': statistics.mean([r.total_time_used for r in successes]) if successes else 0
            }

        return comparison

    def get_scaling_analysis(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Analyze how performance scales with number of searchers.

        Returns:
            Dictionary with scaling data points
        """
        by_searchers: Dict[int, List[ExperimentResult]] = {}

        for result in self.results:
            n = result.num_searchers
            if n not in by_searchers:
                by_searchers[n] = []
            by_searchers[n].append(result)

        scaling = {
            'success_rate': [],
            'avg_time': [],
            'coverage': []
        }

        for n in sorted(by_searchers.keys()):
            results = by_searchers[n]
            successes = [r for r in results if r.success]

            scaling['success_rate'].append((n, len(successes) / len(results)))
            scaling['coverage'].append((n, statistics.mean([r.coverage_rate for r in results])))
            if successes:
                scaling['avg_time'].append((n, statistics.mean([r.total_time_used for r in successes])))

        return scaling

    def export_csv(self, filepath: str) -> None:
        """
        Export results to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        if not self.results:
            return

        # Flatten results for CSV
        rows = []
        for r in self.results:
            row = {
                'experiment_id': r.experiment_id,
                'timestamp': r.timestamp,
                'seed': r.seed,
                'path_length': r.path_length,
                'num_searchers': r.num_searchers,
                'time_budget': r.time_budget,
                'sample_ratio': r.sample_ratio,
                'heterogeneous_terrain': r.heterogeneous_terrain,
                'key_position': r.key_position,
                'success': r.success,
                'key_found_by': r.key_found_by,
                'total_time_used': r.total_time_used,
                'sampling_time': r.sampling_time,
                'execution_time': r.execution_time,
                'time_remaining': r.time_remaining,
                'cells_searched': r.cells_searched,
                'coverage_rate': r.coverage_rate,
                'time_efficiency': r.time_efficiency,
                'search_efficiency': r.search_efficiency,
            }

            if r.benchmark_metrics:
                row['selected_strategy'] = r.benchmark_metrics.selected_strategy
                row['selection_confidence'] = r.benchmark_metrics.selection_confidence

            if r.coordination_metrics:
                row['redundancy_rate'] = r.coordination_metrics.redundancy_rate

            rows.append(row)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def export_json(self, filepath: str) -> None:
        """
        Export results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            'experiment_name': self.experiment_name,
            'total_runs': len(self.results),
            'statistics': self.get_statistics(),
            'results': [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a formatted summary of collected metrics."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("EXPERIMENT METRICS SUMMARY")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Total runs: {stats.get('total_runs', 0)}")
        print()

        print("SUCCESS METRICS:")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"  Successes: {stats.get('successes', 0)}")
        print(f"  Failures: {stats.get('failures', 0)}")
        print()

        print("TIME METRICS:")
        if 'avg_time_to_success' in stats:
            print(f"  Avg time to success: {stats['avg_time_to_success']:.2f}s")
            print(f"  Std time to success: {stats['std_time_to_success']:.2f}s")
            print(f"  Avg time efficiency: {stats.get('avg_time_efficiency', 0):.1%}")
        print()

        print("COVERAGE METRICS:")
        print(f"  Avg coverage: {stats.get('avg_coverage', 0):.1%}")
        print(f"  Std coverage: {stats.get('std_coverage', 0):.1%}")
        if 'avg_search_efficiency' in stats:
            print(f"  Avg search efficiency: {stats['avg_search_efficiency']:.2f} cells/s")

        print("=" * 60)
