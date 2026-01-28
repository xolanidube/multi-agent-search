"""
Strategy Benchmarking Engine.

Before committing to a strategy, run it on sample regions of the
path to empirically measure:
- Throughput: cells searched per second
- Coverage pattern: how it spreads across space
- Efficiency: useful work vs wasted effort
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import time as time_module

from ..core.cell import Path, CellStatus
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher
from ..strategies.base import Strategy


@dataclass
class BenchmarkResult:
    """Results from benchmarking a strategy on samples."""
    strategy_name: str
    avg_throughput: float          # cells per second
    avg_coverage: float            # fraction of sample covered
    cells_searched: int
    time_used: float
    sample_results: List[Dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        """
        Composite score for strategy comparison.

        Throughput weighted higher (0.7) since time is the constraint.
        """
        return 0.7 * self.avg_throughput + 0.3 * self.avg_coverage


class StrategyBenchmark:
    """
    Benchmark strategies on sample regions to estimate performance.

    The key insight: spend bounded time discovering which strategy
    works best on this terrain before committing to full search.
    """

    def __init__(
        self,
        path: Path,
        sample_ratio: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the benchmark engine.

        Args:
            path: The full search path
            sample_ratio: Fraction of path to use for sampling
            seed: Random seed for reproducibility
        """
        self.path = path
        self.sample_ratio = sample_ratio
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def create_sample_regions(self) -> Dict[str, List[int]]:
        """
        Create representative sample regions from the path.

        Returns three types of samples:
        - start_region: Beginning of path
        - mid_region: Middle of path
        - random_scatter: Random cells throughout

        Returns:
            Dictionary mapping sample names to cell ID lists
        """
        path_length = len(self.path)
        sample_size = max(5, int(path_length * self.sample_ratio))

        # Three sampling strategies for robustness
        samples = {
            'start_region': list(range(min(sample_size, path_length))),
            'mid_region': list(range(
                max(0, path_length // 2 - sample_size // 2),
                min(path_length, path_length // 2 + sample_size // 2)
            )),
            'random_scatter': random.sample(
                range(path_length),
                min(sample_size, path_length)
            )
        }

        return samples

    def create_sample_path(self, cell_ids: List[int]) -> Tuple[Path, Dict[int, int]]:
        """
        Create a mini-path for benchmarking from cell IDs.

        Args:
            cell_ids: List of cell IDs from the main path

        Returns:
            Tuple of (sample Path, mapping from sample ID to original ID)
        """
        from ..core.cell import Cell, TerrainType

        sample_cells = []
        id_mapping = {}

        for new_id, orig_id in enumerate(cell_ids):
            orig_cell = self.path[orig_id]
            sample_cell = Cell(
                id=new_id,
                search_time=orig_cell.search_time,
                traversal_cost=orig_cell.traversal_cost,
                detection_modifier=orig_cell.detection_modifier,
                terrain_type=orig_cell.terrain_type,
                prior_probability=orig_cell.prior_probability,
                has_key=False  # No key in sample
            )
            sample_cells.append(sample_cell)
            id_mapping[new_id] = orig_id

        sample_path = Path(cells=sample_cells, key_position=None)
        return sample_path, id_mapping

    def benchmark_strategy(
        self,
        strategy: Strategy,
        sample_cells: List[int],
        time_limit: float,
        searcher_speed: float = 1.0
    ) -> Dict:
        """
        Benchmark a strategy on a sample region.

        Uses a fast simulation model where we measure relative
        performance rather than absolute time. Each action is
        modeled as a unit of work.

        Args:
            strategy: Strategy to benchmark
            sample_cells: Cell IDs in the sample
            time_limit: Time budget for this benchmark (in real seconds)
            searcher_speed: Searcher movement speed

        Returns:
            Dictionary with throughput, coverage, cells_searched, time_used
        """
        if not sample_cells:
            return {'throughput': 0, 'coverage': 0, 'cells_searched': 0, 'time_used': 0}

        # Create sample path
        sample_path, _ = self.create_sample_path(sample_cells)
        sample_marks = MarkingSystem(len(sample_path))

        # Create probe searcher
        probe = Searcher(
            searcher_id=0,
            start_position=0,
            strategy=strategy,
            speed=searcher_speed
        )

        cells_searched = 0
        sim_time = 0.0

        # Use scaled time: each cell search takes 0.1s simulation time
        # This allows benchmarking to complete quickly
        time_scale = 0.1

        # Run benchmark - limit by either time or max iterations
        max_iterations = len(sample_cells) * 2  # Allow some overhead

        for _ in range(max_iterations):
            if sim_time >= time_limit:
                break

            # Get next cell
            next_cell = strategy.get_next_cell(
                probe, sample_path, sample_marks, None
            )

            if next_cell is None:
                break  # Sample exhausted

            # Simulate move and search with scaled time
            move_distance = abs(next_cell - probe.position)
            move_time = move_distance * time_scale * 0.5  # Movement is fast
            search_time = time_scale  # Each search takes base time

            action_time = move_time + search_time
            sim_time += action_time

            # Update state
            probe.position = next_cell
            sample_marks.mark(next_cell, CellStatus.SEARCHED, 0)
            cells_searched += 1

        # Calculate metrics - scale throughput to expected real-world performance
        if sim_time > 0:
            # Throughput in cells per real second
            throughput = cells_searched / sim_time
        else:
            throughput = 0

        coverage = cells_searched / len(sample_cells) if sample_cells else 0

        return {
            'throughput': throughput,
            'coverage': coverage,
            'cells_searched': cells_searched,
            'time_used': sim_time
        }

    def run_full_benchmark(
        self,
        strategies: Dict[str, Strategy],
        time_budget: float,
        searcher_speed: float = 1.0
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark all strategies on all sample regions.

        Args:
            strategies: Dictionary of strategy name -> Strategy
            time_budget: Total time for all benchmarking
            searcher_speed: Searcher movement speed

        Returns:
            Dictionary of strategy name -> BenchmarkResult
        """
        samples = self.create_sample_regions()
        time_per_strategy = time_budget / max(1, len(strategies))

        results = {}

        for strategy_name, strategy in strategies.items():
            strategy_results = []
            time_per_sample = time_per_strategy / len(samples)

            for sample_name, sample_cells in samples.items():
                result = self.benchmark_strategy(
                    strategy,
                    sample_cells,
                    time_per_sample,
                    searcher_speed
                )
                result['sample'] = sample_name
                strategy_results.append(result)

            # Aggregate across samples
            total_throughput = sum(r['throughput'] for r in strategy_results)
            total_coverage = sum(r['coverage'] for r in strategy_results)
            total_cells = sum(r['cells_searched'] for r in strategy_results)
            total_time = sum(r['time_used'] for r in strategy_results)

            num_samples = len(strategy_results)
            avg_throughput = total_throughput / num_samples if num_samples > 0 else 0
            avg_coverage = total_coverage / num_samples if num_samples > 0 else 0

            results[strategy_name] = BenchmarkResult(
                strategy_name=strategy_name,
                avg_throughput=avg_throughput,
                avg_coverage=avg_coverage,
                cells_searched=total_cells,
                time_used=total_time,
                sample_results=strategy_results
            )

        return results

    def quick_benchmark(
        self,
        strategies: Dict[str, Strategy],
        time_budget: float
    ) -> Dict[str, float]:
        """
        Quick benchmark returning just throughput scores.

        Args:
            strategies: Dictionary of strategies
            time_budget: Time for benchmarking

        Returns:
            Dictionary of strategy name -> throughput
        """
        results = self.run_full_benchmark(strategies, time_budget)
        return {name: r.avg_throughput for name, r in results.items()}
