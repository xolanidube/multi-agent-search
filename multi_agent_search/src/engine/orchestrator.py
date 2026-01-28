"""
Master Orchestrator for the complete search pipeline.

Orchestrates the full workflow:
1. Allocate time budgets (sampling vs execution)
2. Benchmark strategies on sample regions
3. Select optimal strategy/mix
4. Execute timed search
5. Compile and report results
"""

from typing import Dict, List, Optional, Callable
import time as time_module

from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.time_manager import TimeManager, Phase
from ..core.searcher import Searcher
from ..strategies.base import Strategy
from ..strategies.greedy_nearest import GreedyNearestStrategy
from ..strategies.territorial import TerritorialStrategy
from ..strategies.probabilistic import ProbabilisticStrategy
from ..strategies.contrarian import ContrarianStrategy
from ..strategies.random_walk import RandomWalkStrategy
from .benchmark import StrategyBenchmark, BenchmarkResult
from .selector import StrategySelector
from .simulation import Simulation, SimulationMetrics


class SearchOrchestrator:
    """
    Master orchestrator for the time-constrained multi-agent search.

    Manages the complete pipeline from benchmarking to execution,
    implementing the core innovation of spending time upfront to
    discover the best strategy before committing to full search.
    """

    def __init__(
        self,
        path_length: int = 100,
        num_searchers: int = 3,
        time_budget: float = 60.0,
        key_position: Optional[int] = None,
        heterogeneous_terrain: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            path_length: Number of cells in the path
            num_searchers: Number of search agents
            time_budget: Total time in seconds
            key_position: Where to place key (random if None)
            heterogeneous_terrain: If True, terrain varies
            seed: Random seed for reproducibility
        """
        self.path_length = path_length
        self.num_searchers = num_searchers
        self.time_budget = time_budget
        self.seed = seed

        # Generate path
        if heterogeneous_terrain:
            self.path = Path.generate_with_hotspots(
                path_length,
                hotspot_count=max(3, path_length // 20),
                seed=seed
            )
        else:
            self.path = Path.generate(path_length, key_position, seed=seed)

        if key_position is not None:
            # Override key position
            if self.path.key_position is not None:
                self.path.cells[self.path.key_position].has_key = False
            self.path.key_position = key_position
            self.path.cells[key_position].has_key = True

        # Create components
        self.time_manager = TimeManager(time_budget)
        self.marks = MarkingSystem(path_length)

        # Available strategies
        self.strategies: Dict[str, Strategy] = {
            'greedy_nearest': GreedyNearestStrategy(),
            'territorial': TerritorialStrategy(num_searchers),
            'probabilistic': ProbabilisticStrategy(),
            'contrarian': ContrarianStrategy(),
            'random_walk': RandomWalkStrategy(seed),
        }

        # Create searchers (strategy assigned later)
        self.searchers = [
            Searcher(i, start_position=0)
            for i in range(num_searchers)
        ]

        # Engines
        self.benchmark_engine = StrategyBenchmark(self.path, seed=seed)
        self.strategy_selector = StrategySelector(self.strategies)

        # Results storage
        self.benchmark_results: Optional[Dict[str, BenchmarkResult]] = None
        self.selected_strategy: Optional[str] = None
        self.allocation: Optional[Dict[str, int]] = None
        self.simulation_metrics: Optional[SimulationMetrics] = None

        # Callbacks
        self.on_phase_change: Optional[Callable[[Phase], None]] = None
        self.on_benchmark_complete: Optional[Callable[[Dict], None]] = None
        self.on_search_step: Optional[Callable[[int, int, bool], None]] = None

    def run(self, sample_ratio: float = 0.15, verbose: bool = True) -> Dict:
        """
        Run the complete search pipeline.

        Args:
            sample_ratio: Fraction of time for benchmarking
            verbose: If True, print progress

        Returns:
            Dictionary with complete results
        """
        # Phase 0: Setup
        self.time_manager.allocate_budgets(sample_ratio)
        self.time_manager.start()

        if verbose:
            print(f"\n{'='*60}")
            print("MULTI-AGENT SEARCH ORCHESTRATOR")
            print(f"{'='*60}")
            print(f"Path length: {self.path_length} cells")
            print(f"Searchers: {self.num_searchers}")
            print(f"Time budget: {self.time_budget:.1f}s")
            print(f"  - Sampling: {self.time_manager.sample_budget:.1f}s")
            print(f"  - Execution: {self.time_manager.search_budget:.1f}s")
            print(f"Key position: {self.path.key_position}")
            print()

        # Phase 1: Benchmark
        if verbose:
            print("[PHASE 1] Benchmarking strategies...")

        if self.on_phase_change:
            self.on_phase_change(Phase.SAMPLING)

        self.benchmark_results = self.benchmark_engine.run_full_benchmark(
            self.strategies,
            self.time_manager.sample_budget
        )

        if self.on_benchmark_complete:
            self.on_benchmark_complete(self.benchmark_results)

        if verbose:
            self._print_benchmark_results()

        # Phase 2: Strategy Selection
        if verbose:
            print("\n[PHASE 2] Selecting optimal strategy...")

        remaining_time = self.time_manager.search_time_remaining()

        best_strategy, confidence, method = self.strategy_selector.select_best(
            self.benchmark_results,
            remaining_time,
            len(self.path)
        )
        self.selected_strategy = best_strategy

        self.allocation = self.strategy_selector.recommend_strategy_mix(
            self.benchmark_results,
            self.num_searchers
        )

        if verbose:
            print(f"  Best strategy: {best_strategy}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Selection method: {method}")
            print(f"  Allocation: {self.allocation}")

        # Phase 3: Assign strategies to searchers
        self._assign_strategies()

        # Phase 4: Execute Search
        if verbose:
            print("\n[PHASE 3] Executing timed search...")

        if self.on_phase_change:
            self.on_phase_change(Phase.SEARCHING)

        simulation = Simulation(
            self.path,
            self.searchers,
            self.time_manager,
            use_simulation_time=True
        )
        simulation.on_step = self.on_search_step

        self.simulation_metrics = simulation.run()

        # Phase 5: Report Results
        results = self._compile_results()

        if verbose:
            self._print_results(results)

        return results

    def _assign_strategies(self) -> None:
        """Assign strategies to searchers based on allocation."""
        strategy_list = []
        for name, count in self.allocation.items():
            strategy_list.extend([name] * count)

        for i, searcher in enumerate(self.searchers):
            if i < len(strategy_list):
                strategy_name = strategy_list[i]
            else:
                strategy_name = strategy_list[-1] if strategy_list else 'greedy_nearest'

            searcher.set_strategy(self.strategies[strategy_name])

    def _compile_results(self) -> Dict:
        """Compile complete results from all phases."""
        return {
            'success': self.simulation_metrics.success,
            'key_position': self.path.key_position,
            'key_found_by': self.simulation_metrics.key_found_by,

            'time_budget': self.time_budget,
            'sample_time': self.time_manager.sample_budget,
            'search_time': self.time_manager.search_budget,
            'time_used': self.simulation_metrics.time_used,
            'time_remaining': self.simulation_metrics.time_remaining,

            'cells_searched': self.simulation_metrics.cells_searched,
            'total_cells': self.path_length,
            'coverage_rate': self.simulation_metrics.coverage_rate,
            'efficiency': self.simulation_metrics.efficiency,
            'redundant_searches': self.simulation_metrics.redundant_searches,

            'selected_strategy': self.selected_strategy,
            'allocation': self.allocation,
            'benchmark_scores': {
                name: result.score
                for name, result in self.benchmark_results.items()
            },

            'searcher_stats': self.simulation_metrics.searcher_stats,
            'rounds_executed': self.simulation_metrics.rounds_executed,
        }

    def _print_benchmark_results(self) -> None:
        """Print formatted benchmark results."""
        print("\n  Strategy Benchmarks:")
        print("  " + "-" * 55)
        print(f"  {'Strategy':<20} | {'Throughput':>10} | {'Coverage':>8} | {'Score':>6}")
        print("  " + "-" * 55)

        sorted_results = sorted(
            self.benchmark_results.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        for name, result in sorted_results:
            print(f"  {name:<20} | {result.avg_throughput:>8.2f}/s | "
                  f"{result.avg_coverage:>7.1%} | {result.score:>6.3f}")

    def _print_results(self, results: Dict) -> None:
        """Print formatted final results."""
        print(f"\n{'='*60}")
        print("SEARCH RESULTS")
        print(f"{'='*60}")

        status = "SUCCESS" if results['success'] else "FAILED"
        status_symbol = "[+]" if results['success'] else "[-]"

        print(f"  {status_symbol} Status: {status}")

        if results['success']:
            print(f"  Key found at cell {results['key_position']} "
                  f"by searcher {results['key_found_by']}")

        print(f"\n  Time:")
        print(f"    Used: {results['time_used']:.2f}s")
        print(f"    Remaining: {results['time_remaining']:.2f}s")
        if results['success']:
            print(f"    Efficiency: {results['efficiency']:.1%}")

        print(f"\n  Coverage:")
        print(f"    Cells searched: {results['cells_searched']} / {results['total_cells']}")
        print(f"    Coverage rate: {results['coverage_rate']:.1%}")
        print(f"    Redundant searches: {results['redundant_searches']}")

        print(f"\n  Strategy: {results['selected_strategy']}")
        print(f"  Allocation: {results['allocation']}")

        print(f"\n  Searcher Performance:")
        for sid, stats in results['searcher_stats'].items():
            print(f"    S{sid}: {stats['cells_searched']} cells, "
                  f"{stats['strategy']} strategy")

        print(f"{'='*60}\n")

    def get_path(self) -> Path:
        """Get the search path."""
        return self.path

    def get_marks(self) -> MarkingSystem:
        """Get the marking system."""
        return self.marks

    def get_searchers(self) -> List[Searcher]:
        """Get the searchers."""
        return self.searchers
