"""
Simulation Engine for running the multi-agent search.

Handles the main simulation loop where searchers execute
their strategies, update the marking system, and race
against the deadline to find the key.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import time as time_module

from ..core.cell import Path, CellStatus
from ..core.marking_system import MarkingSystem
from ..core.time_manager import TimeManager, Phase
from ..core.searcher import Searcher
from ..strategies.base import Strategy


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    success: bool = False
    key_found_by: Optional[int] = None
    key_found_at: Optional[int] = None
    time_used: float = 0.0
    time_remaining: float = 0.0
    cells_searched: int = 0
    total_cells: int = 0
    redundant_searches: int = 0
    rounds_executed: int = 0
    searcher_stats: Dict[int, Dict] = field(default_factory=dict)

    @property
    def coverage_rate(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.cells_searched / self.total_cells

    @property
    def efficiency(self) -> float:
        """Time efficiency: how much time was left when found."""
        if not self.success:
            return 0.0
        total = self.time_used + self.time_remaining
        if total == 0:
            return 0.0
        return self.time_remaining / total


class Simulation:
    """
    Main simulation engine for multi-agent search.

    Runs the search loop where searchers take turns:
    1. Select next cell based on strategy
    2. Move to cell
    3. Search cell and update marking
    4. Check if key found or time expired
    """

    def __init__(
        self,
        path: Path,
        searchers: List[Searcher],
        time_manager: TimeManager,
        use_simulation_time: bool = True
    ):
        """
        Initialize simulation.

        Args:
            path: The search path
            searchers: List of searcher agents
            time_manager: Time budget manager
            use_simulation_time: If True, use simulated time instead of wall clock
        """
        self.path = path
        self.searchers = searchers
        self.time_manager = time_manager
        self.marks = MarkingSystem(len(path))

        self.key_found = False
        self.finder_id: Optional[int] = None
        self.metrics = SimulationMetrics(total_cells=len(path))

        self.simulation_time = 0.0
        self.use_simulation_time = use_simulation_time
        self.time_manager.set_simulation_mode(use_simulation_time)

        # Callbacks for visualization
        self.on_step: Optional[Callable[[int, int, bool], None]] = None
        self.on_round: Optional[Callable[[int], None]] = None

    def run(self) -> SimulationMetrics:
        """
        Run the simulation until key is found or time expires.

        Returns:
            SimulationMetrics with results
        """
        self.time_manager.transition_to_search()
        round_num = 0

        while not self.key_found and not self._is_time_expired():
            # Execute one round (all searchers take one action)
            self._execute_round(round_num)
            round_num += 1

            # Callback for visualization
            if self.on_round:
                self.on_round(round_num)

            # Check if all cells searched
            if not self.marks.get_unmarked_cells():
                break

        # Compile final metrics
        self._compile_metrics(round_num)

        if self.key_found:
            self.time_manager.mark_complete(success=True)
        else:
            self.time_manager.mark_complete(success=False)

        return self.metrics

    def _execute_round(self, round_num: int) -> None:
        """
        Execute one round where all searchers take one action.

        Args:
            round_num: Current round number
        """
        # Get current positions of all searchers for contrarian strategy
        other_positions = [s.position for s in self.searchers]

        for searcher in self.searchers:
            if self.key_found or self._is_time_expired():
                break

            # Get other positions excluding this searcher
            positions_for_strategy = [
                p for i, p in enumerate(other_positions)
                if i != searcher.id
            ]

            # Execute search step
            cell_id, found, action_time = searcher.execute_search_step(
                self.path,
                self.marks,
                self.simulation_time,
                positions_for_strategy
            )

            if cell_id is not None:
                # Update simulation time
                self.simulation_time += action_time
                if self.use_simulation_time:
                    self.time_manager.add_simulation_time(action_time)

                # Update positions for next searcher
                other_positions[searcher.id] = searcher.position

                # Callback for visualization
                if self.on_step:
                    self.on_step(searcher.id, cell_id, found)

                if found:
                    self.key_found = True
                    self.finder_id = searcher.id

    def _is_time_expired(self) -> bool:
        """Check if time budget is exhausted."""
        return self.time_manager.is_expired()

    def _compile_metrics(self, rounds: int) -> None:
        """Compile final simulation metrics."""
        self.metrics.success = self.key_found
        self.metrics.key_found_by = self.finder_id
        self.metrics.key_found_at = self.path.key_position if self.key_found else None
        self.metrics.time_used = self.time_manager.elapsed()
        self.metrics.time_remaining = self.time_manager.remaining()
        self.metrics.cells_searched = self.marks.count_searched()
        self.metrics.redundant_searches = self.marks.get_redundant_searches()
        self.metrics.rounds_executed = rounds

        # Per-searcher stats
        for searcher in self.searchers:
            self.metrics.searcher_stats[searcher.id] = searcher.get_statistics()

    def get_state(self) -> Dict:
        """
        Get current simulation state for visualization.

        Returns:
            Dictionary with current state
        """
        return {
            'simulation_time': self.simulation_time,
            'time_remaining': self.time_manager.remaining(),
            'key_found': self.key_found,
            'cells_by_status': self.marks.count_by_status(),
            'searcher_positions': {s.id: s.position for s in self.searchers},
            'coverage': self.marks.get_coverage_percentage(),
        }

    def reset(self) -> None:
        """Reset simulation state for new run."""
        self.marks.reset()
        self.key_found = False
        self.finder_id = None
        self.simulation_time = 0.0
        self.metrics = SimulationMetrics(total_cells=len(self.path))

        for searcher in self.searchers:
            searcher.reset()


class SimulationRunner:
    """
    Helper class to run multiple simulations with different configurations.
    """

    def __init__(self, path_length: int = 100, seed: Optional[int] = None):
        """
        Initialize the runner.

        Args:
            path_length: Length of the search path
            seed: Random seed for reproducibility
        """
        self.path_length = path_length
        self.seed = seed

    def run_single(
        self,
        num_searchers: int,
        time_budget: float,
        strategies: List[Strategy],
        heterogeneous_terrain: bool = False
    ) -> SimulationMetrics:
        """
        Run a single simulation.

        Args:
            num_searchers: Number of searchers
            time_budget: Total time budget
            strategies: Strategies to assign (cycled if fewer than searchers)
            heterogeneous_terrain: Whether to use varied terrain

        Returns:
            Simulation metrics
        """
        # Generate path
        path = Path.generate(
            self.path_length,
            heterogeneous=heterogeneous_terrain,
            seed=self.seed
        )

        # Create time manager
        time_manager = TimeManager(time_budget)
        time_manager.allocate_budgets(sample_ratio=0.0)  # No sampling
        time_manager.start()

        # Create searchers
        searchers = []
        for i in range(num_searchers):
            strategy = strategies[i % len(strategies)]
            searcher = Searcher(
                searcher_id=i,
                start_position=0,
                strategy=strategy
            )
            searchers.append(searcher)

        # Run simulation
        sim = Simulation(path, searchers, time_manager)
        return sim.run()

    def run_comparison(
        self,
        strategies: List[Strategy],
        num_runs: int = 10,
        num_searchers: int = 3,
        time_budget: float = 60.0
    ) -> Dict[str, Dict]:
        """
        Run multiple simulations to compare strategies.

        Args:
            strategies: Strategies to compare (one per simulation)
            num_runs: Number of runs per strategy
            num_searchers: Number of searchers per run
            time_budget: Time budget per run

        Returns:
            Dictionary of strategy_name -> aggregated metrics
        """
        results = {}

        for strategy in strategies:
            strategy_name = strategy.name
            successes = 0
            total_time = 0.0
            total_coverage = 0.0

            for run in range(num_runs):
                # Use different seed for each run
                if self.seed is not None:
                    import random
                    random.seed(self.seed + run)

                metrics = self.run_single(
                    num_searchers,
                    time_budget,
                    [strategy]  # All searchers use same strategy
                )

                if metrics.success:
                    successes += 1
                    total_time += metrics.time_used

                total_coverage += metrics.coverage_rate

            results[strategy_name] = {
                'success_rate': successes / num_runs,
                'avg_time': total_time / max(1, successes),
                'avg_coverage': total_coverage / num_runs,
            }

        return results
