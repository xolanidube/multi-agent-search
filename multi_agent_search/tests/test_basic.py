"""
Basic tests for the multi-agent search system.

Run with: python -m pytest tests/test_basic.py -v
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cell import Cell, Path as SearchPath, CellStatus, TerrainType
from src.core.marking_system import MarkingSystem
from src.core.time_manager import TimeManager, Phase
from src.core.searcher import Searcher
from src.strategies.greedy_nearest import GreedyNearestStrategy
from src.strategies.territorial import TerritorialStrategy
from src.strategies.random_walk import RandomWalkStrategy
from src.engine.simulation import Simulation
from src.engine.orchestrator import SearchOrchestrator


class TestCell:
    """Tests for Cell and Path classes."""

    def test_cell_creation(self):
        cell = Cell(id=0, search_time=1.0)
        assert cell.id == 0
        assert cell.search_time == 1.0
        assert not cell.has_key

    def test_path_generation(self):
        path = SearchPath.generate(length=10, seed=42)
        assert len(path) == 10
        assert path.key_position is not None
        assert 0 <= path.key_position < 10
        assert path[path.key_position].has_key

    def test_path_with_hotspots(self):
        path = SearchPath.generate_with_hotspots(length=20, hotspot_count=3, seed=42)
        assert len(path) == 20
        # Verify probabilities sum to 1
        total_prob = sum(c.prior_probability for c in path)
        assert abs(total_prob - 1.0) < 0.001


class TestMarkingSystem:
    """Tests for MarkingSystem."""

    def test_marking_system_creation(self):
        marks = MarkingSystem(num_cells=10)
        assert marks.num_cells == 10
        assert marks.count_unmarked() == 10
        assert marks.count_searched() == 0

    def test_mark_cell(self):
        marks = MarkingSystem(num_cells=10)
        marks.mark(5, CellStatus.SEARCHED, searcher_id=0)
        assert marks.get_status(5) == CellStatus.SEARCHED
        assert not marks.is_available(5)
        assert marks.count_searched() == 1

    def test_get_unmarked_cells(self):
        marks = MarkingSystem(num_cells=5)
        marks.mark(2, CellStatus.SEARCHED, 0)
        marks.mark(4, CellStatus.SEARCHED, 0)
        unmarked = marks.get_unmarked_cells()
        assert len(unmarked) == 3
        assert 2 not in unmarked
        assert 4 not in unmarked


class TestTimeManager:
    """Tests for TimeManager."""

    def test_time_manager_creation(self):
        tm = TimeManager(total_budget=60.0)
        assert tm.total_budget == 60.0
        assert tm.phase == Phase.IDLE

    def test_budget_allocation(self):
        tm = TimeManager(total_budget=60.0)
        tm.allocate_budgets(sample_ratio=0.2)
        assert tm.sample_budget == 12.0
        assert tm.search_budget == 48.0

    def test_phase_transitions(self):
        tm = TimeManager(total_budget=60.0)
        tm.allocate_budgets(0.1)
        tm.start()
        assert tm.phase == Phase.SAMPLING
        tm.transition_to_search()
        assert tm.phase == Phase.SEARCHING


class TestStrategies:
    """Tests for search strategies."""

    def test_greedy_nearest(self):
        path = SearchPath.generate(10, seed=42)
        marks = MarkingSystem(10)
        strategy = GreedyNearestStrategy()
        searcher = Searcher(0, start_position=5, strategy=strategy)

        # Should pick nearest unmarked
        next_cell = strategy.get_next_cell(searcher, path, marks, None)
        assert next_cell is not None
        # Should be adjacent (4 or 6)
        assert next_cell in [4, 5, 6]

    def test_territorial(self):
        path = SearchPath.generate(10, seed=42)
        marks = MarkingSystem(10)
        strategy = TerritorialStrategy(num_searchers=2)
        searcher = Searcher(0, start_position=0, strategy=strategy)

        # Searcher 0 should pick from first half
        next_cell = strategy.get_next_cell(searcher, path, marks, None)
        assert next_cell is not None
        assert 0 <= next_cell < 5  # First half

    def test_random_walk(self):
        path = SearchPath.generate(10, seed=42)
        marks = MarkingSystem(10)
        strategy = RandomWalkStrategy(seed=42)
        searcher = Searcher(0, start_position=0, strategy=strategy)

        next_cell = strategy.get_next_cell(searcher, path, marks, None)
        assert next_cell is not None
        assert 0 <= next_cell < 10


class TestSearcher:
    """Tests for Searcher class."""

    def test_searcher_creation(self):
        searcher = Searcher(searcher_id=0, start_position=5)
        assert searcher.id == 0
        assert searcher.position == 5
        assert searcher.cells_searched == 0

    def test_search_cell(self):
        path = SearchPath.generate(10, key_position=5, seed=42)
        marks = MarkingSystem(10)
        searcher = Searcher(0, start_position=5, detection_probability=1.0)

        # Search the cell with the key
        found, time = searcher.search_cell(path[5], marks, 0.0)
        assert found
        assert marks.get_status(5) == CellStatus.FOUND
        assert searcher.cells_searched == 1


class TestSimulation:
    """Tests for Simulation class."""

    def test_simulation_finds_key(self):
        path = SearchPath.generate(20, key_position=10, seed=42)
        time_manager = TimeManager(30.0)
        time_manager.allocate_budgets(0.0)  # No sampling
        time_manager.start()

        searcher = Searcher(0, start_position=0)
        searcher.set_strategy(GreedyNearestStrategy())

        sim = Simulation(path, [searcher], time_manager)
        metrics = sim.run()

        assert metrics.success
        assert metrics.cells_searched > 0

    def test_multiple_searchers(self):
        path = SearchPath.generate(30, seed=42)
        time_manager = TimeManager(30.0)
        time_manager.allocate_budgets(0.0)
        time_manager.start()

        searchers = [
            Searcher(i, start_position=0, strategy=GreedyNearestStrategy())
            for i in range(3)
        ]

        sim = Simulation(path, searchers, time_manager)
        metrics = sim.run()

        assert metrics.cells_searched > 0
        # Multiple searchers should have worked
        assert len(metrics.searcher_stats) == 3


class TestOrchestrator:
    """Tests for SearchOrchestrator."""

    def test_orchestrator_basic(self):
        orch = SearchOrchestrator(
            path_length=20,
            num_searchers=2,
            time_budget=20.0,
            seed=42
        )

        results = orch.run(sample_ratio=0.1, verbose=False)

        assert 'success' in results
        assert 'time_used' in results
        assert 'selected_strategy' in results
        assert results['selected_strategy'] in [
            'greedy_nearest', 'territorial', 'probabilistic',
            'contrarian', 'random_walk'
        ]


def run_tests():
    """Run all tests manually (without pytest)."""
    test_classes = [
        TestCell,
        TestMarkingSystem,
        TestTimeManager,
        TestStrategies,
        TestSearcher,
        TestSimulation,
        TestOrchestrator,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            try:
                getattr(instance, method_name)()
                print(f"  [PASS] {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                total_failed += 1
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                total_failed += 1

    print("\n" + "=" * 40)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 40)

    return total_failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
