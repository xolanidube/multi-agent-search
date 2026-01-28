"""
Searcher Agent for the multi-agent search system.

Each searcher is an autonomous agent that:
- Selects which cell to search next based on its strategy
- Moves through the path
- Searches cells and updates the marking system
- Can switch strategies at runtime
"""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
import random

from .cell import Cell, CellStatus, Path
from .marking_system import MarkingSystem, Mark

if TYPE_CHECKING:
    from ..strategies.base import Strategy


@dataclass
class SearchAction:
    """Record of a single search action."""
    cell_id: int
    found: bool
    time_taken: float
    searcher_id: int
    timestamp: float


class Searcher:
    """
    Autonomous search agent.

    Each searcher has a position, strategy, and capabilities.
    Searchers coordinate indirectly through the marking system.

    Attributes:
        id: Unique identifier
        position: Current cell ID
        strategy: Search strategy function
        speed: Movement speed multiplier (1.0 = normal)
        detection_probability: Probability of finding key if present
        cells_searched: Count of cells this searcher has searched
        distance_traveled: Total distance traveled
        action_history: Log of all search actions
    """

    def __init__(
        self,
        searcher_id: int,
        start_position: int = 0,
        strategy: Optional['Strategy'] = None,
        speed: float = 1.0,
        detection_probability: float = 1.0
    ):
        """
        Initialize a searcher.

        Args:
            searcher_id: Unique identifier
            start_position: Starting cell ID
            strategy: Initial search strategy
            speed: Movement speed (higher = faster)
            detection_probability: Chance of finding key if present
        """
        self.id = searcher_id
        self.position = start_position
        self.strategy = strategy
        self.speed = speed
        self.detection_probability = detection_probability

        # Tracking
        self.cells_searched = 0
        self.distance_traveled = 0.0
        self.time_spent = 0.0
        self.action_history: List[SearchAction] = []

        # Strategy info
        self.strategy_name: str = "none"
        self.estimated_throughput: float = 1.0

    def set_strategy(self, strategy: 'Strategy') -> None:
        """
        Set the search strategy.

        Args:
            strategy: New strategy to use
        """
        self.strategy = strategy
        if hasattr(strategy, 'name'):
            self.strategy_name = strategy.name

    def get_next_cell(self, path: Path, marks: MarkingSystem,
                      other_positions: Optional[List[int]] = None) -> Optional[int]:
        """
        Get the next cell to search based on current strategy.

        Args:
            path: The search path
            marks: Current marking system state
            other_positions: Positions of other searchers (for some strategies)

        Returns:
            Cell ID to search next, or None if no available cells
        """
        if self.strategy is None:
            # No strategy - pick random unmarked
            unmarked = marks.get_unmarked_cells()
            if not unmarked:
                return None
            return random.choice(unmarked)

        return self.strategy.get_next_cell(
            searcher=self,
            path=path,
            marks=marks,
            other_positions=other_positions
        )

    def move_to(self, cell_id: int, path: Path) -> float:
        """
        Move to a target cell.

        Args:
            cell_id: Target cell ID
            path: The search path

        Returns:
            Time taken to move
        """
        distance = path.distance(self.position, cell_id)
        time_taken = distance / self.speed

        self.position = cell_id
        self.distance_traveled += distance
        self.time_spent += time_taken

        return time_taken

    def search_cell(
        self,
        cell: Cell,
        marks: MarkingSystem,
        timestamp: float = 0.0
    ) -> tuple[bool, float]:
        """
        Search a cell for the key.

        This method:
        1. Marks the cell as IN_PROGRESS
        2. Performs the search (takes time based on cell properties)
        3. Determines if key is found (based on detection probability)
        4. Updates marking to SEARCHED or FOUND

        Args:
            cell: The cell to search
            marks: The marking system
            timestamp: Current timestamp

        Returns:
            Tuple of (found: bool, time_taken: float)
        """
        # Mark as in progress
        marks.mark(cell.id, CellStatus.IN_PROGRESS, self.id, timestamp)

        # Calculate search time
        search_time = cell.search_time / self.speed

        # Determine if key is found
        found = False
        if cell.has_key:
            # Apply detection probability and terrain modifier
            effective_detection = self.detection_probability * cell.detection_modifier
            if random.random() < effective_detection:
                found = True

        # Update marking
        if found:
            marks.mark(cell.id, CellStatus.FOUND, self.id, timestamp + search_time)
        else:
            marks.mark(cell.id, CellStatus.SEARCHED, self.id, timestamp + search_time)

        # Update tracking
        self.cells_searched += 1
        self.time_spent += search_time

        # Record action
        action = SearchAction(
            cell_id=cell.id,
            found=found,
            time_taken=search_time,
            searcher_id=self.id,
            timestamp=timestamp
        )
        self.action_history.append(action)

        return found, search_time

    def execute_search_step(
        self,
        path: Path,
        marks: MarkingSystem,
        timestamp: float = 0.0,
        other_positions: Optional[List[int]] = None
    ) -> tuple[Optional[int], bool, float]:
        """
        Execute a complete search step: select cell, move, search.

        Args:
            path: The search path
            marks: The marking system
            timestamp: Current timestamp
            other_positions: Positions of other searchers

        Returns:
            Tuple of (cell_id searched, found, total_time)
        """
        # Get next cell from strategy
        next_cell_id = self.get_next_cell(path, marks, other_positions)

        if next_cell_id is None:
            return None, False, 0.0

        # Move to cell
        move_time = self.move_to(next_cell_id, path)

        # Search cell
        cell = path[next_cell_id]
        found, search_time = self.search_cell(cell, marks, timestamp + move_time)

        total_time = move_time + search_time
        return next_cell_id, found, total_time

    def get_statistics(self) -> dict:
        """Get searcher statistics."""
        return {
            'id': self.id,
            'position': self.position,
            'strategy': self.strategy_name,
            'cells_searched': self.cells_searched,
            'distance_traveled': self.distance_traveled,
            'time_spent': self.time_spent,
            'throughput': self.cells_searched / self.time_spent if self.time_spent > 0 else 0,
            'actions': len(self.action_history),
        }

    def reset(self, start_position: int = 0) -> None:
        """Reset searcher state for a new simulation."""
        self.position = start_position
        self.cells_searched = 0
        self.distance_traveled = 0.0
        self.time_spent = 0.0
        self.action_history.clear()

    def copy(self) -> 'Searcher':
        """Create a copy of this searcher (for benchmarking)."""
        new_searcher = Searcher(
            searcher_id=self.id,
            start_position=self.position,
            strategy=self.strategy,
            speed=self.speed,
            detection_probability=self.detection_probability
        )
        new_searcher.strategy_name = self.strategy_name
        new_searcher.estimated_throughput = self.estimated_throughput
        return new_searcher

    def __repr__(self) -> str:
        return f"Searcher(id={self.id}, pos={self.position}, strategy={self.strategy_name})"
