"""
Contrarian / Gap Filler Strategy - Maximize distance from other searchers.

This strategy aims to fill gaps in coverage by always moving to
the cell furthest from all other searchers. Effective for reducing
redundancy and ensuring broad coverage.

Complexity: O(n * k) where k is number of searchers
Optimal when: Multiple searchers, want to minimize overlap
"""

from typing import Optional, List

from .base import Strategy
from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


class ContrarianStrategy(Strategy):
    """
    Select the cell furthest from all other searchers.

    Acts as a "gap filler" by always targeting the most
    underserved areas of the path. Reduces clustering and
    improves overall coverage efficiency.
    """

    def __init__(self):
        super().__init__(name="contrarian")

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find the unmarked cell furthest from other searchers.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: List of other searchers' positions

        Returns:
            ID of most isolated unmarked cell
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        # If no other positions known, fall back to greedy
        if not other_positions:
            current_pos = searcher.position
            return min(unmarked, key=lambda c: abs(c - current_pos))

        # Find cell with maximum minimum distance to any other searcher
        best_cell = None
        best_min_distance = -1

        for cell_id in unmarked:
            # Find minimum distance to any other searcher
            min_distance = float('inf')
            for other_pos in other_positions:
                distance = abs(cell_id - other_pos)
                min_distance = min(min_distance, distance)

            # We want to maximize the minimum distance
            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_cell = cell_id

        return best_cell


class BalancedContrarianStrategy(Strategy):
    """
    Contrarian strategy balanced with travel cost.

    Avoids traveling very far just to maximize distance from others.
    Uses a weighted combination of isolation and travel cost.
    """

    def __init__(self, travel_penalty: float = 0.3):
        """
        Initialize with travel penalty.

        Args:
            travel_penalty: How much to penalize travel distance (0-1)
        """
        super().__init__(name="balanced_contrarian")
        self.travel_penalty = travel_penalty

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find cell with best isolation-travel tradeoff.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: List of other searchers' positions

        Returns:
            ID of best-scoring unmarked cell
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        current_pos = searcher.position
        path_length = len(path)

        # If no other positions, just use greedy
        if not other_positions:
            return min(unmarked, key=lambda c: abs(c - current_pos))

        best_cell = None
        best_score = float('-inf')

        for cell_id in unmarked:
            # Minimum distance to any other searcher (normalized)
            min_distance_to_others = min(
                abs(cell_id - pos) for pos in other_positions
            ) / path_length

            # Distance from current position (normalized)
            travel_distance = abs(cell_id - current_pos) / path_length

            # Score: maximize isolation, minimize travel
            score = min_distance_to_others - self.travel_penalty * travel_distance

            if score > best_score:
                best_score = score
                best_cell = cell_id

        return best_cell
