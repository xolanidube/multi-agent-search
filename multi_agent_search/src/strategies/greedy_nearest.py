"""
Greedy Nearest Strategy - Select the closest unmarked cell.

This is the simplest strategy: always move to the nearest available
cell. Minimizes individual travel time but may cause conflicts when
multiple searchers converge on the same area.

Complexity: O(n) where n is path length
Optimal when: Single searcher, or low searcher density
"""

from typing import Optional, List

from .base import Strategy
from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


class GreedyNearestStrategy(Strategy):
    """
    Select the nearest unmarked cell to search.

    Simple and effective for single-agent search. May cause
    inefficient clustering when multiple agents use this strategy.
    """

    def __init__(self):
        super().__init__(name="greedy_nearest")

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find the nearest unmarked cell.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not used by this strategy

        Returns:
            ID of nearest unmarked cell, or None if all searched
        """
        unmarked = marks.get_unmarked_cells()

        if not unmarked:
            return None

        # Find closest unmarked cell
        current_pos = searcher.position
        best_cell = None
        best_distance = float('inf')

        for cell_id in unmarked:
            distance = abs(cell_id - current_pos)  # Simple linear distance
            if distance < best_distance:
                best_distance = distance
                best_cell = cell_id

        return best_cell
