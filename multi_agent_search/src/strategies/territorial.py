"""
Territorial Strategy - Divide path into segments and claim one.

Each searcher claims a segment of the path based on their ID.
They search within their territory first, then help others.

This reduces conflicts and provides good coverage when the
number of searchers is known upfront.

Complexity: O(n/k) per searcher where k is number of searchers
Optimal when: Fixed number of searchers, uniform terrain
"""

from typing import Optional, List

from .base import Strategy
from ..core.cell import Path
from ..core.marking_system import MarkingSystem
from ..core.searcher import Searcher


class TerritorialStrategy(Strategy):
    """
    Divide path into segments and claim a territory.

    Each searcher works in their own segment first, reducing
    conflicts. Falls back to greedy behavior when territory exhausted.
    """

    def __init__(self, num_searchers: int = 1):
        """
        Initialize territorial strategy.

        Args:
            num_searchers: Total number of searchers (for segment calculation)
        """
        super().__init__(name="territorial")
        self.num_searchers = num_searchers

    def get_next_cell(
        self,
        searcher: Searcher,
        path: Path,
        marks: MarkingSystem,
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Get next cell in searcher's territory.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Not directly used

        Returns:
            ID of next cell in territory, or nearest unmarked if territory exhausted
        """
        path_length = len(path)

        # Calculate this searcher's segment
        segment_size = path_length // max(1, self.num_searchers)
        my_segment_start = searcher.id * segment_size
        my_segment_end = min(my_segment_start + segment_size, path_length)

        # Handle remainder for last segment
        if searcher.id == self.num_searchers - 1:
            my_segment_end = path_length

        # Search within territory first
        for cell_id in range(my_segment_start, my_segment_end):
            if marks.is_available(cell_id):
                return cell_id

        # Territory exhausted - fall back to greedy nearest
        unmarked = marks.get_unmarked_cells()
        if not unmarked:
            return None

        current_pos = searcher.position
        return min(unmarked, key=lambda c: abs(c - current_pos))

    def set_num_searchers(self, num: int) -> None:
        """Update the number of searchers for segment calculation."""
        self.num_searchers = max(1, num)
