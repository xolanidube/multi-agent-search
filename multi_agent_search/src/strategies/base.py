"""
Base Strategy class for search strategies.

A strategy is a function that maps the current state to the next
cell to search. Different strategies have different trade-offs
in terms of coverage efficiency, coordination, and adaptability.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.cell import Path
    from ..core.marking_system import MarkingSystem
    from ..core.searcher import Searcher


class Strategy(ABC):
    """
    Abstract base class for search strategies.

    Each strategy must implement get_next_cell() which determines
    the next cell to search based on current state.

    Strategies can use:
    - Searcher's current position
    - Marking system state (which cells are searched)
    - Path structure and cell properties
    - Other searchers' positions (for coordination)
    """

    def __init__(self, name: str):
        """
        Initialize the strategy.

        Args:
            name: Human-readable name for the strategy
        """
        self.name = name

    @abstractmethod
    def get_next_cell(
        self,
        searcher: 'Searcher',
        path: 'Path',
        marks: 'MarkingSystem',
        other_positions: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Determine the next cell to search.

        Args:
            searcher: The searcher using this strategy
            path: The search path
            marks: Current marking system state
            other_positions: Positions of other searchers

        Returns:
            Cell ID to search next, or None if no available cells
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
