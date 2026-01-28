"""
Stigmergic Marking System for indirect agent coordination.

Agents communicate through environment modification - marking cells
as searched, in-progress, or found. This enables coordination without
direct agent-to-agent communication.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .cell import CellStatus
import time as time_module


@dataclass
class Mark:
    """
    Marking information for a single cell.

    Attributes:
        status: Current search status of the cell
        searcher_id: ID of the searcher who last modified this mark
        timestamp: When the mark was last updated
        search_count: Number of times this cell has been searched
    """
    status: CellStatus = CellStatus.UNMARKED
    searcher_id: Optional[int] = None
    timestamp: Optional[float] = None
    search_count: int = 0


class MarkingSystem:
    """
    Shared marking system for stigmergic coordination.

    Allows searchers to communicate indirectly through environment
    modification. Each cell can be marked with a status indicating
    whether it has been searched, is currently being searched, or
    contains the found key.

    Attributes:
        marks: Dictionary mapping cell IDs to their marks
        num_cells: Total number of cells in the path
        mark_read_time: Time cost to read marking information
        mark_history: Log of all marking events
    """

    def __init__(self, num_cells: int, mark_read_time: float = 0.01):
        """
        Initialize the marking system.

        Args:
            num_cells: Number of cells to track
            mark_read_time: Time cost for reading marks (coordination overhead)
        """
        self.marks: Dict[int, Mark] = {
            i: Mark() for i in range(num_cells)
        }
        self.num_cells = num_cells
        self.mark_read_time = mark_read_time
        self.mark_history: List[Tuple[float, int, CellStatus, Optional[int]]] = []

    def mark(
        self,
        cell_id: int,
        status: CellStatus,
        searcher_id: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update marking for a cell.

        Args:
            cell_id: ID of the cell to mark
            status: New status for the cell
            searcher_id: ID of the searcher making the mark
            timestamp: Time of marking (uses current time if None)
        """
        if cell_id not in self.marks:
            raise ValueError(f"Invalid cell ID: {cell_id}")

        if timestamp is None:
            timestamp = time_module.time()

        mark = self.marks[cell_id]
        mark.status = status
        mark.searcher_id = searcher_id
        mark.timestamp = timestamp

        if status == CellStatus.SEARCHED:
            mark.search_count += 1

        self.mark_history.append((timestamp, cell_id, status, searcher_id))

    def get_status(self, cell_id: int) -> CellStatus:
        """Get current status of a cell."""
        return self.marks[cell_id].status

    def get_mark(self, cell_id: int) -> Mark:
        """Get full mark information for a cell."""
        return self.marks[cell_id]

    def get_unmarked_cells(self) -> List[int]:
        """Return list of unmarked cell IDs."""
        return [i for i, m in self.marks.items()
                if m.status == CellStatus.UNMARKED]

    def get_available_cells(self) -> List[int]:
        """
        Return cells available for searching.

        Currently same as unmarked, but could be extended to include
        cells that need re-searching (with imperfect detection).
        """
        return [i for i, m in self.marks.items()
                if m.status == CellStatus.UNMARKED]

    def get_searched_cells(self) -> List[int]:
        """Return list of searched cell IDs."""
        return [i for i, m in self.marks.items()
                if m.status == CellStatus.SEARCHED]

    def get_in_progress_cells(self) -> List[int]:
        """Return list of cells currently being searched."""
        return [i for i, m in self.marks.items()
                if m.status == CellStatus.IN_PROGRESS]

    def is_available(self, cell_id: int) -> bool:
        """Check if cell is available for searching."""
        return self.marks[cell_id].status == CellStatus.UNMARKED

    def is_searched(self, cell_id: int) -> bool:
        """Check if cell has been searched."""
        return self.marks[cell_id].status in [CellStatus.SEARCHED, CellStatus.FOUND]

    def count_searched(self) -> int:
        """Count number of searched cells."""
        return sum(1 for m in self.marks.values()
                   if m.status in [CellStatus.SEARCHED, CellStatus.FOUND])

    def count_unmarked(self) -> int:
        """Count number of unmarked cells."""
        return sum(1 for m in self.marks.values()
                   if m.status == CellStatus.UNMARKED)

    def count_by_status(self) -> Dict[CellStatus, int]:
        """Count cells by status."""
        counts = {status: 0 for status in CellStatus}
        for mark in self.marks.values():
            counts[mark.status] += 1
        return counts

    def get_redundant_searches(self) -> int:
        """Count total redundant searches (search_count > 1)."""
        return sum(max(0, m.search_count - 1) for m in self.marks.values())

    def get_total_searches(self) -> int:
        """Get total number of search actions performed."""
        return sum(m.search_count for m in self.marks.values())

    def clear_in_progress(self, timeout: float = 5.0, current_time: Optional[float] = None) -> List[int]:
        """
        Clear stale in_progress marks.

        If a searcher fails mid-search, cells may be stuck as IN_PROGRESS.
        This method clears marks that have been in_progress too long.

        Args:
            timeout: Seconds after which to consider a mark stale
            current_time: Current time (uses actual time if None)

        Returns:
            List of cell IDs that were cleared
        """
        if current_time is None:
            current_time = time_module.time()

        cleared = []
        for cell_id, mark in self.marks.items():
            if mark.status == CellStatus.IN_PROGRESS:
                if mark.timestamp and (current_time - mark.timestamp) > timeout:
                    mark.status = CellStatus.UNMARKED
                    cleared.append(cell_id)

        return cleared

    def reset(self) -> None:
        """Reset all marks to initial state."""
        for cell_id in self.marks:
            self.marks[cell_id] = Mark()
        self.mark_history.clear()

    def get_searcher_cells(self, searcher_id: int) -> List[int]:
        """Get list of cells marked by a specific searcher."""
        return [cell_id for cell_id, mark in self.marks.items()
                if mark.searcher_id == searcher_id]

    def get_coverage_percentage(self) -> float:
        """Get percentage of cells that have been searched."""
        if self.num_cells == 0:
            return 0.0
        return self.count_searched() / self.num_cells * 100

    def copy(self) -> 'MarkingSystem':
        """Create a copy of the marking system (for benchmarking)."""
        new_system = MarkingSystem(self.num_cells, self.mark_read_time)
        for cell_id, mark in self.marks.items():
            new_system.marks[cell_id] = Mark(
                status=mark.status,
                searcher_id=mark.searcher_id,
                timestamp=mark.timestamp,
                search_count=mark.search_count
            )
        return new_system
