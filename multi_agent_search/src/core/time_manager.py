"""
Time Budget Management for time-constrained search.

Manages the partition of total time budget T into:
- T_sample: Time for strategy calibration/benchmarking
- T_search: Time for actual search execution

The key insight is that spending time upfront to discover the best
strategy can improve overall search efficiency.
"""

from enum import Enum
from typing import Optional
import time as time_module
import math


class Phase(Enum):
    """Phases of the search operation."""
    IDLE = "idle"
    SAMPLING = "sampling"
    SEARCHING = "searching"
    COMPLETE = "complete"
    FAILED = "failed"


class TimeManager:
    """
    Manages time budget for the search operation.

    Handles the partition of total time into sampling and search phases,
    tracks elapsed time, and determines phase transitions.

    The optimal sampling ratio depends on:
    - Total time budget T
    - Mean strategy throughput θ̄
    - Heterogeneity of the terrain

    Theoretical optimal: α* ≈ 1 / (1 + √(T·θ̄))
    """

    def __init__(self, total_budget: float = 60.0):
        """
        Initialize the time manager.

        Args:
            total_budget: Total time available in seconds
        """
        self.total_budget = total_budget
        self.start_time: Optional[float] = None
        self.sample_budget: Optional[float] = None
        self.search_budget: Optional[float] = None
        self.phase = Phase.IDLE
        self.phase_start_times: dict = {}

        # Simulation time (can be different from wall clock)
        self.simulation_time = 0.0
        self.use_simulation_time = False

    def allocate_budgets(self, sample_ratio: Optional[float] = None,
                         mean_throughput: float = 1.0) -> None:
        """
        Allocate time budgets between sampling and searching.

        Args:
            sample_ratio: Fraction of time for sampling (0-1).
                         If None, uses theoretical optimal.
            mean_throughput: Expected cells per second (for optimal calculation)
        """
        if sample_ratio is None:
            # Calculate theoretically optimal sampling ratio
            # α* ≈ 1 / (1 + √(T·θ̄))
            sample_ratio = 1.0 / (1.0 + math.sqrt(self.total_budget * mean_throughput))
            # Clamp to reasonable range
            sample_ratio = max(0.05, min(0.25, sample_ratio))

        self.sample_budget = self.total_budget * sample_ratio
        self.search_budget = self.total_budget * (1 - sample_ratio)

    def start(self) -> None:
        """Start the timer and enter sampling phase."""
        self.start_time = time_module.time()
        self.simulation_time = 0.0
        self.phase = Phase.SAMPLING
        self.phase_start_times[Phase.SAMPLING] = self.start_time

    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.use_simulation_time:
            return self.simulation_time
        if self.start_time is None:
            return 0.0
        return time_module.time() - self.start_time

    def remaining(self) -> float:
        """Get remaining time in total budget."""
        return max(0.0, self.total_budget - self.elapsed())

    def sample_time_remaining(self) -> float:
        """Get remaining time in sampling phase."""
        if self.phase != Phase.SAMPLING:
            return 0.0
        if self.sample_budget is None:
            return 0.0
        return max(0.0, self.sample_budget - self.elapsed())

    def search_time_remaining(self) -> float:
        """Get remaining time for search phase."""
        if self.phase == Phase.SEARCHING:
            return self.remaining()
        elif self.search_budget is not None:
            return self.search_budget
        return self.remaining()

    def transition_to_search(self) -> None:
        """Transition from sampling to search phase."""
        self.phase = Phase.SEARCHING
        self.phase_start_times[Phase.SEARCHING] = time_module.time()

    def mark_complete(self, success: bool = True) -> None:
        """Mark the search as complete."""
        self.phase = Phase.COMPLETE if success else Phase.FAILED
        self.phase_start_times[self.phase] = time_module.time()

    def is_expired(self) -> bool:
        """Check if total time budget is exhausted."""
        return self.elapsed() >= self.total_budget

    def is_sample_phase_expired(self) -> bool:
        """Check if sampling phase time is exhausted."""
        if self.sample_budget is None:
            return True
        return self.elapsed() >= self.sample_budget

    def get_phase(self) -> Phase:
        """Get current phase."""
        return self.phase

    def get_urgency(self, cells_remaining: int, max_throughput: float) -> float:
        """
        Calculate urgency level based on time pressure.

        Urgency = cells_remaining / (time_remaining * max_throughput)

        Returns:
            Urgency ratio:
            - < 1.0: On track, comfortable pace
            - 1.0-1.5: Tight but possible
            - > 1.5: Mathematically unlikely to succeed
        """
        time_left = self.remaining()
        if time_left <= 0:
            return float('inf')
        if max_throughput <= 0:
            return float('inf')
        return cells_remaining / (time_left * max_throughput)

    def should_trigger_emergency(self, cells_remaining: int,
                                   max_throughput: float,
                                   threshold: float = 1.2) -> bool:
        """
        Check if emergency protocol should be triggered.

        Args:
            cells_remaining: Number of cells still to search
            max_throughput: Maximum possible throughput
            threshold: Urgency level that triggers emergency

        Returns:
            True if emergency measures needed
        """
        return self.get_urgency(cells_remaining, max_throughput) > threshold

    def get_time_pressure_action(self, cells_remaining: int,
                                   max_throughput: float) -> str:
        """
        Recommend action based on time pressure.

        Time-Pressure Decision Matrix:
        - > 50% time, < 30% cells: maintain current strategy
        - > 50% time, > 70% cells: increase parallelism
        - 20-50% time, > 50% cells: switch to fastest strategy
        - < 20% time, > 30% cells: emergency protocol
        - < 10% time, any: probabilistic hot-zone only
        """
        time_ratio = self.remaining() / self.total_budget
        # Assume total cells = cells_remaining at start, approximate coverage
        # This is simplified; actual implementation would track total cells

        urgency = self.get_urgency(cells_remaining, max_throughput)

        if time_ratio > 0.5:
            if urgency < 0.5:
                return 'maintain_current'
            else:
                return 'increase_parallelism'
        elif time_ratio > 0.2:
            if urgency > 0.8:
                return 'switch_to_fastest'
            else:
                return 'maintain_current'
        elif time_ratio > 0.1:
            return 'emergency_greedy'
        else:
            return 'probabilistic_hotzone_only'

    def add_simulation_time(self, delta: float) -> None:
        """Add time to the simulation clock."""
        self.simulation_time += delta

    def set_simulation_mode(self, enabled: bool = True) -> None:
        """Enable or disable simulation time mode."""
        self.use_simulation_time = enabled

    def get_phase_duration(self, phase: Phase) -> float:
        """Get duration spent in a specific phase."""
        if phase not in self.phase_start_times:
            return 0.0

        start = self.phase_start_times[phase]

        # Find next phase start time or current time
        phase_order = [Phase.IDLE, Phase.SAMPLING, Phase.SEARCHING,
                       Phase.COMPLETE, Phase.FAILED]
        try:
            idx = phase_order.index(phase)
            for next_phase in phase_order[idx + 1:]:
                if next_phase in self.phase_start_times:
                    return self.phase_start_times[next_phase] - start
        except (ValueError, IndexError):
            pass

        # Phase is still ongoing
        return time_module.time() - start

    def get_statistics(self) -> dict:
        """Get timing statistics."""
        return {
            'total_budget': self.total_budget,
            'sample_budget': self.sample_budget,
            'search_budget': self.search_budget,
            'elapsed': self.elapsed(),
            'remaining': self.remaining(),
            'phase': self.phase.value,
            'sample_phase_duration': self.get_phase_duration(Phase.SAMPLING),
            'search_phase_duration': self.get_phase_duration(Phase.SEARCHING),
        }
