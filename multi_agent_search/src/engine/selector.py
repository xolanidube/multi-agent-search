"""
Strategy Selection based on benchmark results.

Selects the optimal strategy or strategy mix based on:
- Benchmark performance (throughput, coverage)
- Time remaining
- Number of searchers
"""

from typing import Dict, List, Tuple, Optional
from .benchmark import BenchmarkResult
from ..strategies.base import Strategy


class StrategySelector:
    """
    Select optimal strategy based on benchmark results and constraints.

    Handles both single-strategy selection and multi-agent strategy
    allocation for diverse strategy mixing.
    """

    def __init__(
        self,
        available_strategies: Dict[str, Strategy],
        throughput_weight: float = 0.7,
        coverage_weight: float = 0.3
    ):
        """
        Initialize the selector.

        Args:
            available_strategies: Dictionary of strategy name -> Strategy
            throughput_weight: Weight for throughput in scoring
            coverage_weight: Weight for coverage in scoring
        """
        self.strategies = available_strategies
        self.throughput_weight = throughput_weight
        self.coverage_weight = coverage_weight

        self.benchmark_results: Optional[Dict[str, BenchmarkResult]] = None
        self.selected_strategy: Optional[str] = None
        self.confidence: float = 0.0

    def select_best(
        self,
        benchmark_results: Dict[str, BenchmarkResult],
        remaining_time: float,
        path_length: int
    ) -> Tuple[str, float, str]:
        """
        Select the best strategy based on benchmark results.

        Algorithm:
        1. Calculate required throughput to cover path in remaining time
        2. Find strategies that meet this requirement (with 80% margin)
        3. Among viable strategies, select highest overall score
        4. If none viable, fall back to fastest

        Args:
            benchmark_results: Results from benchmarking
            remaining_time: Time left for search
            path_length: Number of cells to search

        Returns:
            Tuple of (strategy_name, confidence, selection_method)
        """
        self.benchmark_results = benchmark_results

        # Calculate required throughput
        if remaining_time <= 0:
            remaining_time = 1.0  # Avoid division by zero
        required_throughput = path_length / remaining_time

        # Find viable strategies (those that can finish in time with margin)
        viable_strategies = []
        for name, result in benchmark_results.items():
            if result.avg_throughput >= required_throughput * 0.8:
                viable_strategies.append((name, result))

        if not viable_strategies:
            # No strategy meets requirement - pick fastest and hope
            best_name = max(
                benchmark_results.keys(),
                key=lambda n: benchmark_results[n].avg_throughput
            )
            best_result = benchmark_results[best_name]
            self.selected_strategy = best_name
            self.confidence = best_result.avg_throughput / required_throughput
            return best_name, self.confidence, 'fallback_fastest'

        # Among viable, pick highest overall score
        best_name, best_result = max(
            viable_strategies,
            key=lambda x: self._compute_score(x[1])
        )

        self.selected_strategy = best_name
        self.confidence = min(1.0, best_result.avg_throughput / required_throughput)

        return best_name, self.confidence, 'optimal_selection'

    def _compute_score(self, result: BenchmarkResult) -> float:
        """Compute weighted score for a benchmark result."""
        return (
            self.throughput_weight * result.avg_throughput +
            self.coverage_weight * result.avg_coverage
        )

    def recommend_strategy_mix(
        self,
        benchmark_results: Dict[str, BenchmarkResult],
        num_searchers: int
    ) -> Dict[str, int]:
        """
        Recommend strategy allocation for multiple searchers.

        Distributes searchers across strategies based on performance,
        with better strategies getting more searchers.

        Args:
            benchmark_results: Results from benchmarking
            num_searchers: Number of searchers to allocate

        Returns:
            Dictionary of strategy_name -> number of searchers
        """
        if num_searchers <= 0:
            return {}

        # Sort strategies by score
        sorted_strategies = sorted(
            benchmark_results.items(),
            key=lambda x: self._compute_score(x[1]),
            reverse=True
        )

        allocation = {}

        if num_searchers == 1:
            # Single searcher gets best strategy
            allocation[sorted_strategies[0][0]] = 1

        elif num_searchers == 2:
            # Two searchers: best and second-best
            allocation[sorted_strategies[0][0]] = 1
            if len(sorted_strategies) > 1:
                allocation[sorted_strategies[1][0]] = 1
            else:
                allocation[sorted_strategies[0][0]] = 2

        else:
            # Multiple searchers: weighted allocation
            top_strategies = sorted_strategies[:min(3, len(sorted_strategies))]
            total_score = sum(self._compute_score(r) for _, r in top_strategies)

            if total_score == 0:
                # Uniform distribution
                per_strategy = num_searchers // len(top_strategies)
                for name, _ in top_strategies:
                    allocation[name] = per_strategy
            else:
                # Weighted by score
                allocated = 0
                for i, (name, result) in enumerate(top_strategies):
                    if i == len(top_strategies) - 1:
                        # Last strategy gets remaining
                        allocation[name] = num_searchers - allocated
                    else:
                        score_ratio = self._compute_score(result) / total_score
                        count = max(1, int(num_searchers * score_ratio))
                        allocation[name] = count
                        allocated += count

        return allocation

    def get_strategy_for_searcher(
        self,
        searcher_id: int,
        allocation: Dict[str, int]
    ) -> str:
        """
        Get strategy for a specific searcher based on allocation.

        Args:
            searcher_id: ID of the searcher
            allocation: Strategy allocation dictionary

        Returns:
            Strategy name for this searcher
        """
        # Build list of strategies in order
        strategy_list = []
        for name, count in allocation.items():
            strategy_list.extend([name] * count)

        if not strategy_list:
            return list(self.strategies.keys())[0]

        idx = searcher_id % len(strategy_list)
        return strategy_list[idx]

    def should_switch_strategy(
        self,
        current_performance: float,
        benchmark_prediction: float,
        time_remaining_ratio: float
    ) -> bool:
        """
        Determine if strategy should be switched mid-search.

        Args:
            current_performance: Actual throughput achieved
            benchmark_prediction: Predicted throughput from benchmark
            time_remaining_ratio: Fraction of time remaining

        Returns:
            True if should switch to different strategy
        """
        if benchmark_prediction <= 0:
            return False

        performance_gap = (benchmark_prediction - current_performance) / benchmark_prediction

        # Switch if underperforming by >20% and have >30% time left
        return performance_gap > 0.2 and time_remaining_ratio > 0.3

    def get_ranking(
        self,
        benchmark_results: Dict[str, BenchmarkResult]
    ) -> List[Tuple[str, float]]:
        """
        Get strategies ranked by score.

        Args:
            benchmark_results: Benchmark results

        Returns:
            List of (strategy_name, score) sorted by score descending
        """
        return sorted(
            [(name, self._compute_score(r)) for name, r in benchmark_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
