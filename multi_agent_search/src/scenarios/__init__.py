"""
Realistic search scenarios.

Contains simulations of real-world search problems:
- File system search
- Network topology exploration
- And more to come...
"""

from .file_system_search import (
    FileSystemSimulator,
    FileSearchAgent,
    run_file_search_experiment,
    compare_file_search_strategies
)

__all__ = [
    'FileSystemSimulator',
    'FileSearchAgent',
    'run_file_search_experiment',
    'compare_file_search_strategies'
]
