"""
Run file system search experiments.

Simulates searching for a file among thousands/millions of files
to demonstrate the search framework in a realistic scenario.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scenarios.file_system_search import (
    FileSystemSimulator,
    FileSearchAgent,
    run_file_search_experiment,
    compare_file_search_strategies
)


def demo_small():
    """Demo with small file system."""
    print("\n" + "=" * 60)
    print("DEMO: Small File System (10,000 files)")
    print("=" * 60)

    result = run_file_search_experiment(
        total_files=10_000,
        strategy="bfs",
        time_budget_ms=60000,
        seed=42
    )

    print(f"\nResult:")
    print(f"  Found: {result['found']}")
    print(f"  Search time: {result['search_time_ms']:.1f} ms")
    print(f"  Files checked: {result['files_checked']}")
    print(f"  Directories listed: {result['dirs_listed']}")
    print(f"  Target path: {result['target_path']}")
    print(f"  Target depth: {result['target_depth']}")


def demo_medium():
    """Demo with medium file system."""
    print("\n" + "=" * 60)
    print("DEMO: Medium File System (100,000 files)")
    print("=" * 60)

    result = run_file_search_experiment(
        total_files=100_000,
        strategy="dfs",
        time_budget_ms=120000,
        seed=123
    )

    print(f"\nResult:")
    print(f"  Found: {result['found']}")
    print(f"  Search time: {result['search_time_ms']:.1f} ms")
    print(f"  Files checked: {result['files_checked']}")
    print(f"  Directories listed: {result['dirs_listed']}")
    print(f"  Target depth: {result['target_depth']}")


def demo_large():
    """Demo with large file system (simulating millions)."""
    print("\n" + "=" * 60)
    print("DEMO: Large File System (500,000 files)")
    print("=" * 60)

    result = run_file_search_experiment(
        total_files=500_000,
        strategy="probabilistic",
        time_budget_ms=300000,
        seed=456
    )

    print(f"\nResult:")
    print(f"  Found: {result['found']}")
    print(f"  Search time: {result['search_time_ms']:.1f} ms")
    print(f"  Files checked: {result['files_checked']}")
    print(f"  Directories listed: {result['dirs_listed']}")
    print(f"  Total directories: {result['total_dirs']}")


def compare_strategies():
    """Compare all strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON (50,000 files, 5 runs each)")
    print("=" * 60)

    results = compare_file_search_strategies(
        total_files=50_000,
        time_budget_ms=60000,
        runs=5
    )

    return results


def main():
    print("=" * 60)
    print("FILE SYSTEM SEARCH EXPERIMENTS")
    print("Searching for a needle in a digital haystack")
    print("=" * 60)

    # Run demos
    demo_small()
    demo_medium()

    # Skip very large demo by default (takes time)
    # demo_large()

    # Strategy comparison
    compare_strategies()

    print("\n" + "=" * 60)
    print("Experiments complete!")


if __name__ == "__main__":
    main()
