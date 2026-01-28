"""
Publication-Quality Plotting for Research Paper.

Generates figures for:
1. Sampling ratio analysis
2. Scaling analysis
3. Strategy comparison
4. Time efficiency analysis
5. Coverage patterns
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B1F2B',      # Dark
    'light': '#95B8D1',        # Light blue
}

STRATEGY_COLORS = {
    'with_selection': '#2E86AB',
    'greedy_only': '#F18F01',
    'random_only': '#C73E1D',
    'greedy_nearest': '#2E86AB',
    'territorial': '#A23B72',
    'probabilistic': '#F18F01',
    'contrarian': '#95B8D1',
    'random_walk': '#C73E1D',
}


class ResearchPlotter:
    """Creates publication-quality plots for the research paper."""

    def __init__(self, results_dir: str = "./results", output_dir: str = "./figures"):
        """
        Initialize the plotter.

        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory to save figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.sampling_results = self._load_json("sampling_study.json")
        self.scaling_results = self._load_json("scaling_study.json")

    def _load_json(self, filename: str) -> Optional[Dict]:
        """Load JSON results file."""
        filepath = self.results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return None

    def plot_sampling_ratio_analysis(self, save: bool = True) -> plt.Figure:
        """
        Plot the effect of sampling ratio on success rate and coverage.

        Creates a dual-axis plot showing:
        - Success rate vs sampling ratio (primary axis)
        - Coverage vs sampling ratio (secondary axis)
        """
        if not self.sampling_results:
            print("No sampling results found")
            return None

        fig, ax1 = plt.subplots(figsize=(7, 5))

        # Parse data
        ratios = sorted([float(r) for r in self.sampling_results.keys()])
        success_rates = [self.sampling_results[str(r)]['success_rate'] * 100 for r in ratios]
        coverages = [self.sampling_results[str(r)]['avg_coverage'] * 100 for r in ratios]
        efficiencies = [self.sampling_results[str(r)]['avg_time_efficiency'] * 100 for r in ratios]

        # Convert ratios to percentages for display
        ratio_pcts = [r * 100 for r in ratios]

        # Primary axis: Success rate
        color1 = COLORS['primary']
        ax1.set_xlabel('Sampling Ratio (%)')
        ax1.set_ylabel('Success Rate (%)', color=color1)
        line1 = ax1.plot(ratio_pcts, success_rates, 'o-', color=color1,
                         linewidth=2, markersize=8, label='Success Rate')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([60, 100])

        # Mark optimal point
        max_idx = np.argmax(success_rates)
        ax1.annotate(f'Optimal: {ratio_pcts[max_idx]:.0f}%',
                     xy=(ratio_pcts[max_idx], success_rates[max_idx]),
                     xytext=(ratio_pcts[max_idx] + 3, success_rates[max_idx] - 5),
                     fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='gray'))

        # Secondary axis: Coverage
        ax2 = ax1.twinx()
        color2 = COLORS['secondary']
        ax2.set_ylabel('Coverage (%)', color=color2)
        line2 = ax2.plot(ratio_pcts, coverages, 's--', color=color2,
                         linewidth=2, markersize=7, label='Coverage')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([40, 70])

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')

        ax1.set_title('Effect of Sampling Ratio on Search Performance')
        ax1.set_xticks(ratio_pcts)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_sampling_ratio.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_sampling_ratio.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_scaling_analysis(self, save: bool = True) -> plt.Figure:
        """
        Plot how performance scales with number of searchers.

        Creates a multi-panel figure showing:
        - Success rate vs searchers
        - Coverage vs searchers
        - Time vs searchers
        - Cells per agent vs searchers
        """
        if not self.scaling_results:
            print("No scaling results found")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Parse data
        searchers = sorted([int(s) for s in self.scaling_results.keys()])
        success_rates = [self.scaling_results[str(s)]['success_rate'] * 100 for s in searchers]
        coverages = [self.scaling_results[str(s)]['avg_coverage'] * 100 for s in searchers]
        times = [self.scaling_results[str(s)]['avg_time'] for s in searchers]
        cells_per_agent = [self.scaling_results[str(s)]['cells_per_searcher'] for s in searchers]

        # Plot 1: Success Rate
        ax1 = axes[0, 0]
        ax1.plot(searchers, success_rates, 'o-', color=COLORS['primary'],
                 linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Searchers')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('(a) Success Rate vs. Searchers')
        ax1.set_ylim([0, 100])
        ax1.axhline(y=82, color='gray', linestyle=':', alpha=0.7, label='Single agent baseline')
        ax1.legend(fontsize=9)

        # Plot 2: Coverage
        ax2 = axes[0, 1]
        ax2.plot(searchers, coverages, 's-', color=COLORS['secondary'],
                 linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Searchers')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('(b) Coverage vs. Searchers')
        ax2.set_ylim([0, 80])

        # Plot 3: Time to Success
        ax3 = axes[1, 0]
        ax3.plot(searchers, times, '^-', color=COLORS['tertiary'],
                 linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Searchers')
        ax3.set_ylabel('Avg. Time to Success (s)')
        ax3.set_title('(c) Search Time vs. Searchers')
        ax3.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Time budget')
        ax3.legend(fontsize=9)

        # Plot 4: Efficiency (cells per agent)
        ax4 = axes[1, 1]
        ax4.plot(searchers, cells_per_agent, 'd-', color=COLORS['success'],
                 linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Searchers')
        ax4.set_ylabel('Cells per Agent')
        ax4.set_title('(d) Individual Agent Contribution')

        # Add theoretical scaling line
        theoretical = [cells_per_agent[0] / s for s in searchers]
        ax4.plot(searchers, theoretical, ':', color='gray', alpha=0.7,
                 label='Theoretical (no overhead)')
        ax4.legend(fontsize=9)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_scaling_analysis.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_scaling_analysis.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_strategy_comparison(self, save: bool = True) -> plt.Figure:
        """
        Plot comparison of different strategy selection approaches.

        Bar chart comparing:
        - With online strategy selection
        - Greedy only (no selection)
        - Random only (baseline)
        """
        # Data from experiments
        approaches = ['With Selection', 'Greedy Only', 'Random Only']
        success_rates = [78, 74, 20]
        coverages = [52.6, 64.5, 22.0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(approaches))
        width = 0.6

        # Success rate bars
        colors = [COLORS['primary'], COLORS['tertiary'], COLORS['success']]
        bars1 = ax1.bar(x, success_rates, width, color=colors, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('(a) Success Rate by Approach')
        ax1.set_xticks(x)
        ax1.set_xticklabels(approaches, rotation=15, ha='right')
        ax1.set_ylim([0, 100])

        # Add value labels
        for bar, val in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{val}%', ha='center', va='bottom', fontsize=10)

        # Coverage bars
        bars2 = ax2.bar(x, coverages, width, color=colors, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('(b) Coverage by Approach')
        ax2.set_xticks(x)
        ax2.set_xticklabels(approaches, rotation=15, ha='right')
        ax2.set_ylim([0, 80])

        # Add value labels
        for bar, val in zip(bars2, coverages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_strategy_comparison.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_strategy_comparison.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_terrain_comparison(self, save: bool = True) -> plt.Figure:
        """
        Plot comparison of performance on different terrain types.
        """
        # Data from experiments
        terrains = ['Homogeneous', 'Heterogeneous']
        success_rates = [100, 74]
        coverages = [56.3, 59.0]
        times = [10.72, 16.14]

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))

        x = np.arange(len(terrains))
        width = 0.5
        colors = [COLORS['light'], COLORS['primary']]

        # Success rate
        axes[0].bar(x, success_rates, width, color=colors, edgecolor='black')
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('(a) Success Rate')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(terrains)
        axes[0].set_ylim([0, 110])
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 2, f'{v}%', ha='center', fontsize=10)

        # Coverage
        axes[1].bar(x, coverages, width, color=colors, edgecolor='black')
        axes[1].set_ylabel('Coverage (%)')
        axes[1].set_title('(b) Coverage')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(terrains)
        axes[1].set_ylim([0, 70])
        for i, v in enumerate(coverages):
            axes[1].text(i, v + 1, f'{v}%', ha='center', fontsize=10)

        # Time
        axes[2].bar(x, times, width, color=colors, edgecolor='black')
        axes[2].set_ylabel('Avg. Time (s)')
        axes[2].set_title('(c) Time to Success')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(terrains)
        axes[2].set_ylim([0, 20])
        for i, v in enumerate(times):
            axes[2].text(i, v + 0.5, f'{v:.1f}s', ha='center', fontsize=10)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_terrain_comparison.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_terrain_comparison.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_coordination_overhead(self, save: bool = True) -> plt.Figure:
        """
        Plot coordination overhead analysis.

        Shows how effective throughput degrades with more searchers
        compared to theoretical linear scaling.
        """
        if not self.scaling_results:
            print("No scaling results found")
            return None

        fig, ax = plt.subplots(figsize=(7, 5))

        searchers = sorted([int(s) for s in self.scaling_results.keys()])

        # Calculate effective throughput (cells searched per unit time, total)
        # Using cells_per_agent * num_searchers as proxy for total throughput
        cells_per_agent = [self.scaling_results[str(s)]['cells_per_searcher'] for s in searchers]
        coverages = [self.scaling_results[str(s)]['avg_coverage'] for s in searchers]

        # Actual total cells searched (coverage * 50 cells)
        total_cells = [c * 50 for c in coverages]

        # Theoretical linear scaling (if no coordination overhead)
        base_cells = total_cells[0]  # Single agent performance
        theoretical = [base_cells * s for s in searchers]

        # Effective throughput ratio
        efficiency_ratio = [actual / (theoretical[i] if theoretical[i] > 0 else 1)
                           for i, actual in enumerate(total_cells)]

        # Plot actual vs theoretical
        ax.plot(searchers, total_cells, 'o-', color=COLORS['primary'],
                linewidth=2, markersize=8, label='Actual cells searched')
        ax.plot(searchers, theoretical, '--', color='gray',
                linewidth=2, label='Theoretical (linear scaling)')

        # Fill the gap to show overhead
        ax.fill_between(searchers, total_cells, theoretical,
                        alpha=0.3, color=COLORS['success'],
                        label='Coordination overhead')

        ax.set_xlabel('Number of Searchers')
        ax.set_ylabel('Total Cells Searched')
        ax.set_title('Coordination Overhead: Actual vs. Theoretical Scaling')
        ax.legend(loc='upper left')
        ax.set_ylim([0, max(theoretical) * 1.1])

        # Add efficiency annotation
        ax.annotate(f'Efficiency at 5 agents: {efficiency_ratio[3]:.0%}',
                    xy=(5, total_cells[3]),
                    xytext=(6, total_cells[3] + 30),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'))

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_coordination_overhead.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_coordination_overhead.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_time_budget_partition(self, save: bool = True) -> plt.Figure:
        """
        Visualize the time budget partitioning concept.

        Creates a diagram showing how total time T is split into
        sampling time T_s and execution time T_x.
        """
        fig, ax = plt.subplots(figsize=(8, 3))

        # Time budget parameters
        total_time = 30
        sample_ratios = [0.0, 0.15, 0.30]

        y_positions = [2, 1, 0]
        bar_height = 0.6

        for i, (ratio, y) in enumerate(zip(sample_ratios, y_positions)):
            t_sample = total_time * ratio
            t_exec = total_time * (1 - ratio)

            # Sampling phase (yellow)
            if t_sample > 0:
                ax.barh(y, t_sample, height=bar_height,
                        color=COLORS['tertiary'], edgecolor='black',
                        label='Sampling (T_s)' if i == 0 else '')

            # Execution phase (blue)
            ax.barh(y, t_exec, left=t_sample, height=bar_height,
                    color=COLORS['primary'], edgecolor='black',
                    label='Execution (T_x)' if i == 0 else '')

            # Labels
            ax.text(-2, y, f'α = {ratio:.0%}', va='center', ha='right', fontsize=10)

            if t_sample > 0:
                ax.text(t_sample/2, y, f'{t_sample:.0f}s', va='center', ha='center',
                        fontsize=9, color='white', fontweight='bold')
            ax.text(t_sample + t_exec/2, y, f'{t_exec:.0f}s', va='center', ha='center',
                    fontsize=9, color='white', fontweight='bold')

        ax.set_xlim([-5, 35])
        ax.set_ylim([-0.5, 2.8])
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title('Time Budget Partitioning: T = T_s + T_x')

        # Custom legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['tertiary'], edgecolor='black',
                          label='Sampling Phase (T_s)'),
            mpatches.Patch(facecolor=COLORS['primary'], edgecolor='black',
                          label='Execution Phase (T_x)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.axvline(x=total_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(total_time + 0.5, 2.5, 'Deadline', color='red', fontsize=9)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "fig_time_partition.pdf"
            plt.savefig(filepath)
            plt.savefig(self.output_dir / "fig_time_partition.png")
            print(f"Saved: {filepath}")

        return fig

    def plot_all(self) -> None:
        """Generate all figures for the paper."""
        print("\nGenerating all figures...")
        print("=" * 50)

        self.plot_sampling_ratio_analysis()
        self.plot_scaling_analysis()
        self.plot_strategy_comparison()
        self.plot_terrain_comparison()
        self.plot_coordination_overhead()
        self.plot_time_budget_partition()

        print("=" * 50)
        print(f"All figures saved to: {self.output_dir}/")


def main():
    """Generate all research figures."""
    plotter = ResearchPlotter(
        results_dir="./results",
        output_dir="./figures"
    )
    plotter.plot_all()


if __name__ == "__main__":
    main()
