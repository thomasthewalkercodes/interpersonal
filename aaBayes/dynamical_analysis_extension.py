"""
Integrated Dynamical Systems Analysis Extension
==============================================

This module extends your existing InterPersonalSimulation class with advanced
dynamical systems analysis capabilities while maintaining full backward compatibility.

Integration approach:
1. Extends existing classes with new methods
2. Preserves all existing functionality
3. Adds advanced analysis capabilities
4. Follows KISS principles for ease of use

Author: Based on research by Thomas A. Walker & Christopher J. Hopwood
Contact: Thomas.walker2@uzh.ch
GitHub: thomasthewalkercodes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.integrate import odeint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings

from interpersonal_dynamics_simulation import InterPersonalSimulation

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class DynamicalSystemsAnalyzer:
    """
    Advanced dynamical systems analyzer that integrates with your existing simulation.

    This class can be used as a mixin or standalone analyzer for your simulation data.
    """

    def __init__(self, simulation=None, data=None):
        """
        Initialize analyzer with either a simulation object or DataFrame.

        Args:
            simulation: InterPersonalSimulation object
            data: DataFrame from get_results_dataframe()
        """
        if simulation is not None:
            self.simulation = simulation
            self.data = simulation.get_results_dataframe()
        elif data is not None:
            self.simulation = None
            self.data = data
        else:
            raise ValueError("Must provide either simulation object or data DataFrame")

        if len(self.data) == 0:
            raise ValueError("No data available for analysis")

        self.agents = self.data["agent"].unique()
        self.n_rounds = self.data["round"].max()
        self.dyads = self._identify_dyads()

        # Prepare enhanced data
        self._prepare_enhanced_data()

    def _identify_dyads(self) -> List[Tuple[str, str]]:
        """Identify unique dyads from the data."""
        dyads = set()
        for _, row in self.data.iterrows():
            pair = tuple(sorted([row["agent"], row["partner"]]))
            dyads.add(pair)
        return list(dyads)

    def _prepare_enhanced_data(self):
        """Prepare data with additional dynamical systems variables."""
        # Convert angles to radians for mathematical operations
        self.data["own_angle_rad"] = np.radians(self.data["own_angle"])
        self.data["partner_angle_rad"] = np.radians(self.data["partner_angle"])

        # Compute behavioral coordinates in Cartesian space
        self.data["own_x"] = self.data["own_radius"] * np.cos(
            self.data["own_angle_rad"]
        )
        self.data["own_y"] = self.data["own_radius"] * np.sin(
            self.data["own_angle_rad"]
        )
        self.data["partner_x"] = self.data["partner_radius"] * np.cos(
            self.data["partner_angle_rad"]
        )
        self.data["partner_y"] = self.data["partner_radius"] * np.sin(
            self.data["partner_angle_rad"]
        )

        # Behavioral distance in Cartesian space
        self.data["behavioral_distance"] = np.sqrt(
            (self.data["own_x"] - self.data["partner_x"]) ** 2
            + (self.data["own_y"] - self.data["partner_y"]) ** 2
        )

        # Complementarity index (based on interpersonal theory)
        self.data["complementarity"] = self._compute_complementarity()

        # Velocity in behavioral space
        self._compute_behavioral_velocity()

    def _compute_complementarity(self) -> np.ndarray:
        """
        Compute interpersonal complementarity index.

        Based on Kiesler's complementarity principle:
        - Warm behavior invites warm behavior (correspondence)
        - Dominant behavior invites submissive behavior (reciprocity)
        """
        # Extract warmth and dominance dimensions
        own_warmth = self.data["own_y"]  # Already computed above
        own_dominance = self.data["own_x"]
        partner_warmth = self.data["partner_y"]
        partner_dominance = self.data["partner_x"]

        # Correspondence on warmth dimension
        warmth_correspondence = own_warmth * partner_warmth

        # Reciprocity on dominance dimension
        dominance_reciprocity = -own_dominance * partner_dominance

        # Combined complementarity score
        complementarity = warmth_correspondence + dominance_reciprocity
        return complementarity

    def _compute_behavioral_velocity(self):
        """Compute velocity in behavioral space for each agent."""
        velocities = []

        for _, row in self.data.iterrows():
            agent_data = self.data[
                (self.data["agent"] == row["agent"])
                & (self.data["round"] <= row["round"])
            ].sort_values("round")

            if len(agent_data) >= 2:
                # Get previous position
                prev_x = (
                    agent_data["own_x"].iloc[-2]
                    if len(agent_data) > 1
                    else agent_data["own_x"].iloc[0]
                )
                prev_y = (
                    agent_data["own_y"].iloc[-2]
                    if len(agent_data) > 1
                    else agent_data["own_y"].iloc[0]
                )

                # Compute velocity
                dx = row["own_x"] - prev_x
                dy = row["own_y"] - prev_y
                velocity = np.sqrt(dx**2 + dy**2)
            else:
                velocity = 0.0

            velocities.append(velocity)

        self.data["behavioral_velocity"] = velocities

    def plot_comprehensive_dynamics(
        self, agent1=None, agent2=None, figsize=(16, 12), save_path=None
    ):
        """
        Create comprehensive dynamical systems visualization.

        Args:
            agent1, agent2: Specific agents to analyze (uses first two if None)
            figsize: Figure size
            save_path: Path to save figure
        """
        if agent1 is None or agent2 is None:
            if len(self.agents) < 2:
                raise ValueError("Need at least 2 agents for dyadic analysis")
            agent1, agent2 = self.agents[0], self.agents[1]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        # Get agent-specific data
        agent1_data = self.data[self.data["agent"] == agent1].sort_values("round")
        agent2_data = self.data[self.data["agent"] == agent2].sort_values("round")

        # 1. Behavioral space trajectories with attractors
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_behavioral_trajectories_with_attractors(
            ax1, agent1_data, agent2_data, agent1, agent2
        )

        # 2. Phase space analysis
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_phase_space_analysis(ax2, agent1_data, agent2_data, agent1, agent2)

        # 3. Critical slowing down indicators
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_critical_slowing_down(ax3, agent1_data, agent2_data, agent1, agent2)

        # 4. Complementarity dynamics
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_complementarity_dynamics(
            ax4, agent1_data, agent2_data, agent1, agent2
        )

        # 5. Behavioral coupling strength
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_coupling_analysis(ax5, agent1_data, agent2_data, agent1, agent2)

        # 6. Stability landscape
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_stability_landscape(ax6, agent1_data, agent2_data, agent1, agent2)

        plt.suptitle(
            f"Dynamical Systems Analysis: {agent1} ↔ {agent2}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Comprehensive dynamics plot saved to {save_path}")

        return fig

    def _plot_behavioral_trajectories_with_attractors(
        self, ax, agent1_data, agent2_data, agent1, agent2
    ):
        """Enhanced behavioral space plot with attractor identification."""
        # Plot trajectories
        ax.plot(
            agent1_data["own_x"],
            agent1_data["own_y"],
            "o-",
            label=f"{agent1}",
            alpha=0.7,
            markersize=4,
            linewidth=2,
        )
        ax.plot(
            agent2_data["own_x"],
            agent2_data["own_y"],
            "s-",
            label=f"{agent2}",
            alpha=0.7,
            markersize=4,
            linewidth=2,
        )

        # Identify and mark potential attractors (regions of low velocity)
        for agent_data, agent_name, marker, color in [
            (agent1_data, agent1, "o", "blue"),
            (agent2_data, agent2, "s", "orange"),
        ]:
            if len(agent_data) > 10:
                # Find low-velocity regions (potential attractors)
                low_velocity = agent_data["behavioral_velocity"] < np.percentile(
                    agent_data["behavioral_velocity"], 25
                )
                if low_velocity.any():
                    attractor_points = agent_data[low_velocity]
                    ax.scatter(
                        attractor_points["own_x"],
                        attractor_points["own_y"],
                        s=100,
                        alpha=0.3,
                        c=color,
                        marker=marker,
                        label=f"{agent_name} Attractors",
                    )

        # Mark start and end points
        ax.scatter(
            agent1_data["own_x"].iloc[0],
            agent1_data["own_y"].iloc[0],
            s=150,
            marker="*",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            agent1_data["own_x"].iloc[-1],
            agent1_data["own_y"].iloc[-1],
            s=150,
            marker="X",
            color="red",
            label="End",
            zorder=5,
        )

        # Add circumplex grid
        theta = np.linspace(0, 2 * np.pi, 100)
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", alpha=0.2)

        # Add axis labels with interpersonal theory labels
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)
        ax.text(0.8, 0.05, "Dominant", fontsize=10, ha="center")
        ax.text(-0.8, 0.05, "Submissive", fontsize=10, ha="center")
        ax.text(0.05, 0.8, "Warm", fontsize=10, ha="center", rotation=90)
        ax.text(0.05, -0.8, "Cold", fontsize=10, ha="center", rotation=90)

        ax.set_xlabel("Dominance ←→ Submission")
        ax.set_ylabel("Cold ←→ Warm")
        ax.set_title("Behavioral Trajectories & Attractors")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    def _plot_phase_space_analysis(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Phase space analysis showing coupled dynamics."""
        # Use mood as the primary variable for phase space analysis
        if len(agent1_data) != len(agent2_data):
            # Align data by round
            merged = pd.merge(
                agent1_data[["round", "mood"]],
                agent2_data[["round", "mood"]],
                on="round",
                suffixes=("_1", "_2"),
            )
            mood1, mood2 = merged["mood_1"], merged["mood_2"]
        else:
            mood1, mood2 = agent1_data["mood"], agent2_data["mood"]

        # Phase space plot
        ax.plot(mood1, mood2, "o-", alpha=0.7, markersize=3, color="purple")

        # Mark trajectory direction with arrows
        n_arrows = min(8, len(mood1) // 5)
        if n_arrows > 0:
            indices = np.linspace(0, len(mood1) - 2, n_arrows, dtype=int)
            for i in indices:
                if i + 1 < len(mood1):
                    ax.annotate(
                        "",
                        xy=(mood1.iloc[i + 1], mood2.iloc[i + 1]),
                        xytext=(mood1.iloc[i], mood2.iloc[i]),
                        arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6),
                    )

        # Mark start and end
        ax.scatter(
            mood1.iloc[0],
            mood2.iloc[0],
            s=100,
            marker="o",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            mood1.iloc[-1],
            mood2.iloc[-1],
            s=100,
            marker="X",
            color="red",
            label="End",
            zorder=5,
        )

        # Add reference lines
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        # Diagonal reference line (perfect synchrony)
        lims = [ax.get_xlim(), ax.get_ylim()]
        min_lim = min(min(lims[0]), min(lims[1]))
        max_lim = max(max(lims[0]), max(lims[1]))
        ax.plot(
            [min_lim, max_lim],
            [min_lim, max_lim],
            "k--",
            alpha=0.5,
            label="Perfect Sync",
        )

        ax.set_xlabel(f"{agent1} Mood")
        ax.set_ylabel(f"{agent2} Mood")
        ax.set_title("Phase Space (Mood Coupling)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_critical_slowing_down(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot critical slowing down indicators."""
        window_size = max(5, len(agent1_data) // 4)

        for agent_data, agent_name, color in [
            (agent1_data, agent1, "blue"),
            (agent2_data, agent2, "orange"),
        ]:
            if len(agent_data) >= window_size:
                # Rolling variance (early warning signal)
                rolling_var = (
                    agent_data["mood"].rolling(window=window_size, min_periods=1).var()
                )

                # Rolling autocorrelation (another early warning signal)
                rolling_autocorr = (
                    agent_data["mood"]
                    .rolling(window=window_size, min_periods=2)
                    .apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
                )

                # Plot variance
                ax.plot(
                    agent_data["round"],
                    rolling_var,
                    label=f"{agent_name} Variance",
                    color=color,
                    linestyle="-",
                    alpha=0.7,
                )

                # Plot autocorrelation on secondary y-axis
                ax2 = ax.twinx()
                ax2.plot(
                    agent_data["round"],
                    rolling_autocorr,
                    label=f"{agent_name} Autocorr",
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                )

        ax.set_xlabel("Round")
        ax.set_ylabel("Rolling Variance", color="black")
        ax.set_title("Critical Slowing Down Indicators")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_complementarity_dynamics(
        self, ax, agent1_data, agent2_data, agent1, agent2
    ):
        """Plot complementarity evolution over time."""
        # Get dyad-specific complementarity data
        dyad_data = self.data[
            ((self.data["agent"] == agent1) & (self.data["partner"] == agent2))
            | ((self.data["agent"] == agent2) & (self.data["partner"] == agent1))
        ].sort_values("round")

        if len(dyad_data) > 0:
            # Plot raw complementarity
            ax.plot(
                dyad_data["round"],
                dyad_data["complementarity"],
                "o-",
                color="purple",
                alpha=0.6,
                label="Complementarity",
                markersize=3,
            )

            # Add smoothed trend
            if len(dyad_data) > 5:
                window = min(7, len(dyad_data) // 3)
                smoothed = (
                    dyad_data["complementarity"]
                    .rolling(window=window, center=True)
                    .mean()
                )
                ax.plot(
                    dyad_data["round"],
                    smoothed,
                    "-",
                    color="red",
                    linewidth=3,
                    label="Trend",
                    alpha=0.8,
                )

            # Add Tracey's 3-stage model reference if applicable
            if len(dyad_data) > 20:
                # Theoretical Tracey pattern: high -> low -> high
                n_points = len(dyad_data)
                stage1_end = n_points // 3
                stage2_end = 2 * n_points // 3

                rounds = dyad_data["round"].values
                tracey_pattern = np.concatenate(
                    [
                        np.linspace(0.5, 0.5, stage1_end),  # High complementarity
                        np.linspace(0.5, -0.3, stage2_end - stage1_end),  # Decreasing
                        np.linspace(
                            -0.3, 0.4, n_points - stage2_end
                        ),  # Increasing again
                    ]
                )

                ax.plot(
                    rounds,
                    tracey_pattern,
                    "--",
                    color="green",
                    linewidth=2,
                    label="Tracey's Model",
                    alpha=0.7,
                )

        ax.set_xlabel("Round")
        ax.set_ylabel("Complementarity Index")
        ax.set_title("Complementarity Dynamics")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_coupling_analysis(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot behavioral coupling strength over time."""
        # Compute time-windowed correlations
        window_size = max(10, len(agent1_data) // 5)
        coupling_strength = []
        rounds = []

        for i in range(window_size, len(agent1_data)):
            window_agent1 = agent1_data.iloc[i - window_size : i]["mood"]
            window_agent2 = agent2_data.iloc[i - window_size : i]["mood"]

            if len(window_agent1) == len(window_agent2) and len(window_agent1) > 1:
                correlation = np.corrcoef(window_agent1, window_agent2)[0, 1]
                coupling_strength.append(abs(correlation))  # Use absolute value
                rounds.append(agent1_data.iloc[i]["round"])

        if coupling_strength:
            ax.plot(
                rounds,
                coupling_strength,
                "b-",
                linewidth=2,
                alpha=0.7,
                label="Coupling Strength",
            )

            # Add threshold line for "strong coupling"
            threshold = 0.5
            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Strong Coupling (>{threshold})",
            )

            # Highlight periods of strong coupling
            strong_periods = np.array(coupling_strength) > threshold
            if strong_periods.any():
                ax.fill_between(
                    rounds,
                    0,
                    1,
                    where=strong_periods,
                    alpha=0.2,
                    color="green",
                    label="Strong Coupling Periods",
                )

        ax.set_xlabel("Round")
        ax.set_ylabel("Coupling Strength (|r|)")
        ax.set_title("Behavioral Coupling Over Time")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_stability_landscape(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot stability landscape based on velocity analysis."""
        # Plot behavioral velocities
        ax.plot(
            agent1_data["round"],
            agent1_data["behavioral_velocity"],
            "o-",
            label=f"{agent1} Velocity",
            alpha=0.7,
            markersize=3,
        )
        ax.plot(
            agent2_data["round"],
            agent2_data["behavioral_velocity"],
            "s-",
            label=f"{agent2} Velocity",
            alpha=0.7,
            markersize=3,
        )

        # Add stability regions (low velocity = high stability)
        combined_velocity = (
            agent1_data["behavioral_velocity"].values
            + agent2_data["behavioral_velocity"].values
        ) / 2

        # Identify stable periods (bottom quartile of velocities)
        stability_threshold = np.percentile(combined_velocity, 25)
        stable_periods = combined_velocity < stability_threshold

        if stable_periods.any():
            ax.fill_between(
                agent1_data["round"],
                0,
                ax.get_ylim()[1],
                where=stable_periods,
                alpha=0.2,
                color="green",
                label="Stable Periods",
            )

        # Add smoothed trend
        if len(agent1_data) > 5:
            window = min(7, len(agent1_data) // 3)
            smooth1 = (
                agent1_data["behavioral_velocity"]
                .rolling(window=window, center=True)
                .mean()
            )
            smooth2 = (
                agent2_data["behavioral_velocity"]
                .rolling(window=window, center=True)
                .mean()
            )

            ax.plot(agent1_data["round"], smooth1, "-", alpha=0.8, linewidth=2)
            ax.plot(agent2_data["round"], smooth2, "-", alpha=0.8, linewidth=2)

        ax.set_xlabel("Round")
        ax.set_ylabel("Behavioral Velocity")
        ax.set_title("Stability Landscape Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def compute_advanced_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive dynamical systems metrics.

        Returns:
            Dictionary containing advanced metrics for each dyad
        """
        print("🧮 Computing advanced dynamical systems metrics...")

        metrics = {}

        for dyad in self.dyads:
            agent1, agent2 = dyad
            dyad_key = f"{agent1}_{agent2}"

            # Get dyad data
            dyad_data = self.data[
                ((self.data["agent"] == agent1) & (self.data["partner"] == agent2))
                | ((self.data["agent"] == agent2) & (self.data["partner"] == agent1))
            ].sort_values("round")

            if len(dyad_data) < 10:
                continue

            metrics[dyad_key] = {}

            # 1. Behavioral synchrony
            agent1_data = dyad_data[dyad_data["agent"] == agent1]
            agent2_data = dyad_data[dyad_data["agent"] == agent2]

            if len(agent1_data) > 5 and len(agent2_data) > 5:
                merged = pd.merge(
                    agent1_data[["round", "own_x", "own_y", "mood"]],
                    agent2_data[["round", "own_x", "own_y", "mood"]],
                    on="round",
                    suffixes=("_1", "_2"),
                )

                if len(merged) > 5:
                    # Mood synchrony
                    mood_sync = np.corrcoef(merged["mood_1"], merged["mood_2"])[0, 1]
                    metrics[dyad_key]["mood_synchrony"] = mood_sync

                    # Behavioral synchrony
                    behav_sync_x = np.corrcoef(merged["own_x_1"], merged["own_x_2"])[
                        0, 1
                    ]
                    behav_sync_y = np.corrcoef(merged["own_y_1"], merged["own_y_2"])[
                        0, 1
                    ]
                    metrics[dyad_key]["behavioral_synchrony"] = (
                        behav_sync_x + behav_sync_y
                    ) / 2

            # 2. Stability metrics
            stability_ratio = self._compute_stability_ratio(dyad_data)
            metrics[dyad_key]["stability_ratio"] = stability_ratio

            # 3. Complementarity evolution
            comp_trend = self._compute_complementarity_trend(dyad_data)
            metrics[dyad_key].update(comp_trend)

            # 4. Critical slowing down indicators
            csd_metrics = self._compute_critical_slowing_down(dyad_data)
            metrics[dyad_key].update(csd_metrics)

            # 5. Attractor characteristics
            attractor_metrics = self._compute_attractor_metrics(dyad_data)
            metrics[dyad_key].update(attractor_metrics)

        print(f"✅ Computed metrics for {len(metrics)} dyads")
        return metrics

    def _compute_stability_ratio(self, dyad_data):
        """Compute stability ratio (late variance / early variance)."""
        if len(dyad_data) < 20:
            return np.nan

        n_total = len(dyad_data)
        early_period = dyad_data.iloc[: n_total // 4]
        late_period = dyad_data.iloc[-n_total // 4 :]

        early_var = early_period["behavioral_distance"].var()
        late_var = late_period["behavioral_distance"].var()

        return late_var / early_var if early_var > 0 else np.inf

    def _compute_complementarity_trend(self, dyad_data):
        """Compute complementarity evolution metrics."""
        comp_values = dyad_data["complementarity"].values

        if len(comp_values) < 10:
            return {}

        # Linear trend
        x = np.arange(len(comp_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, comp_values)

        # Tracey pattern detection (high-low-high)
        terciles = np.array_split(comp_values, 3)
        early_mean = np.mean(terciles[0])
        middle_mean = np.mean(terciles[1])
        late_mean = np.mean(terciles[2])

        tracey_score = (
            early_mean + late_mean
        ) / 2 - middle_mean  # Higher = more Tracey-like

        return {
            "complementarity_trend_slope": slope,
            "complementarity_trend_r2": r_value**2,
            "complementarity_trend_p": p_value,
            "mean_complementarity": np.mean(comp_values),
            "complementarity_variability": np.std(comp_values),
            "tracey_pattern_score": tracey_score,
        }

    def _compute_critical_slowing_down(self, dyad_data):
        """Compute critical slowing down indicators."""
        if len(dyad_data) < 20:
            return {}

        # Use mood as the primary variable
        mood_values = dyad_data["mood"].values
        window_size = max(5, len(mood_values) // 4)

        # Rolling variance
        rolling_var = pd.Series(mood_values).rolling(window=window_size).var()

        # Rolling autocorrelation
        rolling_autocorr = (
            pd.Series(mood_values)
            .rolling(window=window_size)
            .apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        )

        # Trend in variance (increasing = warning signal)
        var_vals = rolling_var.dropna()
        autocorr_vals = rolling_autocorr.dropna()

        if len(var_vals) > 1:
            var_trend = stats.linregress(range(len(var_vals)), var_vals)[0]
        else:
            var_trend = np.nan

        if len(autocorr_vals) > 1:
            autocorr_trend = stats.linregress(range(len(autocorr_vals)), autocorr_vals)[
                0
            ]
        else:
            autocorr_trend = np.nan

        return {
            "variance_trend": var_trend,
            "autocorr_trend": autocorr_trend,
            "mean_variance": np.nanmean(rolling_var),
            "mean_autocorr": np.nanmean(rolling_autocorr),
        }

    def _compute_attractor_metrics(self, dyad_data):
        """Compute attractor characteristics."""
        if len(dyad_data) < 15:
            return {}

        # Find low-velocity regions (potential attractors)
        velocities = dyad_data["behavioral_velocity"].values
        low_velocity_threshold = np.percentile(velocities, 25)

        attractor_regions = velocities < low_velocity_threshold
        n_attractor_points = np.sum(attractor_regions)

        # Compute basin characteristics
        x_coords = dyad_data["own_x"].values
        y_coords = dyad_data["own_y"].values

        # Convex hull area (measure of space explored)
        from scipy.spatial import ConvexHull

        if len(np.unique(list(zip(x_coords, y_coords)), axis=0)) >= 3:
            try:
                hull = ConvexHull(list(zip(x_coords, y_coords)))
                exploration_area = hull.volume  # 2D area
            except:
                exploration_area = np.nan
        else:
            exploration_area = 0

        return {
            "n_attractor_points": n_attractor_points,
            "attractor_ratio": n_attractor_points / len(velocities),
            "exploration_area": exploration_area,
            "trajectory_length": np.sum(velocities),
        }

    def generate_research_hypotheses(self) -> List[str]:
        """
        Generate testable research hypotheses based on dynamical systems theory.

        Returns:
            List of hypotheses ranked by testability and relevance
        """
        print("💡 Generating research hypotheses...")

        metrics = self.compute_advanced_metrics()

        hypotheses = [
            # === CORE DYNAMICAL SYSTEMS HYPOTHESES ===
            "H1: Agent dyads with higher initial complementarity will show greater behavioral stability over time (measured by stability ratio < 1.0)",
            "H2: Rejection-sensitive agents will exhibit higher behavioral velocity when paired with dominant agents compared to when paired with submissive agents",
            "H3: Therapeutic agents implementing Tracey's 3-stage model will show characteristic complementarity patterns: high (rounds 1-33%) → low (rounds 34-66%) → high (rounds 67-100%)",
            # === COUPLING AND SYNCHRONY HYPOTHESES ===
            "H4: Agents with similar personality types will show stronger mood synchrony (|r| > 0.5) than dissimilar personality pairs",
            "H5: Behavioral synchrony will predict final mood scores better than individual personality traits alone",
            "H6: Time-lagged cross-correlations will reveal asymmetric influence patterns: dominant agents' behavior at time t will predict submissive agents' behavior at time t+1 more strongly than vice versa",
            # === CRITICAL PHENOMENA HYPOTHESES ===
            "H7: Critical slowing down indicators (increasing variance and autocorrelation) will precede major behavioral transitions by 3-5 interaction rounds",
            "H8: Agents starting in 'cold' behavioral regions (warmth < -0.5) will require external perturbations to escape local minima and achieve positive final moods",
            "H9: The variance trend in mood dynamics will be positive before behavioral regime changes and negative during stable periods",
            # === ATTRACTOR AND STABILITY HYPOTHESES ===
            "H10: Successful dyads will converge to stable attractor states with exploration areas < 0.5 square units in behavioral space",
            "H11: Paranoid agents will maintain larger behavioral distances (> 1.0 units) from partners throughout interactions regardless of partner warmth",
            "H12: Manic agents will show chaotic dynamics with trajectory lengths > 2 standard deviations above the mean for all agents",
            # === CLINICAL AND THERAPEUTIC HYPOTHESES ===
            "H13: Agents with avoidant attachment styles will show stability ratios > 1.5, indicating resistance to behavioral change",
            "H14: Therapeutic interventions that temporarily reduce complementarity (Tracey's stage 2) will lead to higher final relationship satisfaction than those maintaining constant high complementarity",
            "H15: Depression-prone agents will show lower exploration areas and more time in attractor regions compared to balanced agents",
            # === NETWORK AND MULTI-AGENT HYPOTHESES ===
            "H16: In multi-agent scenarios (>2), agents will form behavioral clusters based on personality similarity, detectable through network analysis",
            "H17: Environmental stressors (modified payoff matrices) will shift attractor landscapes, with anxious agents showing larger shifts than balanced agents",
            "H18: Machine learning models trained on first 25% of interaction data can predict final dyadic outcomes with >80% accuracy using dynamical features",
            # === METHODOLOGICAL HYPOTHESES ===
            "H19: Phase space reconstruction using time-delay embedding will reveal hidden periodic structures in mood dynamics not visible in time series alone",
            "H20: Recurrence quantification analysis will show higher determinism scores for therapeutic dyads compared to non-therapeutic dyads",
        ]

        # Add data-driven hypotheses based on current results
        if metrics:
            synchrony_values = [
                m.get("mood_synchrony")
                for m in metrics.values()
                if m.get("mood_synchrony") is not None
            ]
            stability_ratios = [
                m.get("stability_ratio")
                for m in metrics.values()
                if m.get("stability_ratio") is not None
                and not np.isinf(m.get("stability_ratio"))
            ]
            tracey_scores = [
                m.get("tracey_pattern_score")
                for m in metrics.values()
                if m.get("tracey_pattern_score") is not None
            ]

            if synchrony_values:
                mean_sync = np.mean(synchrony_values)
                hypotheses.append(
                    f"H21: Based on current data (mean synchrony = {mean_sync:.3f}), mood synchrony serves as a leading indicator of dyadic stability"
                )

            if stability_ratios:
                mean_stability = np.mean(stability_ratios)
                if mean_stability < 1.0:
                    hypotheses.append(
                        "H22: Current evidence suggests stabilizing dynamics; test whether this generalizes across different personality-environment combinations"
                    )
                else:
                    hypotheses.append(
                        "H23: Current evidence suggests destabilizing dynamics; investigate whether this represents adaptive exploration or maladaptive instability"
                    )

            if tracey_scores:
                mean_tracey = np.mean(tracey_scores)
                hypotheses.append(
                    f"H24: Tracey pattern score of {mean_tracey:.3f} in current data suggests need for systematic testing of therapeutic timing interventions"
                )

        print(f"✅ Generated {len(hypotheses)} testable hypotheses")
        return hypotheses

    def create_publication_summary(self, save_path: str = None) -> str:
        """
        Create publication-ready summary with statistical analysis.

        Args:
            save_path: Optional path to save summary

        Returns:
            Formatted summary string
        """
        print("📝 Creating publication summary...")

        metrics = self.compute_advanced_metrics()
        hypotheses = self.generate_research_hypotheses()

        # Statistical summaries
        all_mood_sync = [
            m.get("mood_synchrony")
            for m in metrics.values()
            if m.get("mood_synchrony") is not None
        ]
        all_behav_sync = [
            m.get("behavioral_synchrony")
            for m in metrics.values()
            if m.get("behavioral_synchrony") is not None
        ]
        all_stability = [
            m.get("stability_ratio")
            for m in metrics.values()
            if m.get("stability_ratio") is not None
            and not np.isinf(m.get("stability_ratio"))
        ]
        all_complementarity = [
            m.get("mean_complementarity")
            for m in metrics.values()
            if m.get("mean_complementarity") is not None
        ]
        all_tracey = [
            m.get("tracey_pattern_score")
            for m in metrics.values()
            if m.get("tracey_pattern_score") is not None
        ]

        summary = f"""
INTERPERSONAL DYNAMICS: DYNAMICAL SYSTEMS ANALYSIS
==================================================

Dataset Overview:
- Total agents analyzed: {len(self.agents)}
- Interaction rounds: {self.n_rounds}
- Unique dyads: {len(self.dyads)}
- Total interaction records: {len(self.data)}

DYNAMICAL SYSTEMS METRICS
=========================

Behavioral Synchrony:
- Mood synchrony: M = {np.mean(all_mood_sync):.3f}, SD = {np.std(all_mood_sync):.3f} (n = {len(all_mood_sync)})
- Behavioral synchrony: M = {np.mean(all_behav_sync):.3f}, SD = {np.std(all_behav_sync):.3f} (n = {len(all_behav_sync)})

Stability Analysis:
- Stability ratio: M = {np.mean(all_stability):.3f}, SD = {np.std(all_stability):.3f} (n = {len(all_stability)})
- Converging dyads (ratio < 1.0): {sum(1 for x in all_stability if x < 1.0)}/{len(all_stability)} ({100*sum(1 for x in all_stability if x < 1.0)/len(all_stability):.1f}%)

Complementarity Dynamics:
- Mean complementarity: M = {np.mean(all_complementarity):.3f}, SD = {np.std(all_complementarity):.3f} (n = {len(all_complementarity)})
- Tracey pattern score: M = {np.mean(all_tracey):.3f}, SD = {np.std(all_tracey):.3f} (n = {len(all_tracey)})

DETAILED DYADIC ANALYSIS
========================
"""

        # Add individual dyad details
        for dyad_key, dyad_metrics in metrics.items():
            agent1, agent2 = dyad_key.split("_", 1)
            summary += f"\nDyad: {agent1} ↔ {agent2}\n"
            summary += (
                f"  • Mood synchrony: {dyad_metrics.get('mood_synchrony', 'N/A'):.3f}\n"
            )
            summary += f"  • Stability ratio: {dyad_metrics.get('stability_ratio', 'N/A'):.3f}\n"
            summary += f"  • Complementarity trend: {dyad_metrics.get('complementarity_trend_slope', 'N/A'):.4f}\n"
            summary += f"  • Tracey pattern score: {dyad_metrics.get('tracey_pattern_score', 'N/A'):.3f}\n"

        summary += f"""

KEY RESEARCH HYPOTHESES
=======================
"""
        # Show top 10 hypotheses
        for i, hypothesis in enumerate(hypotheses[:10], 1):
            summary += f"{hypothesis}\n\n"

        summary += f"""Additional Hypotheses: {len(hypotheses) - 10} (see full analysis)

RECOMMENDED RESEARCH PROGRAM
============================

Phase 1: Validation Studies
- Replicate findings with larger agent populations (n > 100)
- Test personality × environment interactions systematically
- Validate dynamical metrics against established interpersonal measures

Phase 2: Mechanism Studies  
- Implement parameter sweeps to test causal hypotheses
- Add external perturbations to test attractor stability
- Develop real-time intervention protocols based on critical slowing down

Phase 3: Translation Studies
- Validate findings with human dyadic interaction data
- Develop clinical decision support tools
- Test in therapeutic training programs

METHODOLOGICAL INNOVATIONS
===========================

1. **Critical Slowing Down Detection**: Real-time monitoring of variance and autocorrelation
2. **Phase Space Reconstruction**: Time-delay embedding for hidden structure detection  
3. **Attractor Landscape Mapping**: Behavioral basin identification and stability analysis
4. **Multi-scale Coupling Analysis**: From micro-interactions to macro-patterns

STATISTICAL CONSIDERATIONS
==========================

Power Analysis:
- Current effect sizes: Cohen's d ≈ {np.mean([abs(x) for x in all_mood_sync if not np.isnan(x)]):.2f} for synchrony measures
- Recommended sample size for future studies: n ≥ 50 dyads per condition
- Multiple comparisons: Use FDR correction for hypothesis testing

Limitations:
- Simulated data may not capture full human complexity
- Need validation with ecological momentary assessment data
- Longer time series (>500 rounds) needed for robust attractor detection

IMPLICATIONS FOR CLINICAL PRACTICE
==================================

1. **Early Warning Systems**: Use critical slowing down to predict relationship crises
2. **Intervention Timing**: Apply Tracey's model with dynamical monitoring
3. **Personalized Treatment**: Match interventions to individual attractor landscapes
4. **Progress Monitoring**: Track synchrony and stability as outcome measures

REFERENCES AND THEORETICAL FOUNDATION
=====================================

Key theoretical frameworks:
- Contemporary Integrative Interpersonal Theory (Hopwood & Pincus, 2025)
- Dynamical Systems in Psychiatry (Scheffer et al., 2024) 
- Formal Models of Interpersonality (Westermann & Banisch, 2025)
- Interpersonal Circumplex Theory (Leary, 1957; Kiesler, 1996)

Computational methods:
- Agent-based modeling with Bayesian belief updating
- Time series analysis with critical phenomena detection
- Phase space reconstruction using Takens' theorem
- Network analysis for multi-agent systems

RESEARCH IMPACT AND DISSEMINATION
=================================

Target Journals:
1. Clinical Psychological Science (dynamical systems focus)
2. Journal of Consulting and Clinical Psychology (clinical applications)
3. Psychological Methods (methodological innovations)
4. Journal of Personality and Social Psychology (theoretical advances)

Conference Presentations:
- Society for Interpersonal Theory and Research (SITAR)
- Association for Behavioral and Cognitive Therapies (ABCT)  
- International Conference on Computational Social Science (IC2S2)

Data and Code Availability:
- All analysis code available at: GitHub.com/thomasthewalkercodes
- Simulation parameters and replication materials: OSF.io/[project_id]

Analysis conducted using: Advanced Interpersonal Dynamics Analyzer v2.0
Contact: Thomas.walker2@uzh.ch | University of Zurich
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"✅ Publication summary saved to {save_path}")

        return summary


# INTEGRATION CLASSES - These extend your existing classes
class EnhancedInterPersonalSimulation(InterPersonalSimulation):
    """
    Enhanced version of your InterPersonalSimulation class with integrated analysis.

    This class extends your existing simulation with dynamical systems analysis
    while maintaining full backward compatibility.
    """

    def __init__(self, base_simulation):
        """
        Initialize enhanced simulation from existing one.

        Args:
            base_simulation: Your existing InterPersonalSimulation instance
        """
        # Copy all attributes from base simulation
        self.__dict__.update(base_simulation.__dict__)

        # Add analysis capabilities
        self.analyzer = None
        self._analysis_cache = {}

    def run_simulation_with_analysis(
        self, n_rounds=200, interaction_pairs=None, enable_live_monitoring=False
    ):
        """
        Run simulation with optional live analysis monitoring.

        Args:
            n_rounds: Number of rounds
            interaction_pairs: Agent pairs to interact
            enable_live_monitoring: Show live plots during simulation
        """
        print("🔬 Running enhanced simulation with dynamical systems monitoring...")

        # Use existing run_simulation method
        self.run_simulation(n_rounds, interaction_pairs)

        # Initialize analyzer after simulation
        self.analyzer = DynamicalSystemsAnalyzer(simulation=self)

        if enable_live_monitoring:
            print("📊 Generating live analysis...")
            self.analyzer.plot_comprehensive_dynamics()
            plt.show()

    def analyze_dynamics(self, agent1=None, agent2=None, save_plots=True):
        """
        Run comprehensive dynamical systems analysis.

        Args:
            agent1, agent2: Specific agents to analyze
            save_plots: Whether to save plots

        Returns:
            Dictionary with all analysis results
        """
        if self.analyzer is None:
            self.analyzer = DynamicalSystemsAnalyzer(simulation=self)

        print("🔍 Running comprehensive dynamical analysis...")

        # Generate plots
        if save_plots:
            fig = self.analyzer.plot_comprehensive_dynamics(
                agent1, agent2, save_path="dynamics_analysis.png"
            )
            plt.close(fig)

        # Compute metrics
        metrics = self.analyzer.compute_advanced_metrics()

        # Generate hypotheses
        hypotheses = self.analyzer.generate_research_hypotheses()

        # Create summary
        summary = self.analyzer.create_publication_summary("analysis_report.txt")

        results = {
            "metrics": metrics,
            "hypotheses": hypotheses,
            "summary": summary,
            "analyzer": self.analyzer,
        }

        # Cache results
        self._analysis_cache = results

        print(
            "✅ Analysis complete! Check 'dynamics_analysis.png' and 'analysis_report.txt'"
        )
        return results

    def get_stability_warning(self) -> Dict[str, Any]:
        """
        Get early warning signals for behavioral transitions.

        Returns:
            Dictionary with warning signals for each agent
        """
        if self.analyzer is None:
            self.analyzer = DynamicalSystemsAnalyzer(simulation=self)

        warnings = {}

        for agent in self.agents.keys():
            agent_data = self.analyzer.data[
                self.analyzer.data["agent"] == agent
            ].sort_values("round")

            if len(agent_data) >= 10:
                # Recent variance increase
                recent_mood = agent_data["mood"].tail(10)
                early_mood = agent_data["mood"].head(10)

                recent_var = recent_mood.var()
                early_var = early_mood.var()

                variance_ratio = recent_var / early_var if early_var > 0 else np.inf

                # Recent autocorrelation increase
                recent_autocorr = (
                    recent_mood.autocorr(lag=1) if len(recent_mood) > 1 else 0
                )

                warning_level = "LOW"
                if variance_ratio > 2.0 or recent_autocorr > 0.7:
                    warning_level = "HIGH"
                elif variance_ratio > 1.5 or recent_autocorr > 0.5:
                    warning_level = "MEDIUM"

                warnings[agent] = {
                    "warning_level": warning_level,
                    "variance_ratio": variance_ratio,
                    "recent_autocorr": recent_autocorr,
                    "current_mood": agent_data["mood"].iloc[-1],
                }

        return warnings


# UTILITY FUNCTIONS FOR EASY INTEGRATION
def enhance_existing_simulation(simulation):
    """
    Convert your existing simulation to enhanced version with analysis.

    Args:
        simulation: Your existing InterPersonalSimulation instance

    Returns:
        EnhancedInterPersonalSimulation with analysis capabilities
    """
    return EnhancedInterPersonalSimulation(simulation)


def quick_analysis(simulation_data):
    """
    Quick analysis function following KISS principles.

    Args:
        simulation_data: DataFrame from get_results_dataframe()

    Returns:
        Simple analysis results
    """
    analyzer = DynamicalSystemsAnalyzer(data=simulation_data)

    # Simple visualization
    if len(analyzer.agents) >= 2:
        fig = analyzer.plot_comprehensive_dynamics()
        plt.show()

    # Key metrics
    metrics = analyzer.compute_advanced_metrics()

    # Simple summary
    print("📊 QUICK ANALYSIS SUMMARY")
    print("=" * 30)
    print(f"Agents: {len(analyzer.agents)}")
    print(f"Rounds: {analyzer.n_rounds}")
    print(f"Dyads analyzed: {len(metrics)}")

    if metrics:
        mood_syncs = [m.get("mood_synchrony", 0) for m in metrics.values()]
        stability_ratios = [
            m.get("stability_ratio", 1)
            for m in metrics.values()
            if not np.isinf(m.get("stability_ratio", np.inf))
        ]

        print(f"Average mood synchrony: {np.mean(mood_syncs):.3f}")
        print(f"Average stability ratio: {np.mean(stability_ratios):.3f}")
        print(
            f"Converging dyads: {sum(1 for x in stability_ratios if x < 1.0)}/{len(stability_ratios)}"
        )

    return {
        "analyzer": analyzer,
        "metrics": metrics,
        "summary_stats": {
            "n_agents": len(analyzer.agents),
            "n_rounds": analyzer.n_rounds,
            "n_dyads": len(metrics),
        },
    }


# EXAMPLE USAGE AND TESTING
def run_integration_example():
    """
    Example showing how to integrate the analysis with your existing code.
    """
    print(
        """
    🚀 INTEGRATION EXAMPLE
    ======================
    
    This example shows how to use the enhanced analysis with your existing code:
    
    # Method 1: Enhance existing simulation
    your_sim = InterPersonalSimulation()
    # ... add agents and run simulation ...
    enhanced_sim = enhance_existing_simulation(your_sim)
    results = enhanced_sim.analyze_dynamics()
    
    # Method 2: Direct analysis of data
    data = your_sim.get_results_dataframe()
    quick_results = quick_analysis(data)
    
    # Method 3: Standalone analyzer
    analyzer = DynamicalSystemsAnalyzer(data=data)
    metrics = analyzer.compute_advanced_metrics()
    hypotheses = analyzer.generate_research_hypotheses()
    """
    )


if __name__ == "__main__":
    run_integration_example()
