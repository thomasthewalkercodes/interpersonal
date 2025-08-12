"""
Advanced Interpersonal Dynamics Analysis Script
==============================================

A comprehensive analysis tool for interpersonal agent behavior incorporating
dynamical systems theory, visualization, and hypothesis generation.

Key Features:
- Trajectory tracking and phase space analysis
- Dynamical systems stability analysis
- Behavioral coupling and co-evolution metrics
- Publication-ready visualizations
- Hypothesis generation for future research

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

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class InterpersonalDynamicsAnalyzer:
    """
    Advanced analyzer for interpersonal dynamics incorporating dynamical systems theory.

    This class provides tools to analyze agent behavior trajectories, detect dynamical
    patterns, measure behavioral coupling, and generate hypotheses for further research.
    """

    def __init__(self, simulation_data: pd.DataFrame):
        """
        Initialize the analyzer with simulation data.

        Args:
            simulation_data: DataFrame from InteractionHistory.get_results_dataframe()
        """
        self.data = simulation_data.copy()
        self.agents = self.data["agent"].unique()
        self.n_rounds = self.data["round"].max()
        self.dyads = self._identify_dyads()

        # Prepare data for analysis
        self._prepare_data()

    def _identify_dyads(self) -> List[Tuple[str, str]]:
        """Identify unique dyads from the data."""
        dyads = set()
        for _, row in self.data.iterrows():
            pair = tuple(sorted([row["agent"], row["partner"]]))
            dyads.add(pair)
        return list(dyads)

    def _prepare_data(self):
        """Prepare data for advanced analysis."""
        # Convert angles to radians for mathematical operations
        self.data["own_angle_rad"] = np.radians(self.data["own_angle"])
        self.data["partner_angle_rad"] = np.radians(self.data["partner_angle"])

        # Compute behavioral coordinates
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

        # Compute behavioral distances and angles
        self.data["behavioral_distance"] = np.sqrt(
            (self.data["own_x"] - self.data["partner_x"]) ** 2
            + (self.data["own_y"] - self.data["partner_y"]) ** 2
        )

        # Complementarity index (based on interpersonal theory)
        self.data["complementarity"] = self._compute_complementarity()

    def _compute_complementarity(self) -> np.ndarray:
        """
        Compute interpersonal complementarity index.

        Based on Kiesler's complementarity principle:
        - Warm behavior invites warm behavior (correspondence)
        - Dominant behavior invites submissive behavior (reciprocity)
        """
        # Extract warmth and dominance dimensions
        own_warmth = self.data["own_radius"] * np.sin(self.data["own_angle_rad"])
        own_dominance = self.data["own_radius"] * np.cos(self.data["own_angle_rad"])
        partner_warmth = self.data["partner_radius"] * np.sin(
            self.data["partner_angle_rad"]
        )
        partner_dominance = self.data["partner_radius"] * np.cos(
            self.data["partner_angle_rad"]
        )

        # Correspondence on warmth dimension
        warmth_correspondence = own_warmth * partner_warmth

        # Reciprocity on dominance dimension
        dominance_reciprocity = -own_dominance * partner_dominance

        # Combined complementarity score
        complementarity = warmth_correspondence + dominance_reciprocity
        return complementarity

    def plot_agent_trajectories(self, figsize=(15, 12), save_path=None):
        """
        Create comprehensive trajectory visualization for two agents.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if len(self.agents) < 2:
            raise ValueError("Need at least 2 agents for trajectory analysis")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Select first two agents for detailed analysis
        agent1, agent2 = self.agents[0], self.agents[1]

        # Get agent data
        agent1_data = self.data[self.data["agent"] == agent1].sort_values("round")
        agent2_data = self.data[self.data["agent"] == agent2].sort_values("round")

        # 1. Behavioral space trajectories
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_behavioral_space(ax1, agent1_data, agent2_data, agent1, agent2)

        # 2. Phase space (agent1 vs agent2 warmth)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_phase_space(ax2, agent1_data, agent2_data, agent1, agent2)

        # 3. Time series of key variables
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_time_series(ax3, agent1_data, agent2_data, agent1, agent2)

        # 4. Complementarity over time
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_complementarity(ax4, agent1_data, agent2_data, agent1, agent2)

        # 5. Behavioral coupling analysis
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_coupling_analysis(ax5, agent1_data, agent2_data, agent1, agent2)

        # 6. Stability analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_stability_analysis(ax6, agent1_data, agent2_data, agent1, agent2)

        plt.suptitle(
            f"Interpersonal Dynamics Analysis: {agent1} ↔ {agent2}",
            fontsize=16,
            fontweight="bold",
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_behavioral_space(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot agent trajectories in behavioral space."""
        # Plot trajectories
        ax.plot(
            agent1_data["own_x"],
            agent1_data["own_y"],
            "o-",
            label=f"{agent1}",
            alpha=0.7,
            markersize=4,
        )
        ax.plot(
            agent2_data["own_x"],
            agent2_data["own_y"],
            "s-",
            label=f"{agent2}",
            alpha=0.7,
            markersize=4,
        )

        # Mark start and end points
        ax.scatter(
            agent1_data["own_x"].iloc[0],
            agent1_data["own_y"].iloc[0],
            s=100,
            marker="o",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            agent1_data["own_x"].iloc[-1],
            agent1_data["own_y"].iloc[-1],
            s=100,
            marker="X",
            color="red",
            label="End",
            zorder=5,
        )

        # Add circumplex grid
        theta = np.linspace(0, 2 * np.pi, 100)
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", alpha=0.3)

        # Add axis labels
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

        ax.set_xlabel("Dominance ←→ Submission")
        ax.set_ylabel("Cold ←→ Warm")
        ax.set_title("Behavioral Space Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    def _plot_phase_space(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot phase space of agent interactions."""
        # Extract warmth dimensions
        agent1_warmth = agent1_data["own_radius"] * np.sin(agent1_data["own_angle_rad"])
        agent2_warmth = agent2_data["own_radius"] * np.sin(agent2_data["own_angle_rad"])

        # Phase space plot
        ax.plot(agent1_warmth, agent2_warmth, "o-", alpha=0.7, markersize=3)

        # Mark trajectory direction with arrows
        for i in range(0, len(agent1_warmth) - 1, max(1, len(agent1_warmth) // 10)):
            ax.annotate(
                "",
                xy=(agent1_warmth.iloc[i + 1], agent2_warmth.iloc[i + 1]),
                xytext=(agent1_warmth.iloc[i], agent2_warmth.iloc[i]),
                arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6),
            )

        # Mark start and end
        ax.scatter(
            agent1_warmth.iloc[0],
            agent2_warmth.iloc[0],
            s=100,
            marker="o",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            agent1_warmth.iloc[-1],
            agent2_warmth.iloc[-1],
            s=100,
            marker="X",
            color="red",
            label="End",
            zorder=5,
        )

        # Add diagonal reference line (perfect correlation)
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

        ax.set_xlabel(f"{agent1} Warmth")
        ax.set_ylabel(f"{agent2} Warmth")
        ax.set_title("Phase Space (Warmth Coupling)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_time_series(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot time series of key behavioral variables."""
        rounds = agent1_data["round"]

        # Plot mood trajectories
        ax.plot(rounds, agent1_data["mood"], "o-", label=f"{agent1} Mood", alpha=0.7)
        ax.plot(rounds, agent2_data["mood"], "s-", label=f"{agent2} Mood", alpha=0.7)

        # Plot payoff trajectories (normalized)
        payoff1_norm = (
            agent1_data["payoff"] - agent1_data["payoff"].mean()
        ) / agent1_data["payoff"].std()
        payoff2_norm = (
            agent2_data["payoff"] - agent2_data["payoff"].mean()
        ) / agent2_data["payoff"].std()

        ax.plot(rounds, payoff1_norm, "--", label=f"{agent1} Payoff (norm)", alpha=0.7)
        ax.plot(rounds, payoff2_norm, "--", label=f"{agent2} Payoff (norm)", alpha=0.7)

        ax.set_xlabel("Round")
        ax.set_ylabel("Standardized Value")
        ax.set_title("Mood and Payoff Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    def _plot_complementarity(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot complementarity index over time."""
        rounds = agent1_data["round"]

        # Get complementarity for this dyad
        dyad_data = self.data[
            ((self.data["agent"] == agent1) & (self.data["partner"] == agent2))
            | ((self.data["agent"] == agent2) & (self.data["partner"] == agent1))
        ].sort_values("round")

        if len(dyad_data) > 0:
            # Plot complementarity
            ax.plot(
                dyad_data["round"],
                dyad_data["complementarity"],
                "o-",
                color="purple",
                alpha=0.7,
                label="Complementarity Index",
            )

            # Add smoothed trend
            if len(dyad_data) > 5:
                window = min(5, len(dyad_data) // 3)
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
                    linewidth=2,
                    label="Smoothed Trend",
                )

        ax.set_xlabel("Round")
        ax.set_ylabel("Complementarity Index")
        ax.set_title("Interpersonal Complementarity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    def _plot_coupling_analysis(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot behavioral coupling analysis."""
        # Compute cross-correlation between behavioral radii
        radius1 = agent1_data["own_radius"].values
        radius2 = agent2_data["own_radius"].values

        if len(radius1) >= 10:  # Need sufficient data for correlation
            # Compute cross-correlation
            correlation = signal.correlate(
                radius1 - radius1.mean(), radius2 - radius2.mean(), mode="full"
            )
            correlation = correlation / (
                np.std(radius1) * np.std(radius2) * len(radius1)
            )

            lags = signal.correlation_lags(len(radius1), len(radius2), mode="full")

            # Plot cross-correlation
            ax.plot(lags, correlation, "b-", alpha=0.7)
            ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="Zero Lag")

            # Mark maximum correlation
            max_idx = np.argmax(np.abs(correlation))
            ax.plot(
                lags[max_idx],
                correlation[max_idx],
                "ro",
                markersize=8,
                label=f"Max: lag={lags[max_idx]}, r={correlation[max_idx]:.3f}",
            )

        ax.set_xlabel("Lag (rounds)")
        ax.set_ylabel("Cross-correlation")
        ax.set_title("Behavioral Coupling (Intensity)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_stability_analysis(self, ax, agent1_data, agent2_data, agent1, agent2):
        """Plot stability analysis based on dynamical systems theory."""

        # Compute velocity in behavioral space
        def compute_velocity(data):
            dx = np.diff(data["own_x"])
            dy = np.diff(data["own_y"])
            velocity = np.sqrt(dx**2 + dy**2)
            return np.concatenate([[velocity[0]], velocity])  # Pad to match length

        vel1 = compute_velocity(agent1_data)
        vel2 = compute_velocity(agent2_data)
        rounds = agent1_data["round"]

        # Plot velocities
        ax.plot(rounds, vel1, "o-", label=f"{agent1} Velocity", alpha=0.7)
        ax.plot(rounds, vel2, "s-", label=f"{agent2} Velocity", alpha=0.7)

        # Add moving average to show trends
        if len(vel1) > 5:
            window = min(5, len(vel1) // 3)
            vel1_smooth = pd.Series(vel1).rolling(window=window, center=True).mean()
            vel2_smooth = pd.Series(vel2).rolling(window=window, center=True).mean()

            ax.plot(rounds, vel1_smooth, "-", alpha=0.8, linewidth=2)
            ax.plot(rounds, vel2_smooth, "-", alpha=0.8, linewidth=2)

        ax.set_xlabel("Round")
        ax.set_ylabel("Behavioral Velocity")
        ax.set_title("Dynamical Stability Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def compute_dynamical_metrics(self) -> Dict[str, Any]:
        """
        Compute advanced dynamical systems metrics.

        Returns:
            Dictionary containing various dynamical metrics
        """
        metrics = {}

        for dyad in self.dyads:
            agent1, agent2 = dyad

            # Get dyad data
            dyad_data = self.data[
                ((self.data["agent"] == agent1) & (self.data["partner"] == agent2))
                | ((self.data["agent"] == agent2) & (self.data["partner"] == agent1))
            ].sort_values("round")

            if len(dyad_data) < 10:  # Need sufficient data
                continue

            dyad_key = f"{agent1}_{agent2}"
            metrics[dyad_key] = {}

            # 1. Behavioral synchrony (correlation in behavioral space)
            agent1_data = dyad_data[dyad_data["agent"] == agent1]
            agent2_data = dyad_data[dyad_data["agent"] == agent2]

            if len(agent1_data) > 5 and len(agent2_data) > 5:
                # Align data by round
                merged = pd.merge(
                    agent1_data[["round", "own_x", "own_y", "mood", "payoff"]],
                    agent2_data[["round", "own_x", "own_y", "mood", "payoff"]],
                    on="round",
                    suffixes=("_1", "_2"),
                )

                if len(merged) > 5:
                    # Behavioral synchrony
                    sync_x = np.corrcoef(merged["own_x_1"], merged["own_x_2"])[0, 1]
                    sync_y = np.corrcoef(merged["own_y_1"], merged["own_y_2"])[0, 1]
                    metrics[dyad_key]["behavioral_sync_x"] = sync_x
                    metrics[dyad_key]["behavioral_sync_y"] = sync_y
                    metrics[dyad_key]["behavioral_sync_overall"] = (sync_x + sync_y) / 2

                    # Mood synchrony
                    mood_sync = np.corrcoef(merged["mood_1"], merged["mood_2"])[0, 1]
                    metrics[dyad_key]["mood_sync"] = mood_sync

                    # Payoff correlation
                    payoff_corr = np.corrcoef(merged["payoff_1"], merged["payoff_2"])[
                        0, 1
                    ]
                    metrics[dyad_key]["payoff_correlation"] = payoff_corr

            # 2. Dynamical stability (variance in late vs early periods)
            n_total = len(dyad_data)
            if n_total >= 20:
                early_period = dyad_data.iloc[: n_total // 4]
                late_period = dyad_data.iloc[-n_total // 4 :]

                early_var = early_period["behavioral_distance"].var()
                late_var = late_period["behavioral_distance"].var()

                metrics[dyad_key]["stability_ratio"] = (
                    late_var / early_var if early_var > 0 else np.inf
                )
                metrics[dyad_key]["early_variance"] = early_var
                metrics[dyad_key]["late_variance"] = late_var

            # 3. Complementarity evolution
            comp_values = dyad_data["complementarity"].values
            if len(comp_values) >= 10:
                # Linear trend in complementarity
                x = np.arange(len(comp_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, comp_values
                )

                metrics[dyad_key]["complementarity_trend"] = slope
                metrics[dyad_key]["complementarity_trend_r2"] = r_value**2
                metrics[dyad_key]["complementarity_trend_p"] = p_value
                metrics[dyad_key]["mean_complementarity"] = np.mean(comp_values)
                metrics[dyad_key]["complementarity_variability"] = np.std(comp_values)

            # 4. Convergence analysis
            behavioral_distances = dyad_data["behavioral_distance"].values
            if len(behavioral_distances) >= 10:
                # Trend in behavioral distance (convergence if negative)
                x = np.arange(len(behavioral_distances))
                slope, _, r_value, p_value, _ = stats.linregress(
                    x, behavioral_distances
                )

                metrics[dyad_key]["convergence_slope"] = slope
                metrics[dyad_key]["convergence_r2"] = r_value**2
                metrics[dyad_key]["convergence_p"] = p_value
                metrics[dyad_key]["final_distance"] = behavioral_distances[-1]
                metrics[dyad_key]["initial_distance"] = behavioral_distances[0]

        return metrics

    def generate_hypotheses(self) -> List[str]:
        """
        Generate research hypotheses based on dynamical systems theory and current data.

        Returns:
            List of testable hypotheses for future research
        """
        metrics = self.compute_dynamical_metrics()

        hypotheses = [
            # Core dynamical systems hypotheses
            "H1: Agent dyads with higher initial complementarity will show greater behavioral stability over time",
            "H2: Rejection-sensitive agents will exhibit higher behavioral velocity (instability) when paired with dominant agents",
            "H3: Therapeutic agents implementing Tracey's 3-stage model will show characteristic patterns of complementarity: high → low → high over interaction phases",
            # Coupling and synchrony hypotheses
            "H4: Agents with similar personality types will show stronger behavioral synchrony than dissimilar pairs",
            "H5: Mood synchrony will predict relationship satisfaction and behavioral convergence over time",
            "H6: Cross-lagged effects will reveal that warmth behavior at time t predicts partner warmth at time t+1 (behavioral contagion)",
            # Stability and convergence hypotheses
            "H7: Dyads will converge to stable attractor states in behavioral space, with the location determined by personality compatibility",
            "H8: Agents starting in 'cold' behavioral regions will require external perturbations (e.g., therapeutic interventions) to escape local minima",
            "H9: The stability ratio (late variance / early variance) will be lower for successful therapeutic dyads",
            # Clinical and therapeutic hypotheses
            "H10: Agents with avoidant attachment styles will show resistance to behavioral change (higher stability ratios) regardless of partner behavior",
            "H11: Manic agents will show chaotic dynamics (high Lyapunov exponents) that disrupt partner behavioral patterns",
            "H12: Paranoid agents will maintain larger behavioral distances from partners throughout interactions",
            # Methodological and measurement hypotheses
            "H13: Critical slowing down indicators (increased autocorrelation, variance) will precede major behavioral transitions",
            "H14: Phase space reconstruction will reveal hidden attractor structures not visible in time series alone",
            "H15: Network analysis of agent interactions will reveal emergent community structures based on behavioral similarity",
            # Extensions and applications
            "H16: Multi-agent scenarios (>2) will show emergent group dynamics that cannot be predicted from dyadic interactions alone",
            "H17: Environmental stressors (modified payoff matrices) will shift attractor landscapes and change stability patterns",
            "H18: Machine learning models trained on early interaction patterns can predict long-term relationship outcomes with >80% accuracy",
        ]

        # Add data-driven hypotheses based on current results
        if metrics:
            # Analyze patterns in current data
            sync_values = [
                m.get("behavioral_sync_overall", 0)
                for m in metrics.values()
                if "behavioral_sync_overall" in m
            ]
            conv_slopes = [
                m.get("convergence_slope", 0)
                for m in metrics.values()
                if "convergence_slope" in m
            ]

            if sync_values:
                mean_sync = np.mean(sync_values)
                hypotheses.append(
                    f"H19: Based on current data (mean sync = {mean_sync:.3f}), behavioral synchrony serves as a predictor of dyadic stability"
                )

            if conv_slopes:
                mean_slope = np.mean(conv_slopes)
                if mean_slope < 0:
                    hypotheses.append(
                        "H20: Current evidence suggests convergent dynamics; test whether this generalizes across different personality combinations"
                    )
                else:
                    hypotheses.append(
                        "H21: Current evidence suggests divergent dynamics; investigate whether this represents instability or exploration"
                    )

        return hypotheses

    def create_publication_summary(self, save_path: str = None) -> str:
        """
        Create a publication-ready summary of the analysis.

        Args:
            save_path: Optional path to save the summary

        Returns:
            String containing the formatted summary
        """
        metrics = self.compute_dynamical_metrics()
        hypotheses = self.generate_hypotheses()

        summary = f"""
INTERPERSONAL DYNAMICS ANALYSIS REPORT
=====================================

Dataset Overview:
- Agents analyzed: {len(self.agents)}
- Total rounds: {self.n_rounds}
- Unique dyads: {len(self.dyads)}
- Total interactions: {len(self.data)}

Key Dynamical Systems Metrics:
{'-' * 30}
"""

        if metrics:
            # Aggregate metrics across dyads
            all_sync = [
                m.get("behavioral_sync_overall")
                for m in metrics.values()
                if m.get("behavioral_sync_overall") is not None
            ]
            all_conv = [
                m.get("convergence_slope")
                for m in metrics.values()
                if m.get("convergence_slope") is not None
            ]
            all_comp = [
                m.get("mean_complementarity")
                for m in metrics.values()
                if m.get("mean_complementarity") is not None
            ]

            if all_sync:
                summary += f"Mean Behavioral Synchrony: {np.mean(all_sync):.3f} ± {np.std(all_sync):.3f}\n"
            if all_conv:
                summary += f"Mean Convergence Slope: {np.mean(all_conv):.4f} ± {np.std(all_conv):.4f}\n"
            if all_comp:
                summary += f"Mean Complementarity: {np.mean(all_comp):.3f} ± {np.std(all_comp):.3f}\n"

        summary += f"""

Research Hypotheses for Future Testing:
{'-' * 40}
"""
        for i, hypothesis in enumerate(hypotheses[:10], 1):  # Show first 10
            summary += f"{hypothesis}\n\n"

        summary += f"""
Additional Hypotheses Available: {len(hypotheses) - 10}

Recommended Next Steps:
{'-' * 23}
1. Implement phase space reconstruction using time-delay embedding
2. Compute Lyapunov exponents to quantify chaotic dynamics  
3. Apply recurrence quantification analysis (RQA) for pattern detection
4. Develop agent-based models with parameter sweeps for hypothesis testing
5. Create longitudinal studies with real human dyads for validation

Key References:
- Hopwood, C. J., & Pincus, A. L. (2025). The interpersonal situation
- Scheffer, M., et al. (2024). A dynamical systems view of psychiatric disorders  
- Westermann, G., & Banisch, S. (2025). A formal model of affiliative interpersonality

Analysis generated by: Advanced Interpersonal Dynamics Analyzer
Contact: Thomas.walker2@uzh.ch | GitHub: thomasthewalkercodes
"""

        if save_path:
            with open(save_path, "w") as f:
                f.write(summary)

        return summary


# Helper functions for additional analyses
def compute_recurrence_plot(trajectory, eps=0.1):
    """Compute recurrence plot for trajectory analysis."""
    n = len(trajectory)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(trajectory[i] - trajectory[j])

    recurrence_matrix = distances < eps
    return recurrence_matrix, distances


def compute_lyapunov_exponent(data, delay=1, embed_dim=3):
    """
    Estimate largest Lyapunov exponent using the Rosenstein method.

    Args:
        data: 1D time series
        delay: Time delay for embedding
        embed_dim: Embedding dimension

    Returns:
        Estimated Lyapunov exponent
    """
    # Time-delay embedding
    n = len(data)
    m = embed_dim
    tau = delay

    # Create embedded vectors
    embedded = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        embedded[:, i] = data[i * tau : n - (m - 1 - i) * tau]

    # Find nearest neighbors
    n_embed = embedded.shape[0]
    divergences = []

    for i in range(n_embed - 10):  # Leave room for evolution
        distances = np.linalg.norm(embedded[i + 1 :] - embedded[i], axis=1)
        if len(distances) > 0:
            nearest_idx = np.argmin(distances) + i + 1

            # Track divergence
            for k in range(1, min(10, n_embed - max(i, nearest_idx))):
                if i + k < n_embed and nearest_idx + k < n_embed:
                    divergence = np.linalg.norm(
                        embedded[i + k] - embedded[nearest_idx + k]
                    )
                    if divergence > 0:
                        divergences.append((k, np.log(divergence)))

    if len(divergences) > 10:
        # Linear fit to log(divergence) vs time
        times, log_divs = zip(*divergences)
        slope, _, _, _, _ = stats.linregress(times, log_divs)
        return slope
    else:
        return np.nan


def phase_space_reconstruction(data, delay=1, embed_dim=3):
    """
    Reconstruct phase space using time-delay embedding (Takens' theorem).

    Args:
        data: 1D time series
        delay: Time delay
        embed_dim: Embedding dimension

    Returns:
        Reconstructed phase space coordinates
    """
    n = len(data)
    m = embed_dim
    tau = delay

    # Create embedded vectors
    embedded = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        embedded[:, i] = data[i * tau : n - (m - 1 - i) * tau]

    return embedded


def compute_mutual_information(x, y, bins=10):
    """Compute mutual information between two variables."""
    # Create joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # Normalize to get probabilities
    hist_2d = hist_2d / np.sum(hist_2d)

    # Marginal distributions
    p_x = np.sum(hist_2d, axis=1)
    p_y = np.sum(hist_2d, axis=0)

    # Compute mutual information
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (p_x[i] * p_y[j]))

    return mi


class AdvancedDynamicalAnalysis:
    """Extended analysis incorporating cutting-edge dynamical systems methods."""

    def __init__(self, analyzer: InterpersonalDynamicsAnalyzer):
        self.analyzer = analyzer
        self.data = analyzer.data

    def critical_slowing_down_analysis(self, agent_id: str, variable: str = "mood"):
        """
        Detect critical slowing down before behavioral transitions.

        Based on Scheffer et al. (2024) approach for psychiatric disorders.
        """
        agent_data = self.data[self.data["agent"] == agent_id].sort_values("round")
        time_series = agent_data[variable].values

        if len(time_series) < 20:
            return None

        # Compute rolling statistics
        window_size = max(5, len(time_series) // 4)

        rolling_var = pd.Series(time_series).rolling(window=window_size).var()
        rolling_autocorr = (
            pd.Series(time_series)
            .rolling(window=window_size)
            .apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0)
        )

        return {
            "variance": rolling_var.values,
            "autocorrelation": rolling_autocorr.values,
            "rounds": agent_data["round"].values,
        }

    def attractor_reconstruction(
        self, agent_id: str, variable: str = "mood", delay: int = 1, embed_dim: int = 3
    ):
        """Reconstruct attractor from behavioral time series."""
        agent_data = self.data[self.data["agent"] == agent_id].sort_values("round")
        time_series = agent_data[variable].values

        if len(time_series) < embed_dim * delay + 10:
            return None

        # Phase space reconstruction
        reconstructed = phase_space_reconstruction(time_series, delay, embed_dim)

        # Compute attractor characteristics
        centroid = np.mean(reconstructed, axis=0)
        distances_from_center = np.linalg.norm(reconstructed - centroid, axis=1)

        return {
            "reconstructed_space": reconstructed,
            "centroid": centroid,
            "spread": np.std(distances_from_center),
            "max_distance": np.max(distances_from_center),
        }

    def coupling_strength_analysis(self, agent1_id: str, agent2_id: str):
        """
        Analyze coupling strength between agents using multiple methods.

        Returns coupling metrics based on:
        1. Cross-correlation
        2. Mutual information
        3. Granger causality (simplified)
        4. Phase synchronization
        """
        # Get aligned data for both agents
        agent1_data = self.data[self.data["agent"] == agent1_id].sort_values("round")
        agent2_data = self.data[self.data["agent"] == agent2_id].sort_values("round")

        # Merge on common rounds
        merged = pd.merge(
            agent1_data[["round", "mood", "own_radius", "own_angle"]],
            agent2_data[["round", "mood", "own_radius", "own_angle"]],
            on="round",
            suffixes=("_1", "_2"),
        )

        if len(merged) < 10:
            return None

        results = {}

        # 1. Cross-correlation analysis
        mood1, mood2 = merged["mood_1"].values, merged["mood_2"].values
        cross_corr = np.corrcoef(mood1, mood2)[0, 1]
        results["cross_correlation_mood"] = cross_corr

        # 2. Mutual information
        mi_mood = compute_mutual_information(mood1, mood2)
        results["mutual_information_mood"] = mi_mood

        # 3. Phase synchronization (for circular variables like angle)
        angle1, angle2 = merged["own_angle_1"].values, merged["own_angle_2"].values
        phase_diff = np.abs(angle1 - angle2)
        phase_diff = np.minimum(phase_diff, 360 - phase_diff)  # Handle wraparound
        phase_sync = 1 - np.std(phase_diff) / 180  # Normalized synchronization index
        results["phase_synchronization"] = phase_sync

        # 4. Time-lagged correlations (simplified Granger causality)
        if len(merged) > 15:
            # Mood1 predicting Mood2
            lag_corr_1to2 = np.corrcoef(mood1[:-1], mood2[1:])[0, 1]
            lag_corr_2to1 = np.corrcoef(mood2[:-1], mood1[1:])[0, 1]

            results["lagged_correlation_1to2"] = lag_corr_1to2
            results["lagged_correlation_2to1"] = lag_corr_2to1
            results["causal_asymmetry"] = lag_corr_1to2 - lag_corr_2to1

        return results

    def detect_regime_changes(self, agent_id: str, variable: str = "mood"):
        """
        Detect regime changes using change point detection.

        Based on behavioral state transitions in dynamical systems.
        """
        agent_data = self.data[self.data["agent"] == agent_id].sort_values("round")
        time_series = agent_data[variable].values

        if len(time_series) < 15:
            return None

        # Simple change point detection using variance
        n = len(time_series)
        change_scores = []

        for i in range(5, n - 5):  # Leave buffer on both sides
            before = time_series[max(0, i - 5) : i]
            after = time_series[i : min(n, i + 5)]

            var_before = np.var(before)
            var_after = np.var(after)
            mean_before = np.mean(before)
            mean_after = np.mean(after)

            # Change score combines variance and mean changes
            variance_change = abs(var_after - var_before) / (var_before + 1e-6)
            mean_change = abs(mean_after - mean_before)

            change_scores.append(variance_change + mean_change)

        # Find peaks in change scores
        if len(change_scores) > 5:
            threshold = np.mean(change_scores) + 2 * np.std(change_scores)
            change_points = [
                i + 5 for i, score in enumerate(change_scores) if score > threshold
            ]

            return {
                "change_points": change_points,
                "change_scores": change_scores,
                "rounds": agent_data["round"].values,
                "threshold": threshold,
            }

        return None

    def network_analysis(self):
        """
        Analyze the interpersonal network structure.

        Creates a network where agents are nodes and edge weights represent
        coupling strength or interaction frequency.
        """
        # Create adjacency matrix
        n_agents = len(self.analyzer.agents)
        agent_to_idx = {agent: i for i, agent in enumerate(self.analyzer.agents)}

        adjacency = np.zeros((n_agents, n_agents))
        coupling_matrix = np.zeros((n_agents, n_agents))

        # Fill adjacency matrix
        for dyad in self.analyzer.dyads:
            agent1, agent2 = dyad
            idx1, idx2 = agent_to_idx[agent1], agent_to_idx[agent2]

            # Count interactions
            interactions = len(
                self.data[
                    ((self.data["agent"] == agent1) & (self.data["partner"] == agent2))
                    | (
                        (self.data["agent"] == agent2)
                        & (self.data["partner"] == agent1)
                    )
                ]
            )

            adjacency[idx1, idx2] = interactions
            adjacency[idx2, idx1] = interactions

            # Coupling strength
            coupling = self.coupling_strength_analysis(agent1, agent2)
            if coupling and "cross_correlation_mood" in coupling:
                strength = abs(coupling["cross_correlation_mood"])
                coupling_matrix[idx1, idx2] = strength
                coupling_matrix[idx2, idx1] = strength

        # Network metrics
        # Degree centrality
        degrees = np.sum(adjacency > 0, axis=1)

        # Clustering coefficient (simplified)
        clustering = []
        for i in range(n_agents):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) > 1:
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = 0
                for j, neighbor1 in enumerate(neighbors):
                    for neighbor2 in neighbors[j + 1 :]:
                        if adjacency[neighbor1, neighbor2] > 0:
                            actual_edges += 1
                clustering.append(
                    actual_edges / possible_edges if possible_edges > 0 else 0
                )
            else:
                clustering.append(0)

        return {
            "adjacency_matrix": adjacency,
            "coupling_matrix": coupling_matrix,
            "degree_centrality": degrees,
            "clustering_coefficients": clustering,
            "agents": list(self.analyzer.agents),
        }


# Example usage and testing functions
def run_comprehensive_analysis(simulation_data: pd.DataFrame, save_plots: bool = True):
    """
    Run a comprehensive analysis of interpersonal dynamics data.

    Args:
        simulation_data: DataFrame from InteractionHistory.get_results_dataframe()
        save_plots: Whether to save plots to files

    Returns:
        Dictionary containing all analysis results
    """
    print("🔬 Starting Comprehensive Interpersonal Dynamics Analysis...")

    # Initialize analyzers
    analyzer = InterpersonalDynamicsAnalyzer(simulation_data)
    advanced = AdvancedDynamicalAnalysis(analyzer)

    results = {}

    # 1. Basic trajectory analysis
    print("📈 Generating trajectory visualizations...")
    if save_plots:
        fig = analyzer.plot_agent_trajectories(figsize=(16, 12))
        plt.savefig("interpersonal_trajectories.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Compute dynamical metrics
    print("🧮 Computing dynamical systems metrics...")
    results["dynamical_metrics"] = analyzer.compute_dynamical_metrics()

    # 3. Advanced analyses
    print("🔍 Running advanced dynamical analyses...")

    # Critical slowing down for each agent
    results["critical_slowing"] = {}
    for agent in analyzer.agents:
        csd = advanced.critical_slowing_down_analysis(agent)
        if csd:
            results["critical_slowing"][agent] = csd

    # Attractor reconstruction
    results["attractors"] = {}
    for agent in analyzer.agents:
        attractor = advanced.attractor_reconstruction(agent)
        if attractor:
            results["attractors"][agent] = attractor

    # Coupling analysis for all dyads
    results["coupling"] = {}
    for dyad in analyzer.dyads:
        agent1, agent2 = dyad
        coupling = advanced.coupling_strength_analysis(agent1, agent2)
        if coupling:
            results["coupling"][f"{agent1}_{agent2}"] = coupling

    # Regime change detection
    results["regime_changes"] = {}
    for agent in analyzer.agents:
        changes = advanced.detect_regime_changes(agent)
        if changes:
            results["regime_changes"][agent] = changes

    # Network analysis
    print("🕸️ Analyzing network structure...")
    results["network"] = advanced.network_analysis()

    # 4. Generate hypotheses
    print("💡 Generating research hypotheses...")
    results["hypotheses"] = analyzer.generate_hypotheses()

    # 5. Create publication summary
    print("📝 Creating publication summary...")
    summary = analyzer.create_publication_summary()
    results["summary"] = summary

    if save_plots:
        with open("analysis_summary.txt", "w") as f:
            f.write(summary)

    print("✅ Analysis complete! Check results dictionary and saved files.")
    return results


# Additional utility functions for KISS principle compliance
def simple_plot_two_agents(data: pd.DataFrame, agent1: str, agent2: str):
    """
    Simple, focused plot of two agents' behavior over time.
    Follows KISS principle - easy to understand and modify.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Get data for each agent
    a1_data = data[data["agent"] == agent1].sort_values("round")
    a2_data = data[data["agent"] == agent2].sort_values("round")

    # Plot 1: Mood over time
    ax1.plot(a1_data["round"], a1_data["mood"], "o-", label=agent1, alpha=0.7)
    ax1.plot(a2_data["round"], a2_data["mood"], "s-", label=agent2, alpha=0.7)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Mood")
    ax1.set_title("Mood Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Behavioral space
    ax2.plot(a1_data["own_x"], a1_data["own_y"], "o-", label=agent1, alpha=0.7)
    ax2.plot(a2_data["own_x"], a2_data["own_y"], "s-", label=agent2, alpha=0.7)
    ax2.set_xlabel("Dominance ↔ Submission")
    ax2.set_ylabel("Cold ↔ Warm")
    ax2.set_title("Behavioral Trajectories")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    return fig


def quick_stats(data: pd.DataFrame) -> Dict[str, float]:
    """
    Quick summary statistics following KISS principle.
    Returns only the most important metrics.
    """
    stats = {}

    # Overall behavioral spread
    stats["behavioral_spread"] = data["behavioral_distance"].std()

    # Average mood
    stats["avg_mood"] = data["mood"].mean()

    # Mood variability
    stats["mood_variability"] = data["mood"].std()

    # Average complementarity
    stats["avg_complementarity"] = data["complementarity"].mean()

    # Number of unique agent pairs
    stats["n_dyads"] = len(data[["agent", "partner"]].drop_duplicates())

    return stats


if __name__ == "__main__":
    # Example of how to use the analysis tools
    print(
        """
    Interpersonal Dynamics Analysis Toolkit
    =======================================
    
    To use this toolkit with your simulation data:
    
    1. Load your data:
       from interpersonal_dynamics_simulation import InteractionHistory
       history = InteractionHistory()
       # ... run your simulation ...
       data = history.get_results_dataframe()
    
    2. Run analysis:
       results = run_comprehensive_analysis(data)
    
    3. Access specific results:
       print(results['summary'])
       print(results['hypotheses'][:5])  # First 5 hypotheses
    
    4. Simple visualization:
       fig = simple_plot_two_agents(data, 'agent1', 'agent2')
       plt.show()
    
    5. Quick stats:
       stats = quick_stats(data)
       print(stats)
    """
    )
