"""
Motive Conflicts in Interpersonal Circumplex: Agent-Based Modeling Study
Based on Westermann et al. (2017) approach, adapted for interpersonal circumplex

This simulation models interpersonal motives as agents that:
1. Have congruence levels (-1 to 1)
2. Can be behaviorally active
3. Affect connected motives
4. Decay over time
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ============================================================================
# CIRCUMPLEX DEFINITIONS
# ============================================================================

OCTANTS = {
    0: {"name": "Dominant", "angle": 90, "abbrev": "DOM"},
    1: {"name": "Warm-Dominant", "angle": 45, "abbrev": "W-D"},
    2: {"name": "Warm", "angle": 0, "abbrev": "WRM"},
    3: {"name": "Warm-Submissive", "angle": 315, "abbrev": "W-S"},
    4: {"name": "Submissive", "angle": 270, "abbrev": "SUB"},
    5: {"name": "Cold-Submissive", "angle": 225, "abbrev": "C-S"},
    6: {"name": "Cold", "angle": 180, "abbrev": "CLD"},
    7: {"name": "Cold-Dominant", "angle": 135, "abbrev": "C-D"},
}

# ============================================================================
# PERSONALITY PROTOTYPES (Based on means, variances, covariances)
# ============================================================================

PERSONALITY_PROTOTYPES = {
    "balanced": {
        "name": "Balanced",
        "means": np.zeros(8),  # Neutral on all octants
        "variances": np.ones(8) * 0.25,
        "covariance_pattern": "circumplex",  # Standard circumplex pattern
    },
    "borderline": {
        "name": "Borderline",
        "means": np.array([0, 0, 0.5, 0, 0, 0, -0.5, 0]),  # High warm, low cold
        "variances": np.array(
            [0.3, 0.3, 0.8, 0.3, 0.3, 0.3, 0.8, 0.3]
        ),  # High variance on warm/cold
        "covariance_pattern": "unstable",  # High negative correlation between warm/cold
    },
    "narcissistic": {
        "name": "Narcissistic",
        "means": np.array([0.6, 0.4, 0.2, -0.2, -0.4, -0.2, 0.2, 0.4]),  # High dominant
        "variances": np.array([0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.3, 0.4]),
        "covariance_pattern": "rigid",  # Low flexibility
    },
    "avoidant": {
        "name": "Avoidant",
        "means": np.array(
            [-0.3, -0.3, -0.5, -0.3, 0.2, 0.5, 0.6, 0.4]
        ),  # High cold-submissive
        "variances": np.ones(8) * 0.3,
        "covariance_pattern": "circumplex",
    },
    "random": {
        "name": "Random",
        "means": np.random.normal(0, 0.3, 8),
        "variances": np.random.uniform(0.1, 0.5, 8),
        "covariance_pattern": "random",
    },
}


class MotiveAgent:
    """
    Individual motive/octant that acts as an agent
    Based on Westermann et al.'s model
    """

    def __init__(
        self,
        motive_id: int,
        initial_congruence: float = 0.0,
        decay_rate: float = 0.01,
        availability: float = 0.5,
    ):
        self.id = motive_id
        self.name = OCTANTS[motive_id]["name"]
        self.congruence = initial_congruence  # Range: -1 (incongruent) to 1 (congruent)
        self.decay_rate = decay_rate
        self.availability = availability  # How easily this motive can be satisfied
        self.is_active = False
        self.activation_history = []

    def update_congruence(self, delta: float):
        """Update congruence level, bounded between -1 and 1"""
        self.congruence = np.clip(self.congruence + delta, -1, 1)

    def decay(self):
        """Natural decay of congruence over time"""
        if self.congruence > 0:
            self.congruence -= self.decay_rate
        else:
            self.congruence -= self.decay_rate * 1.5  # Faster decay when incongruent
        self.congruence = max(self.congruence, -1)

    def is_incongruent(self) -> bool:
        """Check if motive is incongruent (needs attention)"""
        return self.congruence < 0

    def get_incongruence_level(self) -> float:
        """Return level of incongruence (0 if congruent)"""
        return max(0, -self.congruence)


class PersonalitySystem:
    """
    System of interconnected motives for a single person
    Implements the Westermann et al. model for interpersonal circumplex
    """

    def __init__(self, personality_type: str, person_id: int = 0):
        self.person_id = person_id
        self.personality_type = personality_type
        self.prototype = PERSONALITY_PROTOTYPES[personality_type]

        # Initialize motives
        self.motives = {}
        for i in range(8):
            initial_congruence = self.prototype["means"][i]
            availability = (
                0.5 + self.prototype["means"][i] * 0.3
            )  # Higher mean = more available
            decay_rate = 0.01 * (
                1 + self.prototype["variances"][i]
            )  # Higher variance = faster decay

            self.motives[i] = MotiveAgent(
                i, initial_congruence, decay_rate, availability
            )

        # Create connection matrix (how motives affect each other)
        self.connection_matrix = self._create_connection_matrix()

        # History tracking
        self.behavior_history = []
        self.congruence_history = {i: [] for i in range(8)}
        self.incongruence_history = []
        self.active_motive = None

    def _create_connection_matrix(self) -> np.ndarray:
        """
        Create connection matrix based on circumplex structure
        Positive values = motives support each other
        Negative values = motives compete
        """
        matrix = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                if i == j:
                    continue

                # Calculate angular distance
                angle_diff = abs(OCTANTS[i]["angle"] - OCTANTS[j]["angle"])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                # Convert to correlation based on circumplex structure
                if angle_diff <= 45:  # Adjacent
                    correlation = 0.7
                elif angle_diff <= 90:  # One removed
                    correlation = 0.3
                elif angle_diff <= 135:  # Two removed
                    correlation = -0.3
                else:  # Opposite
                    correlation = -0.7

                # Modify based on personality pattern
                if self.prototype["covariance_pattern"] == "unstable":
                    if i in [2, 6] and j in [2, 6]:  # Warm-Cold conflict
                        correlation = -0.9
                elif self.prototype["covariance_pattern"] == "rigid":
                    correlation *= 0.5  # Less flexibility
                elif self.prototype["covariance_pattern"] == "random":
                    correlation += np.random.normal(0, 0.2)

                matrix[i, j] = correlation

        return matrix

    def step(self):
        """
        Single time step of the system
        Following Westermann et al.'s rules:
        1. All motives decay
        2. If no active motive, select most incongruent
        3. Active motive affects connected motives
        4. Active motive stays active until satisfied
        """

        # 1. Decay all motives
        for motive in self.motives.values():
            motive.decay()

        # 2. Select active motive if none
        if self.active_motive is None:
            incongruent_motives = [
                (m.get_incongruence_level(), m.id)
                for m in self.motives.values()
                if m.is_incongruent()
            ]

            if incongruent_motives:
                # Select most incongruent
                incongruent_motives.sort(reverse=True)
                most_incongruent_id = incongruent_motives[0][1]
                self.active_motive = self.motives[most_incongruent_id]
                self.active_motive.is_active = True
                self.behavior_history.append(most_incongruent_id)

        # 3. Active motive affects system
        if self.active_motive is not None:
            # Increase congruence of active motive
            satisfaction_rate = 0.1 * self.active_motive.availability
            self.active_motive.update_congruence(satisfaction_rate)

            # Affect connected motives
            for other_id, other_motive in self.motives.items():
                if other_id != self.active_motive.id:
                    connection_strength = self.connection_matrix[
                        self.active_motive.id, other_id
                    ]
                    effect = connection_strength * 0.05
                    other_motive.update_congruence(effect)

            # 4. Check if satisfied
            if self.active_motive.congruence >= 0.9:
                self.active_motive.is_active = False
                self.active_motive = None

        # Record history
        for i, motive in self.motives.items():
            self.congruence_history[i].append(motive.congruence)

        total_incongruence = sum(
            m.get_incongruence_level() for m in self.motives.values()
        )
        self.incongruence_history.append(total_incongruence)

    def simulate(self, n_steps: int = 500):
        """Run simulation for n steps"""
        for _ in range(n_steps):
            self.step()

    def get_behavior_frequency(self) -> Dict[int, int]:
        """Get frequency of each behavior"""
        freq = {i: 0 for i in range(8)}
        for behavior in self.behavior_history:
            freq[behavior] += 1
        return freq

    def get_total_behavior_count(self) -> int:
        """Total number of behaviors (regulation effort)"""
        return len(self.behavior_history)

    def get_mean_incongruence(self) -> float:
        """Average incongruence over simulation"""
        return np.mean(self.incongruence_history) if self.incongruence_history else 0


class PopulationSimulation:
    """
    Simulate a population of people with different personality patterns
    """

    def __init__(self, n_persons_per_type: int = 10):
        self.n_persons_per_type = n_persons_per_type
        self.population = []
        self.results = None

    def create_population(self):
        """Create population with different personality types"""
        person_id = 0
        for p_type in PERSONALITY_PROTOTYPES.keys():
            for _ in range(self.n_persons_per_type):
                # Add some random variation to base prototype
                person = PersonalitySystem(p_type, person_id)

                # Add individual differences
                for motive in person.motives.values():
                    motive.congruence += np.random.normal(0, 0.1)
                    motive.availability += np.random.normal(0, 0.05)
                    motive.congruence = np.clip(motive.congruence, -1, 1)
                    motive.availability = np.clip(motive.availability, 0.1, 0.9)

                self.population.append(person)
                person_id += 1

    def run_simulation(self, n_steps: int = 500):
        """Run simulation for entire population"""
        print(f"Simulating {len(self.population)} individuals for {n_steps} steps...")

        for i, person in enumerate(self.population):
            if i % 10 == 0:
                print(f"  Processing person {i+1}/{len(self.population)}...")
            person.simulate(n_steps)

        self._compile_results()

    def _compile_results(self):
        """Compile results into DataFrame"""
        results_list = []

        for person in self.population:
            behavior_freq = person.get_behavior_frequency()

            results_list.append(
                {
                    "person_id": person.person_id,
                    "personality_type": person.personality_type,
                    "total_behaviors": person.get_total_behavior_count(),
                    "mean_incongruence": person.get_mean_incongruence(),
                    **{f"behavior_{i}": behavior_freq[i] for i in range(8)},
                }
            )

        self.results = pd.DataFrame(results_list)

    def analyze_results(self):
        """Analyze and visualize results"""
        if self.results is None:
            print("No results to analyze. Run simulation first.")
            return

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Behavior frequency by personality type
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_behavior_by_type(ax1)

        # 2. Total behaviors (regulation effort) by type
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_regulation_effort(ax2)

        # 3. Mean incongruence by type
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_incongruence_by_type(ax3)

        # 4. Behavior patterns heatmap
        ax4 = fig.add_subplot(gs[1, 1:])
        self._plot_behavior_heatmap(ax4)

        # 5. Individual trajectories example
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_example_trajectories(ax5)

        plt.suptitle(
            "Motive Conflicts: Population Simulation Results",
            fontsize=16,
            fontweight="bold",
        )

        # Save figure
        if not os.path.exists("results"):
            os.makedirs("results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/population_simulation_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"\nResults saved to: {filename}")

        plt.show()

        # Statistical analysis
        self._statistical_analysis()

    def _plot_behavior_by_type(self, ax):
        """Plot behavior frequency by personality type"""
        behavior_cols = [f"behavior_{i}" for i in range(8)]

        grouped = self.results.groupby("personality_type")[behavior_cols].mean()

        x = np.arange(len(grouped.index))
        width = 0.1

        for i, col in enumerate(behavior_cols):
            offset = (i - 3.5) * width
            ax.bar(x + offset, grouped[col], width, label=OCTANTS[i]["abbrev"])

        ax.set_xlabel("Personality Type")
        ax.set_ylabel("Mean Behavior Frequency")
        ax.set_title("Behavior Patterns by Personality Type")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.legend(ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    def _plot_regulation_effort(self, ax):
        """Plot total behaviors (regulation effort)"""
        grouped = self.results.groupby("personality_type")["total_behaviors"].agg(
            ["mean", "std"]
        )

        ax.bar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            capsize=5,
            color="steelblue",
            alpha=0.7,
        )
        ax.set_xlabel("Personality Type")
        ax.set_ylabel("Total Behaviors (Regulation Effort)")
        ax.set_title("Behavioral Regulation Effort by Type")
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.grid(True, alpha=0.3)

        # Add significance test
        borderline_behaviors = self.results[
            self.results["personality_type"] == "borderline"
        ]["total_behaviors"]
        balanced_behaviors = self.results[
            self.results["personality_type"] == "balanced"
        ]["total_behaviors"]
        if len(borderline_behaviors) > 0 and len(balanced_behaviors) > 0:
            t_stat, p_value = stats.ttest_ind(borderline_behaviors, balanced_behaviors)
            ax.text(
                0.5,
                0.95,
                f"Borderline vs Balanced: p={p_value:.3f}",
                transform=ax.transAxes,
                ha="center",
            )

    def _plot_incongruence_by_type(self, ax):
        """Plot mean incongruence by personality type"""
        grouped = self.results.groupby("personality_type")["mean_incongruence"].agg(
            ["mean", "std"]
        )

        ax.bar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            capsize=5,
            color="coral",
            alpha=0.7,
        )
        ax.set_xlabel("Personality Type")
        ax.set_ylabel("Mean Incongruence")
        ax.set_title("Average Incongruence by Type")
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_behavior_heatmap(self, ax):
        """Plot heatmap of behavior patterns"""
        behavior_cols = [f"behavior_{i}" for i in range(8)]

        # Create matrix for heatmap
        grouped = self.results.groupby("personality_type")[behavior_cols].mean()

        # Normalize by row to show relative patterns
        grouped_norm = grouped.div(grouped.sum(axis=1), axis=0)

        im = ax.imshow(grouped_norm.T, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(np.arange(len(grouped.index)))
        ax.set_yticks(np.arange(8))
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.set_yticklabels([OCTANTS[i]["name"] for i in range(8)])
        ax.set_title("Relative Behavior Patterns (Normalized)")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations
        for i in range(len(grouped.index)):
            for j in range(8):
                text = ax.text(
                    i,
                    j,
                    f"{grouped_norm.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    def _plot_example_trajectories(self, ax):
        """Plot example congruence trajectories"""
        # Select one person from each type
        example_persons = []
        for p_type in PERSONALITY_PROTOTYPES.keys():
            persons_of_type = [
                p for p in self.population if p.personality_type == p_type
            ]
            if persons_of_type:
                example_persons.append(persons_of_type[0])

        # Plot trajectories
        for person in example_persons[:3]:  # Show only 3 for clarity
            incongruence = person.incongruence_history
            ax.plot(
                incongruence,
                label=f"{person.personality_type} (ID: {person.person_id})",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Total Incongruence")
        ax.set_title("Example Incongruence Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _statistical_analysis(self):
        """Perform statistical analysis"""
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)

        # ANOVA for total behaviors
        groups = [
            group["total_behaviors"].values
            for name, group in self.results.groupby("personality_type")
        ]
        f_stat, p_value = stats.f_oneway(*groups)

        print(f"\nANOVA for Total Behaviors across personality types:")
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

        # Post-hoc comparisons
        print("\nMean Total Behaviors by Type:")
        summary = self.results.groupby("personality_type")["total_behaviors"].agg(
            ["mean", "std"]
        )
        print(summary)

        # Correlation between incongruence and behavior
        correlation = self.results["mean_incongruence"].corr(
            self.results["total_behaviors"]
        )
        print(
            f"\nCorrelation between incongruence and total behaviors: {correlation:.3f}"
        )

        # Save detailed results
        results_file = (
            f'results/population_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        self.results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")


def main():
    """Main function to run the simulation"""
    print("=" * 60)
    print("MOTIVE CONFLICTS: EMPIRICALLY INFORMED AGENT-BASED MODELING")
    print("Based on Westermann et al. (2017)")
    print("=" * 60)

    # Create and run population simulation
    pop_sim = PopulationSimulation(n_persons_per_type=20)

    print("\nCreating population...")
    pop_sim.create_population()

    print("\nRunning simulation...")
    pop_sim.run_simulation(n_steps=500)

    print("\nAnalyzing results...")
    pop_sim.analyze_results()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
