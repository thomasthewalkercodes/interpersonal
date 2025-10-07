"""
Circumplex Model of Interpersonal Motives Simulation
Author: [Your Name]
Date: 2024
Description: Agent-based simulation with availability and conflict parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Import configuration
from simulation_config import CircumplexConfig, OCTANT_NAMES, OCTANT_PAIRS


class CircumplexAgent:
    """Agent with circumplex interpersonal motives"""

    def smooth_conflict(
        self,
        last_behavior,
        n_octants=8,
        elevation=0.2,
        amplitude=0.8,
        angular_shift=0.0,
    ):
        conflicts = np.zeros(n_octants)
        for i in range(n_octants):
            # Angular distance from last_behavior
            angle = ((i - last_behavior) / n_octants) * 2 * np.pi + angular_shift
            # Sine wave mapped to [0, 1] then scaled by elevation & amplitude
            conflicts[i] = elevation + amplitude * (0.5 * (np.sin(angle) + 1))
        return conflicts

    def __init__(self, config: CircumplexConfig):
        """
        Initialize agent with configuration

        Parameters:
        -----------
        config : CircumplexConfig
            Configuration object containing availabilities and conflicts
        """
        self.config = config
        self.behavior_history = []
        self.probability_history = []
        self.last_behavior = None
        self.time_steps = []

    def calculate_probabilities(self) -> np.ndarray:
        """Calculate behavior probabilities based on availability and smooth sine-wave conflict"""

        # Start with base availabilities
        probabilities = self.config.availabilities.copy()

        if self.last_behavior is not None:
            # Generate smooth conflict values for all octants
            smooth_conflicts = self.smooth_conflict(
                last_behavior=self.last_behavior,
                n_octants=8,
                elevation=self.config.conflict_elevation,
                amplitude=self.config.conflict_amplitude,
                angular_shift=self.config.conflict_angular_shift,
            )

            # Apply smooth conflict scaling (reduce probabilities according to conflict curve)
            probabilities *= 1 - smooth_conflicts

            # Boost adjacent behaviors slightly (continuity effect)
            adj1 = (self.last_behavior + 1) % 8
            adj2 = (self.last_behavior - 1) % 8
            probabilities[adj1] *= self.config.adjacency_boost
            probabilities[adj2] *= self.config.adjacency_boost

        # Normalize probabilities so they sum to 1
        total = probabilities.sum()
        if total > 0:
            probabilities /= total

        return probabilities

    def select_behavior(self, time_step: int) -> int:
        """Select next behavior based on current probabilities"""
        probabilities = self.calculate_probabilities()

        # Store probability history
        self.probability_history.append(probabilities.copy())
        self.time_steps.append(time_step)

        # Sample behavior
        behavior = np.random.choice(8, p=probabilities)
        self.last_behavior = behavior
        self.behavior_history.append(behavior)

        return behavior


class CircumplexSimulation:
    """Main simulation class"""

    def __init__(self, config: CircumplexConfig, n_steps: int = 200):
        self.config = config
        self.n_steps = n_steps
        self.agent = CircumplexAgent(config)

    def run(self, verbose: bool = False):
        """Run simulation for n_steps"""
        for step in range(self.n_steps):
            behavior = self.agent.select_behavior(step)

            if verbose and step % 20 == 0:
                print(f"Step {step:3d}: {OCTANT_NAMES[behavior]}")

    def create_comprehensive_figure(self):
        """Create comprehensive figure with all visualizations"""

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Availability Radar Plot (top left)
        ax1 = fig.add_subplot(gs[0, 0], projection="polar")
        self.plot_availability_radar(ax1)

        # 2. Conflict Matrix Visualization (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_conflict_matrix(ax2)

        # 3. Behavior Frequency Radar (top right)
        ax3 = fig.add_subplot(gs[0, 2], projection="polar")
        self.plot_behavior_frequency_radar(ax3)

        # 4. Probability Evolution Over Time (middle, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        self.plot_probability_evolution(ax4)

        # 5. Statistics Box (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self.plot_statistics(ax5)

        # 6. Behavior Timeline (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        self.plot_behavior_timeline(ax6)

        # Add main title with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(
            f"Circumplex Model Simulation Results\n{timestamp}",
            fontsize=16,
            fontweight="bold",
        )

        return fig

    def plot_availability_radar(self, ax):
        """Plot radar diagram of availabilities"""
        # Angles for each octant (adjusted so Dominant is at top, Warm at right)
        # Original: 0=Dom, 1=W-D, 2=Warm, 3=W-S, 4=Sub, 5=C-S, 6=Cold, 7=C-D
        # We want: Dom at 90°, Warm at 0°, Sub at 270°, Cold at 180°
        angles_degrees = [90, 45, 0, 315, 270, 225, 180, 135]  # Degrees for each octant
        angles = np.array([np.radians(a) for a in angles_degrees])
        availabilities = self.config.availabilities

        # Close the plot
        angles = np.concatenate([angles, [angles[0]]])
        availabilities = np.concatenate([availabilities, [availabilities[0]]])

        # Plot
        ax.plot(
            angles,
            availabilities,
            "o-",
            linewidth=2,
            markersize=8,
            color="blue",
            label="Availability",
        )
        ax.fill(angles, availabilities, alpha=0.25, color="blue")

        # Set labels at correct positions
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(OCTANT_NAMES, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title("Behavior Availabilities", fontsize=12, fontweight="bold", pad=20)
        ax.grid(True)
        ax.legend(loc="upper right")

        # Add values as text
        for angle, avail, name in zip(
            angles[:-1], self.config.availabilities, OCTANT_NAMES
        ):
            ax.text(angle, avail + 0.05, f"{avail:.2f}", ha="center", fontsize=8)

    def plot_conflict_matrix(self, ax):
        """Visualize conflict relationships"""
        # Create conflict matrix visualization
        conflict_matrix = np.zeros((8, 8))

        # Fill in conflicts for opposite pairs
        for i, (oct1, oct2) in enumerate(OCTANT_PAIRS):
            conflict_matrix[oct1, oct2] = self.config.octant_conflicts[i]
            conflict_matrix[oct2, oct1] = self.config.octant_conflicts[i]

        # Plot heatmap
        im = ax.imshow(conflict_matrix, cmap="Reds", vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([name[:3] for name in OCTANT_NAMES], rotation=45)
        ax.set_yticklabels([name[:3] for name in OCTANT_NAMES])

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add title
        ax.set_title(
            "Conflict Matrix\n(Darker = Higher Conflict)",
            fontsize=12,
            fontweight="bold",
        )

        # Add conflict values as text
        for i in range(8):
            for j in range(8):
                if conflict_matrix[i, j] > 0:
                    text = ax.text(
                        j,
                        i,
                        f"{conflict_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if conflict_matrix[i, j] > 0.5 else "black",
                    )

    def plot_behavior_frequency_radar(self, ax):
        """Plot radar diagram of behavior frequencies"""
        # Angles for each octant (adjusted so Dominant is at top, Warm at right)
        angles_degrees = [90, 45, 0, 315, 270, 225, 180, 135]
        angles = np.array([np.radians(a) for a in angles_degrees])

        # Count behaviors
        counts = np.zeros(8)
        for b in self.agent.behavior_history:
            counts[b] += 1

        # Normalize to proportions
        if len(self.agent.behavior_history) > 0:
            proportions = counts / len(self.agent.behavior_history)
        else:
            proportions = counts

        # Close the plot
        angles = np.concatenate([angles, [angles[0]]])
        proportions = np.concatenate([proportions, [proportions[0]]])

        # Plot
        ax.plot(
            angles,
            proportions,
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
            label="Frequency",
        )
        ax.fill(angles, proportions, alpha=0.25, color="green")

        # Set labels at correct positions
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(OCTANT_NAMES, fontsize=10)
        ax.set_ylim(0, max(proportions) * 1.2 if max(proportions) > 0 else 0.5)
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_title("Behavior Frequencies", fontsize=12, fontweight="bold", pad=20)
        ax.grid(True)
        ax.legend(loc="upper right")

        # Add counts as text
        for angle, prop, count, name in zip(
            angles[:-1], proportions[:-1], counts, OCTANT_NAMES
        ):
            ax.text(angle, prop + 0.02, f"{int(count)}", ha="center", fontsize=8)

    def plot_probability_evolution(self, ax):
        """Plot evolution of probabilities over time"""
        if len(self.agent.probability_history) == 0:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return

        prob_matrix = np.array(self.agent.probability_history).T
        time_steps = self.agent.time_steps

        # Plot each octant's probability over time with adjusted scaling
        colors = plt.cm.Set3(np.linspace(0, 1, 8))

        # Find max probability for better scaling
        max_prob = np.max(prob_matrix) if prob_matrix.size > 0 else 1.0

        for i in range(8):
            ax.plot(
                time_steps,
                prob_matrix[i],
                label=OCTANT_NAMES[i][:3],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )

        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title("Probability Evolution Over Time", fontsize=12, fontweight="bold")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on actual data range
        ax.set_ylim(-0.02, min(max_prob * 1.1, 1.0))

        # Add horizontal lines at key probability levels
        ax.axhline(
            y=0.125, color="gray", linestyle="--", alpha=0.3, linewidth=0.5
        )  # Equal probability line (1/8)
        ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

    def plot_statistics(self, ax):
        """Display key statistics"""
        ax.axis("off")

        # Calculate statistics
        counts = np.zeros(8)
        for b in self.agent.behavior_history:
            counts[b] += 1

        # Calculate dimensional scores
        dominance = (
            counts[0] + counts[1] + counts[7] - counts[3] - counts[4] - counts[5]
        )
        warmth = counts[1] + counts[2] + counts[3] - counts[5] - counts[6] - counts[7]

        # Find most and least frequent behaviors
        if len(self.agent.behavior_history) > 0:
            most_freq_idx = np.argmax(counts)
            least_freq_idx = np.argmin(counts)
            most_freq = OCTANT_NAMES[most_freq_idx]
            least_freq = OCTANT_NAMES[least_freq_idx]
        else:
            most_freq = "N/A"
            least_freq = "N/A"

        # Create statistics text
        stats_text = f"""
        SIMULATION STATISTICS
        ━━━━━━━━━━━━━━━━━━━━
        
        Preset: {self.config.preset_name}
        Total Steps: {len(self.agent.behavior_history)}
        
        Dimensional Scores:
        • Dominance: {dominance:+.1f}
        • Warmth: {warmth:+.1f}
        
        Most Frequent:
        {most_freq} ({int(counts[most_freq_idx] if len(self.agent.behavior_history) > 0 else 0)} times)
        
        Least Frequent:
        {least_freq} ({int(counts[least_freq_idx] if len(self.agent.behavior_history) > 0 else 0)} times)
        
        Configuration:
        • Adjacency Boost: {self.config.adjacency_boost:.2f}
        • Learning Rate: {self.config.learning_rate:.2f}
        """

        ax.text(
            0.1,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def plot_behavior_timeline(self, ax):
        """Plot timeline of behaviors"""
        if len(self.agent.behavior_history) == 0:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
            return

        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, 8))

        # Plot each behavior as a colored bar
        for i, behavior in enumerate(self.agent.behavior_history):
            ax.barh(
                0,
                1,
                left=i,
                height=0.8,
                color=colors[behavior],
                edgecolor="white",
                linewidth=0.5,
            )

        # Add legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(8)]
        ax.legend(
            handles,
            OCTANT_NAMES,
            loc="upper left",
            bbox_to_anchor=(0, 1.15),
            ncol=8,
            frameon=False,
            fontsize=9,
        )

        ax.set_xlim(0, len(self.agent.behavior_history))
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_title(
            "Behavior Sequence Timeline", fontsize=12, fontweight="bold", pad=20
        )
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


def main():
    """Main function to run simulation"""

    print("=" * 60)
    print("CIRCUMPLEX MODEL SIMULATION")
    print("=" * 60)

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Created 'results' directory")

    # Load configuration (this will automatically print the preset being used)
    config = CircumplexConfig()

    # Display current configuration
    print("\nCurrent Configuration:")
    print("-" * 30)
    print("Availabilities:")
    for i, name in enumerate(OCTANT_NAMES):
        bar = "█" * int(config.availabilities[i] * 20)
        print(f"  {name:20s}: {config.availabilities[i]:.2f} {bar}")

    print("\nOctant Conflicts:")
    for i, (oct1, oct2) in enumerate(OCTANT_PAIRS):
        bar = "█" * int(config.octant_conflicts[i] * 20)
        print(
            f"  {OCTANT_NAMES[oct1]:12s} <-> {OCTANT_NAMES[oct2]:12s}: {config.octant_conflicts[i]:.2f} {bar}"
        )

    # Use n_steps from config
    n_steps = config.n_steps

    # Run simulation
    print(f"\nRunning simulation for {n_steps} steps...")
    sim = CircumplexSimulation(config, n_steps=n_steps)
    sim.run(verbose=True)

    # Display results
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    # Show visualizations
    print("\nGenerating visualizations...")
    fig = sim.create_comprehensive_figure()

    # Automatically save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preset_name = config.preset_name.replace(" ", "_").lower()

    # Save figure
    figure_filename = f"results/circumplex_{preset_name}_{timestamp}.png"
    fig.savefig(figure_filename, dpi=150, bbox_inches="tight")
    print(f"\n✅ Figure saved as: {figure_filename}")

    # Save data
    results = {
        "preset": config.preset_name,
        "config": {
            "availabilities": config.availabilities.tolist(),
            "octant_conflicts": config.octant_conflicts.tolist(),
            "adjacency_boost": config.adjacency_boost,
            "learning_rate": config.learning_rate,
        },
        "behavior_history": sim.agent.behavior_history,
        "n_steps": n_steps,
        "timestamp": timestamp,
    }

    data_filename = f"results/circumplex_{preset_name}_{timestamp}.json"
    with open(data_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Data saved as: {data_filename}")

    # Show the plot
    plt.show()

    print("\n" + "=" * 60)
    print("Simulation complete! Files saved in 'results' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
