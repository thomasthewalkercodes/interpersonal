"""
Main simulation engine for the interpersonal circumplex agent-based model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from bayesian_agents_class import Agent
from circumplex_model_core_functions import CircumplexSpace, PayoffMatrix
from agent_personality_configurations import get_personality, list_personalities
import pandas as pd


class InterPersonalSimulation:
    """
    Manages interactions between agents in the interpersonal circumplex space.
    """

    def __init__(self, payoff_matrix: Optional[PayoffMatrix] = None):
        """
        Initialize simulation.

        Args:
            payoff_matrix: Custom payoff matrix (uses default if None)
        """
        self.agents = {}
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.history = []
        self.current_round = 0

    def add_agent(
        self, agent_id: str, personality_name: str, custom_params: Optional[Dict] = None
    ):
        """
        Add an agent to the simulation.

        Args:
            agent_id: Unique identifier for the agent
            personality_name: Name of personality template
            custom_params: Optional parameter overrides
        """
        personality = get_personality(personality_name)
        if custom_params:
            personality.update(custom_params)

        agent = Agent(agent_id, personality)
        self.agents[agent_id] = agent

        # Initialize beliefs about all other agents
        for other_id in self.agents:
            if other_id != agent_id:
                agent.initialize_belief(other_id)
                self.agents[other_id].initialize_belief(agent_id)

    def run_interaction(self, agent1_id: str, agent2_id: str) -> Dict:
        """
        Run a single interaction between two agents.

        Returns:
            Dictionary with interaction details
        """
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]

        # Agents simultaneously choose behaviors based on beliefs
        behavior1 = agent1.choose_behavior(agent2_id)
        behavior2 = agent2.choose_behavior(agent1_id)

        # Calculate payoffs
        payoff1 = self.payoff_matrix.calculate_payoff(behavior1, behavior2)
        payoff2 = self.payoff_matrix.calculate_payoff(behavior2, behavior1)

        # Update agents based on interaction
        agent1.update_after_interaction(agent2_id, behavior2, payoff1)
        agent2.update_after_interaction(agent1_id, behavior1, payoff2)

        # Record interaction
        interaction = {
            "round": self.current_round,
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "behavior1": behavior1,
            "behavior2": behavior2,
            "payoff1": payoff1,
            "payoff2": payoff2,
            "mood1": agent1.mood,
            "mood2": agent2.mood,
            "belief_uncertainty1": agent1.beliefs[agent2_id].get_uncertainty(),
            "belief_uncertainty2": agent2.beliefs[agent1_id].get_uncertainty(),
        }

        return interaction

    def run_simulation(
        self,
        n_rounds: int = 200,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Run the full simulation.

        Args:
            n_rounds: Number of interaction rounds
            interaction_pairs: List of (agent1_id, agent2_id) pairs to interact
                             If None, all agents interact with all others
        """
        if interaction_pairs is None:
            # Create all possible pairs
            agent_ids = list(self.agents.keys())
            interaction_pairs = [
                (agent_ids[i], agent_ids[j])
                for i in range(len(agent_ids))
                for j in range(i + 1, len(agent_ids))
            ]

        print(
            f"Starting simulation with {len(self.agents)} agents for {n_rounds} rounds"
        )
        print(f"Interaction pairs: {interaction_pairs}")

        for round_num in range(n_rounds):
            self.current_round = round_num
            round_interactions = []

            for agent1_id, agent2_id in interaction_pairs:
                interaction = self.run_interaction(agent1_id, agent2_id)
                round_interactions.append(interaction)

            self.history.extend(round_interactions)

            # Progress indicator
            if (round_num + 1) % 50 == 0:
                print(f"Completed round {round_num + 1}/{n_rounds}")

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert simulation history to pandas DataFrame for analysis."""
        if not self.history:
            return pd.DataFrame()

        # Flatten the nested structure
        records = []
        for interaction in self.history:
            # Record for agent 1's perspective
            records.append(
                {
                    "round": interaction["round"],
                    "agent": interaction["agent1_id"],
                    "partner": interaction["agent2_id"],
                    "own_angle": interaction["behavior1"][0],
                    "own_radius": interaction["behavior1"][1],
                    "partner_angle": interaction["behavior2"][0],
                    "partner_radius": interaction["behavior2"][1],
                    "payoff": interaction["payoff1"],
                    "mood": interaction["mood1"],
                    "belief_uncertainty": interaction["belief_uncertainty1"],
                }
            )

            # Record for agent 2's perspective
            records.append(
                {
                    "round": interaction["round"],
                    "agent": interaction["agent2_id"],
                    "partner": interaction["agent1_id"],
                    "own_angle": interaction["behavior2"][0],
                    "own_radius": interaction["behavior2"][1],
                    "partner_angle": interaction["behavior1"][0],
                    "partner_radius": interaction["behavior1"][1],
                    "payoff": interaction["payoff2"],
                    "mood": interaction["mood2"],
                    "belief_uncertainty": interaction["belief_uncertainty2"],
                }
            )

        return pd.DataFrame(records)

    def plot_behavior_trajectories(self, figsize=(15, 10)):
        """Plot how behaviors evolve over time."""
        df = self.get_results_dataframe()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot angle trajectories
        ax = axes[0, 0]
        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]
            ax.plot(
                agent_data["round"], agent_data["own_angle"], label=agent_id, alpha=0.7
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Behavior Angle (degrees)")
        ax.set_title("Behavioral Style Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot radius trajectories
        ax = axes[0, 1]
        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]
            ax.plot(
                agent_data["round"], agent_data["own_radius"], label=agent_id, alpha=0.7
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Behavior Intensity (radius)")
        ax.set_title("Behavioral Intensity Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot mood trajectories
        ax = axes[1, 0]
        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]
            ax.plot(agent_data["round"], agent_data["mood"], label=agent_id, alpha=0.7)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mood")
        ax.set_title("Mood Evolution")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot payoff trajectories
        ax = axes[1, 1]
        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]
            # Calculate rolling average
            window = min(10, len(agent_data) // 10)
            if window > 0:
                rolling_payoff = (
                    agent_data["payoff"].rolling(window=window, min_periods=1).mean()
                )
                ax.plot(agent_data["round"], rolling_payoff, label=agent_id, alpha=0.7)
        ax.set_xlabel("Round")
        ax.set_ylabel("Payoff (rolling avg)")
        ax.set_title("Payoff Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_circumplex_positions(
        self, round_num: Optional[int] = None, figsize=(10, 10)
    ):
        """
        Plot agent positions on the interpersonal circumplex.

        Args:
            round_num: Specific round to plot (None for last round)
        """
        df = self.get_results_dataframe()

        if round_num is None:
            round_num = df["round"].max()

        round_data = df[df["round"] == round_num]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

        # Plot circumplex grid
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        labels = [
            "Warm",
            "Dom-Warm",
            "Dom",
            "Dom-Cold",
            "Cold",
            "Sub-Cold",
            "Sub",
            "Sub-Warm",
        ]
        ax.set_thetagrids(np.degrees(angles), labels)

        # Plot agents
        for agent_id in self.agents:
            agent_data = round_data[round_data["agent"] == agent_id].iloc[0]
            angle_rad = np.radians(agent_data["own_angle"])
            radius = agent_data["own_radius"]

            # Plot position
            ax.scatter(angle_rad, radius, s=200, alpha=0.7, label=agent_id)

            # Add label
            ax.annotate(
                agent_id, (angle_rad, radius), xytext=(5, 5), textcoords="offset points"
            )

        ax.set_ylim(0, 1)
        ax.set_title(f"Agent Positions at Round {round_num}", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        return fig

    def analyze_convergence(self) -> Dict:
        """Analyze whether agents converged to stable behaviors."""
        df = self.get_results_dataframe()

        convergence_metrics = {}

        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]

            # Check last 20% of rounds for stability
            n_check = max(10, len(agent_data) // 5)
            recent_data = agent_data.tail(n_check)

            # Calculate variance in recent behavior
            angle_variance = recent_data["own_angle"].var()
            radius_variance = recent_data["own_radius"].var()

            # Calculate average payoff
            avg_payoff = recent_data["payoff"].mean()
            final_mood = recent_data["mood"].iloc[-1]

            convergence_metrics[agent_id] = {
                "angle_stability": angle_variance < 100,  # Threshold for "stable"
                "angle_variance": angle_variance,
                "radius_variance": radius_variance,
                "avg_recent_payoff": avg_payoff,
                "final_mood": final_mood,
            }

        return convergence_metrics

    def summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for all agents."""
        df = self.get_results_dataframe()

        summary = []
        for agent_id in self.agents:
            agent_data = df[df["agent"] == agent_id]
            personality_type = self.agents[agent_id].personality.get(
                "description", "Unknown"
            )

            summary.append(
                {
                    "agent_id": agent_id,
                    "personality": personality_type,
                    "mean_payoff": agent_data["payoff"].mean(),
                    "std_payoff": agent_data["payoff"].std(),
                    "final_mood": agent_data["mood"].iloc[-1],
                    "mood_change": agent_data["mood"].iloc[-1]
                    - agent_data["mood"].iloc[0],
                    "mean_angle": agent_data["own_angle"].mean(),
                    "mean_radius": agent_data["own_radius"].mean(),
                    "total_angle_movement": agent_data["own_angle"].diff().abs().sum(),
                    "total_radius_movement": agent_data["own_radius"]
                    .diff()
                    .abs()
                    .sum(),
                }
            )

        return pd.DataFrame(summary)


# Example usage function
def run_example_simulation():
    """Run an example simulation with different personality types."""

    # Create simulation
    sim = InterPersonalSimulation()

    # Add agents with different personalities
    sim.add_agent("Agent_Balanced", "balanced")
    sim.add_agent("Agent_Anxious", "anxious")
    sim.add_agent("Agent_Narcissistic", "narcissistic")
    sim.add_agent("Agent_Depressed", "depressed")

    # Run simulation
    sim.run_simulation(n_rounds=200)

    # Generate visualizations
    fig1 = sim.plot_behavior_trajectories()
    plt.show()

    fig2 = sim.plot_circumplex_positions()
    plt.show()

    # Analyze results
    convergence = sim.analyze_convergence()
    print("\nConvergence Analysis:")
    for agent_id, metrics in convergence.items():
        print(
            f"{agent_id}: Stable={metrics['angle_stability']}, "
            f"Final mood={metrics['final_mood']:.2f}"
        )

    # Summary statistics
    summary = sim.summary_statistics()
    print("\nSummary Statistics:")
    print(summary.to_string())

    return sim


if __name__ == "__main__":
    # Run example simulation
    simulation = run_example_simulation()
