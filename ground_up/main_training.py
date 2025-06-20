"""
Main training script for your SAC interpersonal behavior simulation.
Uses your Gaussian payoff function and agent configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
from comprehensive_logging import LoggingTrainerWrapper

# Import your modules
from agent_configuration import (
    BaseAgentConfig,
    CooperativeAgentConfig,
    CompetitiveAgentConfig,
    AdaptiveAgentConfig,
    CautiousAgentConfig,
)
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment
from gaussian_payoff_graph import calculate_warmth_payoff
from interfaces import PayoffCalculator


class GaussianPayoffCalculator(PayoffCalculator):
    """Payoff calculator using your Gaussian warmth function."""

    def __init__(self, alpha: float = 4, beta: float = 10):
        """
        Initialize with your Gaussian parameters.

        Args:
            alpha: Mismatch penalty factor (higher = more sensitive to mismatches)
            beta: Risk factor weight (higher = more penalty for rejection)
        """
        self.alpha = alpha
        self.beta = beta

    def calculate_payoff(
        self, agent1_action: float, agent2_action: float, agent1_id: str, agent2_id: str
    ) -> tuple[float, float]:
        """Calculate payoffs using your Gaussian warmth function."""
        # Convert actions from [-1, 1] to [0, 1] for warmth
        w1 = (agent1_action + 1) / 2
        w2 = (agent2_action + 1) / 2

        # Calculate payoffs using your function
        payoff1 = calculate_warmth_payoff(w1, w2, self.alpha, self.beta)
        payoff2 = calculate_warmth_payoff(w2, w1, self.alpha, self.beta)

        return payoff1, payoff2


class ExperimentRunner:
    """Runs SAC experiments with your configurations."""

    def __init__(self):
        self.results = {}

    def create_agent_pair(self, config1, config2, experiment_name, alpha=4, beta=10):
        """Create a pair of agents with your Gaussian payoff function."""
        # Create agent states
        state1 = config1.create_initial_state()
        state2 = config2.create_initial_state()

        # Get state dimension
        state_dim = state1.get_state_dimension()

        # Create SAC agents
        agent1 = SACAgent(state_dim, config1.get_sac_params())
        agent2 = SACAgent(state_dim, config2.get_sac_params())

        # Create your Gaussian payoff calculator
        payoff_calculator = GaussianPayoffCalculator(alpha=alpha, beta=beta)

        # Create environment
        environment = InterpersonalEnvironment(
            payoff_calculator=payoff_calculator,
            agent1_state=state1,
            agent2_state=state2,
            agent1_id=f"{experiment_name}_agent1",
            agent2_id=f"{experiment_name}_agent2",
            max_steps_per_episode=50,
        )

        return agent1, agent2, environment

    def run_experiment(
        self, config1, config2, experiment_name, episodes=300, alpha=4, beta=10
    ):
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {experiment_name}")
        print(f"Episodes: {episodes}, Alpha: {alpha}, Beta: {beta}")
        print(f"{'='*60}")

        # Create agents and environment
        agent1, agent2, environment = self.create_agent_pair(
            config1, config2, experiment_name, alpha, beta
        )

        # Create trainer
        trainer = SACTrainer(
            agent1=agent1,
            agent2=agent2,
            environment=environment,
            payoff_calculator=environment.payoff_calculator,
            episodes_per_training=episodes,
            steps_per_episode=50,
            evaluation_frequency=max(100, episodes // 10),
            save_frequency=max(500, episodes // 2),
        )

        # Train agents
        save_dir = f"./models/{experiment_name}"
        logging_wrapper = LoggingTrainerWrapper(trainer, experiment_name)
        results = logging_wrapper.train_with_logging(save_dir)

        # Store results
        self.results[experiment_name] = results

        print(f"\nExperiment {experiment_name} completed!")
        return results

    def analyze_results(self, experiment_name):
        """Analyze and plot results."""
        if experiment_name not in self.results:
            print(f"No results found for {experiment_name}")
            return

        results = self.results[experiment_name]

        # Create plots
        plt.figure(figsize=(15, 5))

        # Plot 1: Episode rewards
        plt.subplot(1, 3, 1)
        episodes = range(len(results["episode_rewards"]["agent1"]))
        plt.plot(
            episodes,
            results["episode_rewards"]["agent1"],
            label="Agent 1",
            alpha=0.7,
            linewidth=1,
        )
        plt.plot(
            episodes,
            results["episode_rewards"]["agent2"],
            label="Agent 2",
            alpha=0.7,
            linewidth=1,
        )

        # Add moving averages
        window = min(50, len(episodes) // 10)
        if len(episodes) > window:
            avg1 = np.convolve(
                results["episode_rewards"]["agent1"],
                np.ones(window) / window,
                mode="valid",
            )
            avg2 = np.convolve(
                results["episode_rewards"]["agent2"],
                np.ones(window) / window,
                mode="valid",
            )
            avg_episodes = range(window - 1, len(episodes))
            plt.plot(avg_episodes, avg1, "--", linewidth=2, label="Agent 1 (avg)")
            plt.plot(avg_episodes, avg2, "--", linewidth=2, label="Agent 2 (avg)")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{experiment_name}: Episode Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Learning curves
        plt.subplot(1, 3, 2)
        if results["training_metrics"]["agent1"]:
            steps = range(len(results["training_metrics"]["agent1"]))
            alpha_values1 = [m["alpha"] for m in results["training_metrics"]["agent1"]]
            # Convert tensors to float if needed
            alpha_values1 = [
                float(a.detach()) if hasattr(a, "detach") else float(a)
                for a in alpha_values1
            ]
            plt.plot(steps, alpha_values1, label="Agent 1 Temperature", alpha=0.7)

        if results["training_metrics"]["agent2"]:
            steps = range(len(results["training_metrics"]["agent2"]))
            alpha_values2 = [m["alpha"] for m in results["training_metrics"]["agent2"]]
            # Convert tensors to float if needed
            alpha_values2 = [
                float(a.detach()) if hasattr(a, "detach") else float(a)
                for a in alpha_values2
            ]
            plt.plot(steps, alpha_values2, label="Agent 2 Temperature", alpha=0.7)

        plt.xlabel("Training Step")
        plt.ylabel("Temperature (Alpha)")
        plt.title(f"{experiment_name}: Learning Temperature")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Final performance
        plt.subplot(1, 3, 3)
        final_episodes = min(100, len(results["episode_rewards"]["agent1"]) // 4)
        final_rewards1 = results["episode_rewards"]["agent1"][-final_episodes:]
        final_rewards2 = results["episode_rewards"]["agent2"][-final_episodes:]

        labels = ["Agent 1", "Agent 2"]
        means = [np.mean(final_rewards1), np.mean(final_rewards2)]
        stds = [np.std(final_rewards1), np.std(final_rewards2)]

        plt.bar(
            labels,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=["skyblue", "lightcoral"],
        )
        plt.ylabel(f"Average Reward (Last {final_episodes} Episodes)")
        plt.title(f"{experiment_name}: Final Performance")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        os.makedirs("./results", exist_ok=True)
        plt.savefig(
            f"./results/{experiment_name}_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # Print summary
        print(f"\n{'='*50}")
        print(f"RESULTS SUMMARY: {experiment_name}")
        print(f"{'='*50}")
        print(f"Total Episodes: {len(results['episode_rewards']['agent1'])}")
        print(
            f"Agent 1 - Final Avg Reward: {np.mean(final_rewards1):.3f} ± {np.std(final_rewards1):.3f}"
        )
        print(
            f"Agent 2 - Final Avg Reward: {np.mean(final_rewards2):.3f} ± {np.std(final_rewards2):.3f}"
        )

        if results["training_metrics"]["agent1"]:
            final_alpha1 = results["training_metrics"]["agent1"][-1]["alpha"]
            print(f"Agent 1 - Final Temperature: {final_alpha1:.4f}")
        if results["training_metrics"]["agent2"]:
            final_alpha2 = results["training_metrics"]["agent2"][-1]["alpha"]
            print(f"Agent 2 - Final Temperature: {final_alpha2:.4f}")


def main():
    """Main function to run experiments."""
    print("SAC Interpersonal Behavior Simulation with Gaussian Payoffs")
    print("=" * 60)

    # Create experiment runner
    runner = ExperimentRunner()

    # Define experiments
    experiments = [
        {
            "name": "competitive_vs_competitive",
            "config1": CompetitiveAgentConfig(memory_length=50),  # Same memory length
            "config2": CompetitiveAgentConfig(memory_length=50),  # Same memory length
            "episodes": 300,
            "alpha": 1,  # Your Gaussian parameters
            "beta": 3,
        },
        {
            "name": "more rejection sens competitive_vs_competitive",
            "config1": CompetitiveAgentConfig(memory_length=50),  # Same memory length
            "config2": CompetitiveAgentConfig(memory_length=50),  # Same memory length
            "episodes": 300,
            "alpha": 8,  # Your Gaussian parameters
            "beta": 10,
        },
        {
            "name": "symmetric_cooperative",
            "config1": CooperativeAgentConfig(memory_length=50),
            "config2": CooperativeAgentConfig(memory_length=50),
            "episodes": 300,
            "alpha": 8,  # Less mismatch sensitivity
            "beta": 10,
        },
        {
            "name": "adaptive_vs_cautious",
            "config1": AdaptiveAgentConfig(memory_length=50),
            "config2": CautiousAgentConfig(memory_length=50),  # Same memory length
            "episodes": 300,
            "alpha": 6,  # High mismatch sensitivity
            "beta": 15,
        },
    ]

    # Run experiments
    for exp in experiments:
        try:
            # Run experiment
            runner.run_experiment(
                config1=exp["config1"],
                config2=exp["config2"],
                experiment_name=exp["name"],
                episodes=exp["episodes"],
                alpha=exp["alpha"],
                beta=exp["beta"],
            )

            # Analyze results
            # IF YOU WANT YOUR OLD GRAPHS runner.analyze_results(exp["name"])

        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print("Check the ./results/ folder for plots and ./models/ for saved agents.")


if __name__ == "__main__":
    main()
