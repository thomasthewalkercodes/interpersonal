"""
SAC Control Center - Main interface for running interpersonal agent simulations.

This module provides a clean interface for setting up and running simulations
between different types of SAC agents with configurable parameters.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Import your existing modules
from agent_configs.sac_agents import (
    BaseAgentConfig,
    CooperativeAgentConfig,
    CompetitiveAgentConfig,
    AdaptiveAgentConfig,
    CautiousAgentConfig,
)
from ml_algos.sac_algo import SACAgent, SACTrainer
from payoff_functions.gaussian_payoff import calculate_warmth_payoff


@dataclass
class SimulationConfig:
    """Configuration for a complete simulation run."""

    # Agent configuration
    agent1_type: str  # 'cooperative', 'competitive', 'adaptive', 'cautious', 'custom'
    agent2_type: str
    agent1_custom_params: Optional[Dict[str, Any]] = None
    agent2_custom_params: Optional[Dict[str, Any]] = None

    # Simulation parameters
    episodes: int = 1000
    steps_per_episode: int = 50
    training_frequency: int = 1
    evaluation_frequency: int = 100
    save_frequency: int = 500

    # Environment parameters
    payoff_alpha: float = 4.0  # Mismatch penalty factor
    payoff_beta: float = 10.0  # Risk factor weight

    # Output configuration
    save_models: bool = True
    save_plots: bool = True
    output_dir: str = "./results"
    run_name: str = None  # Auto-generated if None


class SimpleEnvironment:
    """
    Simple environment for two-agent warmth interactions.

    This environment simulates interpersonal interactions where agents
    exchange warmth levels and receive payoffs based on their matching
    and risk-taking behavior.
    """

    def __init__(self, payoff_alpha: float = 4.0, payoff_beta: float = 10.0):
        self.payoff_alpha = payoff_alpha
        self.payoff_beta = payoff_beta
        self.state_dim = (
            4  # [own_last_action, other_last_action, own_trust, other_perceived_warmth]
        )

        # Initialize state tracking
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment and return initial states for both agents."""
        # Initial state: [last_action, other_last_action, trust, other_warmth]
        self.state1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.step_count = 0
        self.history = {"agent1_actions": [], "agent2_actions": [], "rewards": []}

        return self.state1.copy(), self.state2.copy()

    def step(
        self, action1: float, action2: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Execute one step of the interaction.

        Args:
            action1: Warmth level from agent 1 ([-1, 1] -> [0, 1])
            action2: Warmth level from agent 2 ([-1, 1] -> [0, 1])

        Returns:
            next_state1, next_state2, reward1, reward2, done
        """
        # Convert actions from [-1, 1] to [0, 1] for warmth
        warmth1 = (action1 + 1) / 2
        warmth2 = (action2 + 1) / 2

        # Calculate payoffs
        reward1 = calculate_warmth_payoff(
            warmth1, warmth2, self.payoff_alpha, self.payoff_beta
        )
        reward2 = calculate_warmth_payoff(
            warmth2, warmth1, self.payoff_alpha, self.payoff_beta
        )

        # Update trust based on interaction outcome
        trust_delta1 = 0.1 * (reward1 - 0.5)  # Trust increases with good outcomes
        trust_delta2 = 0.1 * (reward2 - 0.5)

        # Update states
        new_trust1 = np.clip(self.state1[2] + trust_delta1, -1.0, 1.0)
        new_trust2 = np.clip(self.state2[2] + trust_delta2, -1.0, 1.0)

        # Next states: [own_last_action, other_last_action, own_trust, other_perceived_warmth]
        next_state1 = np.array(
            [action1, action2, new_trust1, warmth2], dtype=np.float32
        )
        next_state2 = np.array(
            [action2, action1, new_trust2, warmth1], dtype=np.float32
        )

        # Store history
        self.history["agent1_actions"].append(warmth1)
        self.history["agent2_actions"].append(warmth2)
        self.history["rewards"].append((reward1, reward2))

        # Update internal state
        self.state1 = next_state1.copy()
        self.state2 = next_state2.copy()
        self.step_count += 1

        # Episode ends after max steps or if both agents become very cold
        done = (self.step_count >= 50) or (warmth1 < 0.1 and warmth2 < 0.1)

        return next_state1, next_state2, reward1, reward2, done


class SACSControlCenter:
    """
    Main control center for running SAC agent simulations.

    This class provides a clean interface for:
    - Configuring different agent types and their interactions
    - Running simulations with various parameters
    - Saving results and generating plots
    """

    AGENT_TYPES = {
        "cooperative": CooperativeAgentConfig,
        "competitive": CompetitiveAgentConfig,
        "adaptive": AdaptiveAgentConfig,
        "cautious": CautiousAgentConfig,
        "base": BaseAgentConfig,
    }

    def __init__(self):
        self.results_history = []

    def create_agent_config(
        self, agent_type: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> BaseAgentConfig:
        """Create an agent configuration based on type and custom parameters."""
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {list(self.AGENT_TYPES.keys())}"
            )

        config_class = self.AGENT_TYPES[agent_type]

        if custom_params:
            return config_class(**custom_params)
        else:
            return config_class()

    def setup_simulation(
        self, config: SimulationConfig
    ) -> Tuple[SACAgent, SACAgent, SimpleEnvironment, SACTrainer]:
        """
        Set up a complete simulation based on the provided configuration.

        Returns:
            agent1, agent2, environment, trainer
        """
        # Create environment
        environment = SimpleEnvironment(
            payoff_alpha=config.payoff_alpha, payoff_beta=config.payoff_beta
        )

        # Create agent configurations
        agent1_config = self.create_agent_config(
            config.agent1_type, config.agent1_custom_params
        )
        agent2_config = self.create_agent_config(
            config.agent2_type, config.agent2_custom_params
        )

        # Create agents
        agent1 = SACAgent(environment.state_dim, agent1_config.get_sac_params())
        agent2 = SACAgent(environment.state_dim, agent2_config.get_sac_params())

        # Create trainer
        trainer = SACTrainer(
            agent1=agent1,
            agent2=agent2,
            environment=environment,
            payoff_calculator=calculate_warmth_payoff,
            episodes_per_training=config.episodes,
            steps_per_episode=config.steps_per_episode,
            training_frequency=config.training_frequency,
            evaluation_frequency=config.evaluation_frequency,
            save_frequency=config.save_frequency,
        )

        return agent1, agent2, environment, trainer

    def run_simulation(self, config: SimulationConfig) -> Dict[str, Any]:
        """
        Run a complete simulation and return results.

        Args:
            config: Simulation configuration

        Returns:
            Dictionary containing simulation results and metadata
        """
        # Generate run name if not provided
        if config.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.run_name = (
                f"{config.agent1_type}_vs_{config.agent2_type}_{timestamp}"
            )

        print(f"Starting simulation: {config.run_name}")
        print(f"Agent 1: {config.agent1_type}")
        print(f"Agent 2: {config.agent2_type}")
        print(f"Episodes: {config.episodes}")

        # Setup simulation
        agent1, agent2, environment, trainer = self.setup_simulation(config)

        # Create output directory
        run_dir = os.path.join(config.output_dir, config.run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save configuration
        config_dict = asdict(config)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Run training
        training_results = trainer.train(
            save_dir=run_dir if config.save_models else None
        )

        # Create results summary
        results = {
            "config": config_dict,
            "training_results": training_results,
            "run_name": config.run_name,
            "output_dir": run_dir,
            "final_evaluation": trainer.evaluate(num_episodes=20),
            "environment_history": environment.history,
        }

        # Save results
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Store in history
        self.results_history.append(results)

        print(f"Simulation complete! Results saved to: {run_dir}")

        # Generate plots if requested
        if config.save_plots:
            self.generate_plots(results, run_dir)

        return results

    def generate_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate and save plots for the simulation results."""
        try:
            # Import here to avoid dependency issues if plotting not needed
            from sim_plots.sac_plot import SimulationPlotter

            plotter = SimulationPlotter()
            plotter.create_all_plots(results, output_dir)
            print(f"Plots saved to: {output_dir}")

        except ImportError:
            print("Warning: Could not import plotting module. Plots not generated.")
        except Exception as e:
            print(f"Warning: Error generating plots: {e}")

    def run_comparison_study(
        self, agent_types: List[str], base_config: SimulationConfig, num_runs: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run a comparison study between different agent types.

        Args:
            agent_types: List of agent types to compare
            base_config: Base configuration to use for all runs
            num_runs: Number of runs per agent pair

        Returns:
            Dictionary mapping agent pairs to their results
        """
        comparison_results = {}

        for i, agent1_type in enumerate(agent_types):
            for j, agent2_type in enumerate(agent_types):
                if i <= j:  # Avoid duplicate comparisons
                    pair_key = f"{agent1_type}_vs_{agent2_type}"
                    comparison_results[pair_key] = []

                    for run in range(num_runs):
                        config = SimulationConfig(
                            agent1_type=agent1_type,
                            agent2_type=agent2_type,
                            episodes=base_config.episodes,
                            steps_per_episode=base_config.steps_per_episode,
                            payoff_alpha=base_config.payoff_alpha,
                            payoff_beta=base_config.payoff_beta,
                            save_models=False,  # Don't save models for comparison runs
                            save_plots=False,  # Generate plots separately
                            output_dir=base_config.output_dir,
                            run_name=f"{pair_key}_run_{run+1}",
                        )

                        result = self.run_simulation(config)
                        comparison_results[pair_key].append(result)

        # Save comparison summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = os.path.join(
            base_config.output_dir, f"comparison_study_{timestamp}"
        )
        os.makedirs(comparison_dir, exist_ok=True)

        with open(os.path.join(comparison_dir, "comparison_results.json"), "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)

        # Generate comparison plots
        try:
            from plotting.simulation_plots import SimulationPlotter

            plotter = SimulationPlotter()
            plotter.create_comparison_plots(comparison_results, comparison_dir)
        except ImportError:
            print("Warning: Could not import plotting module for comparison plots.")

        return comparison_results

    def list_available_configs(self) -> Dict[str, str]:
        """Return a dictionary of available agent configurations and their descriptions."""
        return {
            "cooperative": "Higher initial trust, less exploration, forgiving memory",
            "competitive": "Lower initial trust, more exploration, longer memory",
            "adaptive": "Fast learning, quick adjustments, balanced approach",
            "cautious": "Slow learning, conservative exploration, very long memory",
            "base": "Default configuration with standard parameters",
        }

    def quick_run(
        self,
        agent1_type: str,
        agent2_type: str,
        episodes: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Quick simulation run with default parameters.

        Args:
            agent1_type: Type of first agent
            agent2_type: Type of second agent
            episodes: Number of episodes to run
            verbose: Whether to print progress

        Returns:
            Simulation results
        """
        config = SimulationConfig(
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            episodes=episodes,
            save_models=False,
            save_plots=True,
        )

        if verbose:
            print(f"Quick run: {agent1_type} vs {agent2_type}")

        return self.run_simulation(config)


def main():
    """Example usage of the control center."""

    # Initialize control center
    control = SACSControlCenter()

    # Show available configurations
    print("Available agent configurations:")
    for agent_type, description in control.list_available_configs().items():
        print(f"  {agent_type}: {description}")

    # Example 1: Quick simulation run
    print("\n" + "=" * 50)
    print("Example 1: Quick run - Cooperative vs Competitive")
    print("=" * 50)

    results = control.quick_run("cooperative", "competitive", episodes=300)

    # Example 2: Custom configuration
    print("\n" + "=" * 50)
    print("Example 2: Custom configuration with specific parameters")
    print("=" * 50)

    custom_config = SimulationConfig(
        agent1_type="adaptive",
        agent2_type="cautious",
        episodes=500,
        steps_per_episode=30,
        payoff_alpha=2.0,  # Less mismatch penalty
        payoff_beta=5.0,  # Less risk penalty
        save_models=True,
        save_plots=True,
        run_name="adaptive_vs_cautious_gentle_payoff",
    )

    results = control.run_simulation(custom_config)

    # Example 3: Custom agent parameters
    print("\n" + "=" * 50)
    print("Example 3: Custom agent parameters")
    print("=" * 50)

    custom_agent_config = SimulationConfig(
        agent1_type="base",
        agent2_type="base",
        agent1_custom_params={
            "lr_actor": 1e-3,
            "initial_trust": 0.8,
            "memory_length": 20,
        },
        agent2_custom_params={
            "lr_actor": 5e-4,
            "initial_trust": -0.5,
            "memory_length": 80,
        },
        episodes=400,
        run_name="custom_trust_experiment",
    )

    results = control.run_simulation(custom_agent_config)

    # Example 4: Comparison study
    print("\n" + "=" * 50)
    print("Example 4: Comparison study")
    print("=" * 50)

    comparison_config = SimulationConfig(
        agent1_type="base",  # This will be overridden
        agent2_type="base",  # This will be overridden
        episodes=200,
        output_dir="./comparison_results",
    )

    agent_types = ["cooperative", "competitive", "adaptive"]
    comparison_results = control.run_comparison_study(
        agent_types=agent_types, base_config=comparison_config, num_runs=2
    )

    print("Comparison study complete!")


if __name__ == "__main__":
    main()
