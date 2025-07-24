"""
Example usage of the SAC Control Center for interpersonal agent simulations.

This script demonstrates various ways to use the control system:
1. Simple pairwise interactions
2. Custom agent configurations
3. Comparison studies
4. Parameter sweeps
"""

import os
import sys
from typing import List, Dict, Any

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from control_center.sac_control import SACSControlCenter, SimulationConfig
from sim_plots.sac_plot import SimulationPlotter


def example_1_basic_interaction():
    """Example 1: Basic interaction between two different agent types."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Cooperative vs Competitive Interaction")
    print("=" * 60)

    control = SACSControlCenter()

    # Simple configuration
    config = SimulationConfig(
        agent1_type="cooperative",
        agent2_type="competitive",
        episodes=500,
        steps_per_episode=30,
        save_plots=True,
        run_name="example_1_coop_vs_comp",
    )

    results = control.run_simulation(config)

    # Print summary
    final_eval = results["final_evaluation"]
    print(f"\nFinal Results:")
    print(f"Cooperative agent average reward: {final_eval['agent1_avg_reward']:.3f}")
    print(f"Competitive agent average reward: {final_eval['agent2_avg_reward']:.3f}")

    return results


def example_2_custom_agents():
    """Example 2: Custom agent configurations with specific parameters."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Agent Configurations")
    print("=" * 60)

    control = SACSControlCenter()

    # Create a trusting agent vs a suspicious agent
    config = SimulationConfig(
        agent1_type="base",
        agent2_type="base",
        agent1_custom_params={
            "initial_trust": 0.8,  # Very trusting
            "memory_length": 20,  # Short memory (forgiving)
            "lr_actor": 2e-4,  # Slower learning
            "noise_scale": 0.05,  # Less exploration
        },
        agent2_custom_params={
            "initial_trust": -0.7,  # Very suspicious
            "memory_length": 100,  # Long memory (holds grudges)
            "lr_actor": 8e-4,  # Faster learning
            "noise_scale": 0.2,  # More exploration
        },
        episodes=800,
        steps_per_episode=40,
        payoff_alpha=3.0,  # Custom payoff parameters
        payoff_beta=8.0,
        save_plots=True,
        run_name="example_2_trust_vs_suspicion",
    )

    results = control.run_simulation(config)

    # Analyze trust dynamics
    print(f"\nCustom Agent Results:")
    print(
        f"Trusting agent performance: {results['final_evaluation']['agent1_avg_reward']:.3f}"
    )
    print(
        f"Suspicious agent performance: {results['final_evaluation']['agent2_avg_reward']:.3f}"
    )

    return results


def example_3_comparison_study():
    """Example 3: Systematic comparison of all agent types."""
    print("=" * 60)
    print("EXAMPLE 3: Systematic Agent Type Comparison")
    print("=" * 60)

    control = SACSControlCenter()

    # Configuration for comparison study
    base_config = SimulationConfig(
        agent1_type="base",  # Will be overridden
        agent2_type="base",  # Will be overridden
        episodes=300,
        steps_per_episode=25,
        save_models=False,  # Don't save models for comparison
        save_plots=False,  # Generate summary plots instead
        output_dir="./comparison_results",
    )

    # Compare all agent types
    agent_types = ["cooperative", "competitive", "adaptive", "cautious"]

    comparison_results = control.run_comparison_study(
        agent_types=agent_types,
        base_config=base_config,
        num_runs=3,  # Run each pairing 3 times
    )

    # Analyze results
    print(f"\nComparison Study Results:")
    for pair, runs in comparison_results.items():
        avg_rewards = []
        for run in runs:
            eval_result = run["final_evaluation"]
            total_reward = (
                eval_result["agent1_avg_reward"] + eval_result["agent2_avg_reward"]
            )
            avg_rewards.append(total_reward)

        mean_performance = sum(avg_rewards) / len(avg_rewards)
        print(f"{pair}: Average total reward = {mean_performance:.3f}")

    return comparison_results


def example_4_parameter_sweep():
    """Example 4: Parameter sweep to understand payoff function effects."""
    print("=" * 60)
    print("EXAMPLE 4: Payoff Parameter Sweep")
    print("=" * 60)

    control = SACSControlCenter()

    # Test different payoff parameters
    alpha_values = [2.0, 4.0, 6.0]  # Mismatch penalty
    beta_values = [5.0, 10.0, 15.0]  # Risk penalty

    sweep_results = {}

    for alpha in alpha_values:
        for beta in beta_values:
            print(f"Testing α={alpha}, β={beta}")

            config = SimulationConfig(
                agent1_type="adaptive",
                agent2_type="adaptive",
                episodes=400,
                payoff_alpha=alpha,
                payoff_beta=beta,
                save_models=False,
                save_plots=False,
                run_name=f"sweep_alpha_{alpha}_beta_{beta}",
            )

            result = control.run_simulation(config)

            # Store key metrics
            final_eval = result["final_evaluation"]
            sweep_results[f"α={alpha}_β={beta}"] = {
                "total_reward": final_eval["agent1_avg_reward"]
                + final_eval["agent2_avg_reward"],
                "agent1_reward": final_eval["agent1_avg_reward"],
                "agent2_reward": final_eval["agent2_avg_reward"],
                "alpha": alpha,
                "beta": beta,
            }

    # Print parameter sweep results
    print(f"\nParameter Sweep Results:")
    print("Configuration\t\tTotal Reward\tAgent1\tAgent2")
    print("-" * 60)
    for config_name, metrics in sweep_results.items():
        print(
            f"{config_name}\t{metrics['total_reward']:.3f}\t\t{metrics['agent1_reward']:.3f}\t{metrics['agent2_reward']:.3f}"
        )

    return sweep_results


def example_5_longitudinal_analysis():
    """Example 5: Longitudinal analysis of agent development."""
    print("=" * 60)
    print("EXAMPLE 5: Longitudinal Development Analysis")
    print("=" * 60)

    control = SACSControlCenter()

    # Run a longer simulation to see development patterns
    config = SimulationConfig(
        agent1_type="adaptive",
        agent2_type="cautious",
        episodes=2000,  # Longer simulation
        steps_per_episode=50,
        evaluation_frequency=200,  # More frequent evaluation
        save_frequency=500,
        save_models=True,
        save_plots=True,
        run_name="example_5_longitudinal_development",
    )

    results = control.run_simulation(config)

    # Analyze learning phases
    episode_rewards = results["training_results"]["episode_rewards"]
    agent1_rewards = episode_rewards["agent1"]
    agent2_rewards = episode_rewards["agent2"]

    # Divide into phases
    phase_size = len(agent1_rewards) // 4
    phases = ["Early", "Mid-Early", "Mid-Late", "Late"]

    print(f"\nLongitudinal Analysis:")
    print("Phase\t\tAgent1 Avg\tAgent2 Avg\tDifference")
    print("-" * 60)

    for i, phase in enumerate(phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 3 else len(agent1_rewards)

        phase_rewards1 = agent1_rewards[start_idx:end_idx]
        phase_rewards2 = agent2_rewards[start_idx:end_idx]

        avg1 = sum(phase_rewards1) / len(phase_rewards1)
        avg2 = sum(phase_rewards2) / len(phase_rewards2)
        diff = avg1 - avg2

        print(f"{phase}\t\t{avg1:.3f}\t\t{avg2:.3f}\t\t{diff:+.3f}")

    return results


def example_6_interactive_demo():
    """Example 6: Interactive demo where user can specify configurations."""
    print("=" * 60)
    print("EXAMPLE 6: Interactive Configuration Demo")
    print("=" * 60)

    control = SACSControlCenter()

    # Show available options
    print("Available agent types:")
    for agent_type, description in control.list_available_configs().items():
        print(f"  {agent_type}: {description}")

    # Get user input (for demo, we'll use predefined values)
    # In a real interactive version, you could use input() here

    print(f"\nRunning demo with user-specified configuration...")

    # Example "user choices"
    user_config = SimulationConfig(
        agent1_type="cooperative",
        agent2_type="adaptive",
        episodes=600,
        steps_per_episode=35,
        payoff_alpha=3.5,
        payoff_beta=7.5,
        save_plots=True,
        run_name="interactive_demo_coop_vs_adaptive",
    )

    results = control.run_simulation(user_config)

    # Provide detailed feedback
    print(f"\nInteractive Demo Results:")
    print(f"Configuration: {user_config.agent1_type} vs {user_config.agent2_type}")
    print(f"Episodes run: {user_config.episodes}")
    print(
        f"Payoff parameters: α={user_config.payoff_alpha}, β={user_config.payoff_beta}"
    )

    final_eval = results["final_evaluation"]
    winner = (
        "Agent 1"
        if final_eval["agent1_avg_reward"] > final_eval["agent2_avg_reward"]
        else "Agent 2"
    )
    print(f"Better performer: {winner}")

    return results


def run_all_examples():
    """Run all examples in sequence."""
    print("Running all examples...")

    examples = [
        example_1_basic_interaction,
        example_2_custom_agents,
        example_3_comparison_study,
        example_4_parameter_sweep,
        example_5_longitudinal_analysis,
        example_6_interactive_demo,
    ]

    results = {}

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n\n{'='*80}")
            print(f"RUNNING EXAMPLE {i}")
            print(f"{'='*80}")

            result = example_func()
            results[f"example_{i}"] = result

            print(f"Example {i} completed successfully!")

        except Exception as e:
            print(f"Error in example {i}: {e}")
            results[f"example_{i}"] = {"error": str(e)}

    return results


def create_payoff_visualization():
    """Create standalone payoff landscape visualization."""
    print("=" * 60)
    print("BONUS: Payoff Landscape Visualization")
    print("=" * 60)

    plotter = SimulationPlotter()

    # Create payoff landscapes for different parameter settings
    os.makedirs("./payoff_analysis", exist_ok=True)

    parameter_sets = [
        (2.0, 5.0),  # Gentle
        (4.0, 10.0),  # Standard
        (6.0, 15.0),  # Harsh
    ]

    for alpha, beta in parameter_sets:
        print(f"Creating payoff landscape for α={alpha}, β={beta}")
        plotter.plot_payoff_landscape(
            alpha=alpha,
            beta=beta,
            output_dir=f"./payoff_analysis/alpha_{alpha}_beta_{beta}",
        )

    print("Payoff visualizations saved to ./payoff_analysis/")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        example_num = sys.argv[1]

        if example_num == "all":
            run_all_examples()
        elif example_num == "payoff":
            create_payoff_visualization()
        elif example_num.isdigit():
            example_functions = {
                "1": example_1_basic_interaction,
                "2": example_2_custom_agents,
                "3": example_3_comparison_study,
                "4": example_4_parameter_sweep,
                "5": example_5_longitudinal_analysis,
                "6": example_6_interactive_demo,
            }

            if example_num in example_functions:
                example_functions[example_num]()
            else:
                print(f"Unknown example number: {example_num}")
        else:
            print(f"Unknown command: {example_num}")
    else:
        # Default: run a quick demo
        print("No arguments provided. Running quick demo...")
        print("Use 'python example_usage.py <1-6|all|payoff>' for specific examples")

        control = SACSControlCenter()
        results = control.quick_run("cooperative", "competitive", episodes=200)

        final_eval = results["final_evaluation"]
        print(f"\nQuick Demo Results:")
        print(f"Cooperative: {final_eval['agent1_avg_reward']:.3f}")
        print(f"Competitive: {final_eval['agent2_avg_reward']:.3f}")

        print(f"\nTo run specific examples:")
        print(f"  python example_usage.py 1    # Basic interaction")
        print(f"  python example_usage.py 2    # Custom agents")
        print(f"  python example_usage.py 3    # Comparison study")
        print(f"  python example_usage.py 4    # Parameter sweep")
        print(f"  python example_usage.py 5    # Longitudinal analysis")
        print(f"  python example_usage.py 6    # Interactive demo")
        print(f"  python example_usage.py all  # All examples")
        print(f"  python example_usage.py payoff # Payoff visualizations")
