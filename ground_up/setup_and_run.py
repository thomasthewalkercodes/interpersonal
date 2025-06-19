"""
Quick setup and run script for the SAC interpersonal behavior simulation.
This script helps you get started with minimal setup.
"""

import os
import sys


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ["torch", "numpy", "matplotlib"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"[MISSING] {package} is missing")

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False

    return True


def create_directories():
    """Create necessary directories."""
    directories = ["models", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created directory: {directory}")


def quick_demo():
    """Run a quick demonstration of the system."""
    print("\n" + "=" * 50)
    print("QUICK DEMO: SAC Interpersonal Behavior Learning")
    print("=" * 50)

    # Import our modules (assuming they're in the same directory)
    try:
        from agent_config import CooperativeAgentConfig, CompetitiveAgentConfig
        from corrected_agent_state import InterpersonalAgentState
        from sac_algorithm import SACAgent
        from interaction_environment import (
            InterpersonalEnvironment,
            SimplePayoffCalculator,
        )

        print("[OK] All modules imported successfully")

        # Create simple demonstration
        print("\n1. Creating cooperative and competitive agents...")

        # Create configs
        coop_config = CooperativeAgentConfig(memory_length=20)
        comp_config = CompetitiveAgentConfig(memory_length=20)

        # Create states
        state1 = coop_config.create_initial_state()
        state2 = comp_config.create_initial_state()
        state_dim = state1.get_state_dimension()

        print(f"[OK] State dimension: {state_dim}")

        # Create agents
        agent1 = SACAgent(state_dim, coop_config.get_sac_params())
        agent2 = SACAgent(state_dim, comp_config.get_sac_params())

        # Create environment
        payoff_calc = SimplePayoffCalculator()
        environment = InterpersonalEnvironment(
            payoff_calculator=payoff_calc,
            agent1_state=state1,
            agent2_state=state2,
            max_steps_per_episode=20,
        )

        print("[OK] Environment created")

        # Run a few interactions
        print("\n2. Running sample interactions...")

        state1, state2 = environment.reset()

        for step in range(5):
            action1 = agent1.select_action(state1, training=False)
            action2 = agent2.select_action(state2, training=False)

            next_state1, next_state2, reward1, reward2, done = environment.step(
                action1, action2
            )

            print(
                f"Step {step+1}: Actions=({action1:.2f}, {action2:.2f}), "
                f"Rewards=({reward1:.2f}, {reward2:.2f})"
            )

            state1, state2 = next_state1, next_state2

            if done:
                break

        print("\n[OK] Demo completed successfully!")
        print("\nYou can now run the full training with: python example_usage.py")

    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Make sure all the module files are in the same directory:")
        print("- interfaces.py")
        print("- corrected_agent_state.py")
        print("- agent_config.py")
        print("- sac_algorithm.py")
        print("- interaction_environment.py")
        print("- example_usage.py")
        return False

    except Exception as e:
        print(f"[ERROR] Error during demo: {e}")
        return False

    return True


def main():
    print("SAC Interpersonal Behavior Simulation Setup")
    print("=" * 44)

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing packages and try again.")
        return

    # Create directories
    create_directories()

    # Run quick demo
    if quick_demo():
        print("\n" + "=" * 50)
        print("SETUP COMPLETE!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Run full experiments: python example_usage.py")
        print("2. Modify agent configurations in agent_config.py")
        print("3. Implement your own payoff function in interaction_environment.py")
        print("4. Adjust SAC hyperparameters as needed")

        # Training time estimates
        print("\nTraining Time Estimates:")
        print("- Quick test (100 episodes): ~30 seconds")
        print("- Small experiment (1000 episodes): ~5 minutes")
        print("- Full experiment (2000 episodes): ~10-15 minutes")
        print("- GPU recommended for faster training")

    else:
        print("\n" + "=" * 50)
        print("SETUP ENCOUNTERED ISSUES")
        print("=" * 50)
        print("Please check the error messages above and fix any issues.")


if __name__ == "__main__":
    main()
