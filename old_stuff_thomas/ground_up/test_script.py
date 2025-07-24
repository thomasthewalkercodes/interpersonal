"""
Simple test script for the SAC interpersonal behavior simulation.
Windows-compatible version without special Unicode characters.
"""


def test_basic_functionality():
    """Test basic functionality of the SAC system."""
    print("Testing SAC Interpersonal Behavior System")
    print("=" * 45)

    # Test 1: Check imports
    print("\n1. Testing imports...")
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt

        print("   - PyTorch: OK")
        print("   - NumPy: OK")
        print("   - Matplotlib: OK")
    except ImportError as e:
        print(f"   - Missing package: {e}")
        return False

    # Test 2: Import our modules
    print("\n2. Testing custom modules...")
    try:
        from interfaces import (
            PayoffCalculator,
            AgentState,
            AgentConfig,
            ReinforcementLearner,
        )

        print("   - interfaces.py: OK")

        from agent_state import InterpersonalAgentState

        print("   - agent_state.py: OK")

        from agent_configuration import (
            BaseAgentConfig,
            CooperativeAgentConfig,
            CompetitiveAgentConfig,
        )

        print("   - agent_configuration.py: OK")

        from sac_algorithm import SACAgent, SACTrainer

        print("   - sac_algorithm.py: OK")

        from interaction_environment import (
            InterpersonalEnvironment,
            SimplePayoffCalculator,
        )

        print("   - interaction_environment.py: OK")

    except ImportError as e:
        print(f"   - Import error: {e}")
        print("   - Make sure all files are in the same directory")
        return False

    # Test 3: Create basic components
    print("\n3. Testing component creation...")
    try:
        # Create agent config
        config = BaseAgentConfig(memory_length=10)
        print("   - Agent config: OK")

        # Create agent state
        state = config.create_initial_state()
        state_dim = state.get_state_dimension()
        print(f"   - Agent state (dim={state_dim}): OK")

        # Create SAC agent
        agent = SACAgent(state_dim, config.get_sac_params())
        print("   - SAC agent: OK")

        # Create payoff calculator
        payoff_calc = SimplePayoffCalculator()
        print("   - Payoff calculator: OK")

    except Exception as e:
        print(f"   - Component creation error: {e}")
        return False

    # Test 4: Test single interaction
    print("\n4. Testing single interaction...")
    try:
        # Create two agents
        config1 = BaseAgentConfig(memory_length=5)
        config2 = BaseAgentConfig(memory_length=5)

        state1 = config1.create_initial_state()
        state2 = config2.create_initial_state()

        agent1 = SACAgent(state1.get_state_dimension(), config1.get_sac_params())
        agent2 = SACAgent(state2.get_state_dimension(), config2.get_sac_params())

        # Create environment
        environment = InterpersonalEnvironment(
            payoff_calculator=SimplePayoffCalculator(),
            agent1_state=state1,
            agent2_state=state2,
            max_steps_per_episode=10,
        )

        # Reset and get initial states
        initial_state1, initial_state2 = environment.reset()
        print(
            f"   - Initial states shape: {initial_state1.shape}, {initial_state2.shape}"
        )

        # Select actions
        action1 = agent1.select_action(initial_state1, training=False)
        action2 = agent2.select_action(initial_state2, training=False)
        print(f"   - Actions: {action1:.3f}, {action2:.3f}")

        # Execute step
        next_state1, next_state2, reward1, reward2, done = environment.step(
            action1, action2
        )
        print(f"   - Rewards: {reward1:.3f}, {reward2:.3f}")
        print("   - Single interaction: OK")

    except Exception as e:
        print(f"   - Interaction test error: {e}")
        return False

    # Test 5: Test training step
    print("\n5. Testing training step...")
    try:
        # Store a transition
        agent1.store_transition(initial_state1, action1, reward1, next_state1, done)
        agent2.store_transition(initial_state2, action2, reward2, next_state2, done)

        # Try training step (won't actually train with just one sample)
        metrics1 = agent1.train_step()
        metrics2 = agent2.train_step()

        print("   - Training step: OK")

    except Exception as e:
        print(f"   - Training test error: {e}")
        return False

    print("\n" + "=" * 45)
    print("ALL TESTS PASSED!")
    print("=" * 45)
    print("\nThe system is ready to use. You can now:")
    print("1. Run 'python example_usage.py' for full experiments")
    print("2. Modify agent configurations as needed")
    print("3. Implement your own payoff functions")

    return True


def quick_training_demo():
    """Run a very quick training demo (just a few episodes)."""
    print("\n" + "=" * 45)
    print("QUICK TRAINING DEMO")
    print("=" * 45)

    try:
        from agent_configuration import CooperativeAgentConfig, CompetitiveAgentConfig
        from agent_state import InterpersonalAgentState
        from sac_algorithm import SACAgent, SACTrainer
        from interaction_environment import (
            InterpersonalEnvironment,
            SimplePayoffCalculator,
        )

        # Create different agent types
        coop_config = CooperativeAgentConfig(memory_length=10)
        comp_config = CompetitiveAgentConfig(memory_length=10)

        # Create agents
        state1 = coop_config.create_initial_state()
        state2 = comp_config.create_initial_state()

        agent1 = SACAgent(state1.get_state_dimension(), coop_config.get_sac_params())
        agent2 = SACAgent(state2.get_state_dimension(), comp_config.get_sac_params())

        # Create environment
        environment = InterpersonalEnvironment(
            payoff_calculator=SimplePayoffCalculator(),
            agent1_state=state1,
            agent2_state=state2,
            max_steps_per_episode=20,
        )

        # Create trainer
        trainer = SACTrainer(
            agent1=agent1,
            agent2=agent2,
            environment=environment,
            payoff_calculator=environment.payoff_calculator,
            episodes_per_training=10,  # Just 10 episodes for demo
            steps_per_episode=10,
            evaluation_frequency=5,
        )

        print("Running quick training (10 episodes)...")
        results = trainer.train("./demo_models")

        print(f"Training completed!")
        print(
            f"Agent 1 final avg reward: {np.mean(results['episode_rewards']['agent1'][-5:]):.2f}"
        )
        print(
            f"Agent 2 final avg reward: {np.mean(results['episode_rewards']['agent2'][-5:]):.2f}"
        )

        return True

    except Exception as e:
        print(f"Training demo error: {e}")
        return False


if __name__ == "__main__":
    import numpy as np

    # Run basic tests
    if test_basic_functionality():
        # Ask if user wants to run quick training demo
        try:
            response = input("\nRun quick training demo? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                quick_training_demo()
        except KeyboardInterrupt:
            print("\nDemo skipped.")

        print("\nSystem is ready! Next steps:")
        print("- Run 'python example_usage.py' for full experiments")
        print("- Check generated models and results")
    else:
        print("\nPlease fix the errors above before proceeding.")
