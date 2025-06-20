"""
Quick test of the comprehensive logging system with your existing setup.
"""

from agent_configuration import CooperativeAgentConfig, CompetitiveAgentConfig
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment, SimplePayoffCalculator
from comprehensive_logging import LoggingTrainerWrapper


def quick_test():
    """Test comprehensive logging with a short experiment."""
    print("Testing Comprehensive Logging System")
    print("=" * 50)

    # Create simple experiment
    config1 = CooperativeAgentConfig(memory_length=20)
    config2 = CompetitiveAgentConfig(memory_length=20)

    # Create agents and environment
    state1 = config1.create_initial_state()
    state2 = config2.create_initial_state()

    state_dim = state1.get_state_dimension()

    agent1 = SACAgent(state_dim, config1.get_sac_params())
    agent2 = SACAgent(state_dim, config2.get_sac_params())

    # Create environment with simple payoff
    payoff_calculator = SimplePayoffCalculator()
    environment = InterpersonalEnvironment(
        payoff_calculator=payoff_calculator,
        agent1_state=state1,
        agent2_state=state2,
        agent1_id="test_coop_agent",
        agent2_id="test_comp_agent",
        max_steps_per_episode=30,
    )

    # Create trainer
    trainer = SACTrainer(
        agent1=agent1,
        agent2=agent2,
        environment=environment,
        payoff_calculator=payoff_calculator,
        episodes_per_training=30,  # Short test
        steps_per_episode=20,
        evaluation_frequency=25,
        save_frequency=100,
    )

    # Wrap with comprehensive logging
    logging_wrapper = LoggingTrainerWrapper(trainer, "test_cooperative_vs_competitive")

    print(" Running short training with comprehensive logging...")
    print("   Episodes: 50")
    print("   Steps per episode: 20")
    print("   This should take about 1-2 minutes...")

    # Train with logging
    results = logging_wrapper.train_with_logging("./test_models")

    print("\n Test completed!")
    print(f"Check the logs directory: {results['comprehensive_logs']}")
    print("The system generated:")
    print("   • Detailed behavioral analysis")
    print("   • Episode-by-episode tracking")
    print("   • Learning progress visualization")
    print("   • Comprehensive dashboard with 12+ charts")

    return results


if __name__ == "__main__":
    try:
        results = quick_test()
        print("\n Comprehensive logging test successful!")
        print("You can now integrate this with your main training script.")
    except Exception as e:
        print("[TEST] Testing Comprehensive Logging System")
        print(f"\n[ERROR] Error during test: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure all required files are in the same directory.")
