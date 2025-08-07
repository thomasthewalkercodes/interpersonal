# test_continuous_bayesian.py (place in project root, not in src/)
"""
Simple test script for continuous Bayesian agents
Run from project root directory: python test_continuous_bayesian.py
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Now imports should work
from continuous_bayesian_agent import create_continuous_bayesian_agent
from continuous_simulation import run_continuous_simulation
from payoff_functions import gaussian_matching_payoff


def test_basic_functionality():
    """Test basic functionality of the continuous Bayesian system"""

    print("Testing Continuous Bayesian Learning System")
    print("=" * 50)

    # Test 1: Create agents
    print("1. Creating agents...")
    try:
        agent1 = create_continuous_bayesian_agent(
            agent_id="test_agent_1",
            prior_mean=0.6,
            prior_confidence=2.0,
            lambda_loss=1.5,
        )

        agent2 = create_continuous_bayesian_agent(
            agent_id="test_agent_2",
            prior_mean=0.4,
            prior_confidence=1.5,
            lambda_loss=2.0,
        )

        print("✓ Agents created successfully")
        print(f"Agent 1: {agent1}")
        print(f"Agent 2: {agent2}")

    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False

    # Test 2: Test payoff function
    print("\n2. Testing payoff function...")
    try:
        test_payoffs = [
            (0.9, 0.9, "Warm-Warm"),
            (0.1, 0.1, "Cold-Cold"),
            (0.9, 0.1, "Mismatch 1"),
            (0.1, 0.9, "Mismatch 2"),
            (0.5, 0.5, "Medium-Medium"),
        ]

        for my_action, opp_action, description in test_payoffs:
            payoff = gaussian_matching_payoff(my_action, opp_action)
            print(f"  {description}: {payoff:.2f}")

        print("✓ Payoff function working")

    except Exception as e:
        print(f"✗ Payoff function failed: {e}")
        return False

    # Test 3: Test belief updating
    print("\n3. Testing belief updates...")
    try:
        import numpy as np

        initial_belief = agent1.get_belief_stats()["warmth_belief"]
        print(f"  Initial belief about opponent: {initial_belief:.3f}")

        # Simulate some opponent actions
        opponent_actions = [0.8, 0.7, 0.9, 0.6, 0.8]  # Mostly warm actions

        for action in opponent_actions:
            agent1.update_beliefs(np.array([action]))

        final_belief = agent1.get_belief_stats()["warmth_belief"]
        print(f"  Final belief after observing {opponent_actions}: {final_belief:.3f}")
        print(f"  Belief moved from {initial_belief:.3f} to {final_belief:.3f}")

        if final_belief > initial_belief:
            print("✓ Belief updating working correctly (learned opponent is warm)")
        else:
            print("? Belief updating may have issue")

    except Exception as e:
        print(f"✗ Belief updating failed: {e}")
        return False

    # Test 4: Short simulation
    print("\n4. Testing short simulation...")
    try:
        # Reset agents for clean test
        agent1 = create_continuous_bayesian_agent(
            agent_id="sim_agent_1", prior_mean=0.7, lambda_loss=1.0
        )

        agent2 = create_continuous_bayesian_agent(
            agent_id="sim_agent_2", prior_mean=0.3, lambda_loss=2.0
        )

        results = run_continuous_simulation(
            agent1=agent1,
            agent2=agent2,
            payoff_function1=gaussian_matching_payoff,
            payoff_function2=gaussian_matching_payoff,
            n_rounds=50,  # Short simulation for testing
            verbose=False,
        )

        print(f"  Simulation completed: {results.summary['total_rounds']} rounds")
        print(f"  Agent 1 final action: {results.agent1_actions[-1]:.3f}")
        print(f"  Agent 2 final action: {results.agent2_actions[-1]:.3f}")
        print(
            f"  Final action difference: {results.summary['final_action_difference']:.3f}"
        )

        print("✓ Simulation completed successfully")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All basic tests passed!")
    print("The continuous Bayesian learning system is working correctly.")

    return True


def test_action_selection():
    """Test different action selection methods"""

    print("\n5. Testing action selection methods...")

    try:
        from continuous_action_selection import select_action_continuous

        agent = create_continuous_bayesian_agent(prior_mean=0.5)

        methods = ["thompson", "ucb", "softmax", "greedy"]

        for method in methods:
            action = select_action_continuous(
                agent, gaussian_matching_payoff, method=method
            )
            print(f"  {method}: {action:.3f}")

        print("✓ All action selection methods working")

    except Exception as e:
        print(f"✗ Action selection failed: {e}")
        return False

    return True


def main():
    """Main test function"""

    print("Continuous Bayesian Learning - System Test")
    print("=" * 60)

    # Run tests
    basic_test_passed = test_basic_functionality()
    action_test_passed = test_action_selection()

    if basic_test_passed and action_test_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your continuous Bayesian learning system is ready to use.")
        print("\nNext steps:")
        print("- Run the full example: python examples/continuous_example.py")
        print("- Modify parameters and experiment")
        print("- Add self-confidence parameters as discussed")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
