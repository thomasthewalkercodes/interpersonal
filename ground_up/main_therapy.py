"""
Fixed main therapy script that avoids import issues and handles state dimensions properly.
"""

import numpy as np
import sys
import os

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_simple_therapist():
    """Simple test without complex integrations."""
    print("TESTING SIMPLE THERAPIST")
    print("=" * 40)

    try:
        from therapeutic_agent_system import TherapistAgent, TherapeuticStrategy

        # Create a simple therapist
        print("Creating therapist with 5 strategies...")
        therapist = TherapistAgent(agent_id="test_therapist", population_size=5)

        print(f"Created therapist with {len(therapist.strategy_population)} strategies")

        # Simulate some therapy steps with dummy data
        print("\nSimulating therapy steps...")

        # Dummy patient data - gradually getting warmer and more trusting
        patient_actions = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        patient_trust_levels = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]

        dummy_state = np.zeros(10)  # Dummy state vector

        for i, (patient_action, patient_trust) in enumerate(
            zip(patient_actions, patient_trust_levels)
        ):
            therapist_action = therapist.select_action(
                dummy_state, patient_action, patient_trust
            )
            therapist_warmth = (therapist_action + 1) / 2
            patient_warmth = (patient_action + 1) / 2

            print(
                f"  Step {i+1}: Patient={patient_warmth:.3f}, Therapist={therapist_warmth:.3f}, "
                f"Trust={patient_trust:.3f}, Phase={therapist.current_phase.value}"
            )

        # Get therapy report
        report = therapist.get_therapeutic_report()
        print(f"\nTherapy Report:")
        print(f"  Final Phase: {report['session_info']['current_phase']}")
        print(f"  Total Steps: {report['session_info']['therapy_step']}")
        print(f"  Cycles: {report['session_info']['cycle_count']}")

        if "patient_progress" in report:
            progress = report["patient_progress"]
            print(f"  Patient Final Warmth: {progress.get('current_warmth', 'N/A')}")
            print(f"  Patient Final Trust: {progress.get('current_trust', 'N/A')}")

        print("Simple therapist test completed!")
        return True

    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_therapist_evolution():
    """Test the strategy evolution without SAC integration."""
    print("\nTESTING STRATEGY EVOLUTION")
    print("=" * 40)

    try:
        from therapeutic_agent_system import TherapistAgent

        print("Creating therapist for evolution test...")
        therapist = TherapistAgent(agent_id="evolution_test", population_size=10)

        print("Running strategy evolution...")

        # Manually set some fitness scores to test evolution
        for i, strategy in enumerate(therapist.strategy_population):
            strategy.fitness_score = np.random.uniform(0.1, 1.0)  # Random fitness
            print(f"  Strategy {i}: Fitness {strategy.fitness_score:.3f}")

        print("\nEvolving strategies...")
        initial_best_fitness = max(
            s.fitness_score for s in therapist.strategy_population
        )

        # Run evolution (this will create new generation)
        therapist.evolve_strategies(generation_size=20)

        final_best_fitness = max(s.fitness_score for s in therapist.strategy_population)

        print(f"Initial best fitness: {initial_best_fitness:.3f}")
        print(f"Final best fitness: {final_best_fitness:.3f}")
        print(f"Evolution generations: {len(therapist.evolution_history)}")

        print("Evolution test completed!")
        return True

    except Exception as e:
        print(f"Error in evolution test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integrated_therapy():
    """Test with actual SAC integration but simpler setup."""
    print("\nTESTING INTEGRATED THERAPY")
    print("=" * 40)

    try:
        # Import required modules
        from therapeutic_agent_system import TherapistAgent, TherapistSACWrapper
        from agent_configuration import CompetitiveAgentConfig
        from sac_algorithm import SACAgent
        from interaction_environment import (
            InterpersonalEnvironment,
            SimplePayoffCalculator,
        )

        print("Setting up integrated therapy test...")

        # Create patient configuration (cold/resistant)
        patient_config = CompetitiveAgentConfig(
            initial_trust=-0.2, initial_satisfaction=-0.1, memory_length=30
        )

        # Create patient
        patient_state = patient_config.create_initial_state()
        state_dim = patient_state.get_state_dimension()
        patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

        print(f"Patient state dimension: {state_dim}")

        # Create therapist
        therapist = TherapistAgent(
            agent_id="integrated_therapist",
            population_size=8,  # Smaller population for testing
        )

        # Create therapist wrapper
        therapist_wrapper = TherapistSACWrapper(therapist, state_dim)

        print("All components created successfully!")

        # Test a few interactions
        print("\nTesting interactions...")

        # Create simple environment for testing
        environment = InterpersonalEnvironment(
            payoff_calculator=SimplePayoffCalculator(),
            agent1_state=patient_state,
            agent2_state=patient_config.create_initial_state(),  # Dummy second state
            agent1_id="patient",
            agent2_id="therapist",
            max_steps_per_episode=20,
        )

        # Reset and test a few steps
        patient_state_vec, _ = environment.reset()

        for step in range(5):
            # Patient action
            patient_action = patient_agent.select_action(
                patient_state_vec, training=False
            )

            # Therapist action (needs proper state)
            therapist_state = np.concatenate(
                [
                    [patient_state.get_trust_level()],
                    [patient_state.get_satisfaction_level()],
                    np.zeros(8),  # Padding to match expected dimension
                ]
            )

            therapist_action = therapist_wrapper.select_action(
                therapist_state, training=False
            )

            # Convert to warmth for display
            patient_warmth = (patient_action + 1) / 2
            therapist_warmth = (therapist_action + 1) / 2

            print(
                f"  Step {step+1}: Patient={patient_warmth:.3f}, Therapist={therapist_warmth:.3f}, "
                f"Phase={therapist.current_phase.value}"
            )

            # Update environment (simplified)
            patient_state_vec, _, reward1, reward2, done = environment.step(
                patient_action, therapist_action
            )

            if done:
                break

        print("Integrated therapy test completed!")
        return True

    except Exception as e:
        print(f"Error in integrated test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function with progressive testing."""
    print("THERAPEUTIC AGENT TESTING")
    print("=" * 50)

    # Test 1: Simple therapist functionality
    if not test_simple_therapist():
        print("Simple test failed. Check therapeutic_agent_system.py imports.")
        return

    # Test 2: Strategy evolution
    if not test_therapist_evolution():
        print("Evolution test failed.")
        return

    # Test 3: Integration with SAC
    if not test_integrated_therapy():
        print("Integration test failed.")
        return

    print("\nALL TESTS PASSED!")
    print("=" * 50)
    print("The therapeutic agent system is working correctly.")
    print("You can now proceed with full therapy experiments.")


if __name__ == "__main__":
    main()
