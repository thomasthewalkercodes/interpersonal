"""
Complete Therapeutic Training Integration
Runs the therapeutic agent system with your existing SAC infrastructure.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, Tuple
import json

# Import your existing modules
from agent_configuration import CompetitiveAgentConfig, BaseAgentConfig
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment
from gaussian_payoff_graph import calculate_warmth_payoff
from interfaces import PayoffCalculator
from comprehensive_logging import LoggingTrainerWrapper

# Import the fixed therapeutic system
from therapeutic_agent_system import (
    TherapistAgent,
    TherapistSACWrapper,
    TherapeuticPhase,
)


class TherapeuticPayoffCalculator(PayoffCalculator):
    """
    Specialized payoff calculator for therapeutic interactions.
    Rewards therapist for patient progress and mutual warmth.
    """

    def __init__(self, alpha: float = 4.0, beta: float = 10.0):
        self.alpha = alpha
        self.beta = beta

    def calculate_payoff(
        self,
        therapist_action: float,
        patient_action: float,
        therapist_id: str,
        patient_id: str,
    ) -> Tuple[float, float]:
        """Calculate therapeutic payoffs using Gaussian base + therapeutic bonuses."""

        # Convert to warmth space [0, 1]
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Base payoff using your Gaussian function
        base_therapist_payoff = calculate_warmth_payoff(
            therapist_warmth, patient_warmth, self.alpha, self.beta
        )
        base_patient_payoff = calculate_warmth_payoff(
            patient_warmth, therapist_warmth, self.alpha, self.beta
        )

        # Therapeutic modifications for therapist

        # 1. Strong reward for patient warmth progress
        patient_warmth_bonus = patient_warmth * 3.0

        # 2. Bonus for mutual warmth (both warm)
        mutual_warmth_bonus = min(therapist_warmth, patient_warmth) * 2.5

        # 3. Penalty for therapist being too cold (should model warmth)
        therapist_coldness_penalty = (1.0 - therapist_warmth) * 1.0

        # 4. Bonus for leading (therapist warmer than patient)
        leadership_bonus = max(0, therapist_warmth - patient_warmth) * 1.5

        # 5. Stability bonus (reward consistent therapeutic direction)
        consistency_bonus = 0.5 if therapist_warmth >= 0.4 else 0.0

        # Calculate final payoffs
        therapist_payoff = (
            base_therapist_payoff
            + patient_warmth_bonus
            + mutual_warmth_bonus
            + leadership_bonus
            + consistency_bonus
            - therapist_coldness_penalty
        )

        # Patient gets base payoff plus some mutual warmth bonus
        patient_payoff = base_patient_payoff + mutual_warmth_bonus * 0.3

        return therapist_payoff, patient_payoff


class TherapeuticEnvironment(InterpersonalEnvironment):
    """
    Enhanced environment for therapeutic interactions with strategy evolution.
    """

    def __init__(
        self,
        therapist: TherapistAgent,
        patient_state: InterpersonalAgentState,
        payoff_calculator: TherapeuticPayoffCalculator,
        evolution_frequency: int = 100,
        max_steps_per_episode: int = 50,
    ):
        # Create a dummy second state for the parent class
        dummy_therapist_state = patient_state.__class__(
            memory_length=patient_state.memory_length,
            initial_trust=0.0,
            initial_satisfaction=0.0,
        )

        super().__init__(
            payoff_calculator=payoff_calculator,
            agent1_state=dummy_therapist_state,  # Therapist (we'll override this)
            agent2_state=patient_state,  # Patient
            agent1_id="therapist",
            agent2_id="patient",
            max_steps_per_episode=max_steps_per_episode,
        )

        self.therapist = therapist
        self.patient_state = patient_state
        self.evolution_frequency = evolution_frequency
        self.total_interactions = 0
        self.therapy_sessions_completed = 0

        # Track therapeutic progress
        self.session_results = []
        self.last_patient_action = 0.0

    def step(self, therapist_action: float, patient_action: float):
        """Execute one therapeutic interaction step."""

        # Store patient action for therapist
        self.therapist.last_patient_action = patient_action
        self.last_patient_action = patient_action

        # Calculate payoffs using therapeutic calculator
        therapist_payoff, patient_payoff = self.payoff_calculator.calculate_payoff(
            therapist_action, patient_action, "therapist", "patient"
        )

        # Update patient state (therapist doesn't have traditional state updates)
        self.patient_state.update_state(
            patient_action, therapist_action, patient_payoff
        )

        # Get next states
        patient_next_state = self.patient_state.get_state_vector()
        therapist_next_state = self._get_therapist_state()

        # Update counters
        self.current_step += 1
        self.total_interactions += 1

        # Check for episode termination
        done = self._check_therapeutic_termination()

        # Evolve therapist strategies periodically
        if (
            self.total_interactions % self.evolution_frequency == 0
            and self.total_interactions > 0
        ):
            print(
                f"\n[EVOLUTION] Evolving therapist strategies at step {self.total_interactions}"
            )
            self._evolve_and_report()

        return (
            therapist_next_state,
            patient_next_state,
            therapist_payoff,
            patient_payoff,
            done,
        )

    def reset(self):
        """Reset environment for new episode."""
        # Reset patient state
        self.patient_state.reset_state()
        self.current_step = 0
        self.episode_count += 1

        # Periodically restart therapy sessions
        if self.episode_count % 15 == 1:  # New therapy session every 15 episodes
            self.therapist._restart_therapy_session()
            self.therapy_sessions_completed += 1
            print(
                f"[THERAPY] Starting new therapy session #{self.therapy_sessions_completed}"
            )

        # Return initial states
        patient_initial_state = self.patient_state.get_state_vector()
        therapist_initial_state = self._get_therapist_state()

        return therapist_initial_state, patient_initial_state

    def _get_therapist_state(self):
        """Create therapist's state vector with therapeutic information."""

        # Get patient information
        patient_trust = (
            self.patient_state.get_trust_level()
            if hasattr(self.patient_state, "get_trust_level")
            else 0.0
        )
        patient_satisfaction = (
            self.patient_state.get_satisfaction_level()
            if hasattr(self.patient_state, "get_satisfaction_level")
            else 0.0
        )

        # Encode current therapeutic phase
        phase_encoding = {
            TherapeuticPhase.ASSESSMENT: 0.0,
            TherapeuticPhase.MATCHING: 0.2,
            TherapeuticPhase.LEADING: 0.4,
            TherapeuticPhase.STABILIZING: 0.6,
            TherapeuticPhase.ADVANCING: 0.8,
        }
        current_phase_encoding = phase_encoding.get(self.therapist.current_phase, 0.0)

        # Patient warmth estimate
        patient_warmth = (self.last_patient_action + 1) / 2

        # Create comprehensive state vector (matching your agent_state dimensions)
        base_state = np.array(
            [
                patient_trust,  # 0: Patient trust
                patient_satisfaction,  # 1: Patient satisfaction
                patient_warmth,  # 2: Patient warmth
                current_phase_encoding,  # 3: Therapy phase
                self.therapist.trust_trend,  # 4: Trust trend
                self.therapist.warmth_trend,  # 5: Warmth trend
                self.therapist.therapy_step / 100.0,  # 6: Normalized step count
                self.therapist.cycle_count / 10.0,  # 7: Normalized cycle count
                self.therapist.current_strategy.warmth_target,  # 8: Target warmth
                self.therapist.current_strategy.trust_threshold,  # 9: Trust threshold
            ],
            dtype=np.float32,
        )

        # Pad to match patient state dimension if needed
        patient_state_dim = len(self.patient_state.get_state_vector())
        if len(base_state) < patient_state_dim:
            padding = np.zeros(patient_state_dim - len(base_state), dtype=np.float32)
            state = np.concatenate([base_state, padding])
        else:
            state = base_state[:patient_state_dim]

        return state

    def _check_therapeutic_termination(self):
        """Check if episode should terminate based on therapeutic criteria."""

        # Standard termination
        if self.current_step >= self.max_steps_per_episode:
            return True

        # Early termination for therapeutic success
        if hasattr(self.patient_state, "get_trust_level") and hasattr(
            self.patient_state, "get_satisfaction_level"
        ):

            trust = self.patient_state.get_trust_level()

            # Success criteria: high trust + warm patient behavior
            if (
                trust > 0.6
                and len(self.therapist.patient_warmth_history) > 10
                and self.therapist.patient_warmth_history[-1] > 0.7
            ):
                print(
                    f"[SUCCESS] Therapeutic breakthrough! Trust: {trust:.3f}, Warmth: {self.therapist.patient_warmth_history[-1]:.3f}"
                )
                return True

        return False

    def _evolve_and_report(self):
        """Evolve strategies and generate progress report."""

        # Evaluate current strategy fitness
        fitness = self.therapist.evaluate_strategy_fitness()
        self.therapist.current_strategy.fitness_score = fitness

        # Evolve strategies
        self.therapist.evolve_strategies(generation_size=30)

        # Generate progress report
        report = self.therapist.get_therapeutic_report()
        print(f"[THERAPY PROGRESS]")

        if "patient_progress" in report:
            progress = report["patient_progress"]
            print(
                f"  Patient Warmth: {progress.get('baseline_warmth', 0):.3f} -> {progress.get('current_warmth', 0):.3f}"
            )
            print(f"  Trust Level: {progress.get('current_trust', 0):.3f}")
            print(f"  Warmth Progress: {progress.get('warmth_progress', 0):.3f}")

        if "strategy_info" in report:
            strategy = report["strategy_info"]
            print(
                f"  Strategy Fitness: {strategy.get('current_strategy_fitness', 0):.3f}"
            )
            print(f"  Strategy Generation: {strategy.get('strategy_generation', 0)}")

    def get_therapeutic_stats(self):
        """Get comprehensive therapeutic statistics."""
        return {
            "total_interactions": self.total_interactions,
            "therapy_sessions": self.therapy_sessions_completed,
            "current_phase": self.therapist.current_phase.value,
            "evolution_generations": len(self.therapist.evolution_history),
            "best_strategy_fitness": (
                max([gen["best_fitness"] for gen in self.therapist.evolution_history])
                if self.therapist.evolution_history
                else 0.0
            ),
            "patient_final_warmth": (
                self.therapist.patient_warmth_history[-1]
                if self.therapist.patient_warmth_history
                else 0.0
            ),
            "patient_final_trust": (
                self.therapist.patient_trust_history[-1]
                if self.therapist.patient_trust_history
                else 0.0
            ),
        }


def run_therapeutic_experiment(
    experiment_name: str = "therapeutic_intervention",
    episodes: int = 500,
    evolution_frequency: int = 75,
    therapist_population_size: int = 20,
):
    """
    Run a complete therapeutic experiment.

    Args:
        experiment_name: Name for the experiment
        episodes: Number of training episodes
        evolution_frequency: How often to evolve therapist strategies
        therapist_population_size: Size of therapist strategy population
    """

    print("=" * 70)
    print(f"THERAPEUTIC INTERVENTION EXPERIMENT: {experiment_name}")
    print("=" * 70)

    # Create cold/resistant patient configuration
    print("Creating resistant patient configuration...")
    patient_config = CompetitiveAgentConfig(
        initial_trust=-0.4,  # Start very distrustful
        initial_satisfaction=-0.3,  # Start dissatisfied
        memory_length=80,  # Long memory of negative experiences
        lr_actor=5e-4,  # Learn moderately fast
        lr_critic=5e-4,
        alpha=0.25,  # Moderate exploration
        noise_scale=0.08,  # Some behavioral consistency
    )

    # Create patient agent
    print("Creating patient agent...")
    patient_state = patient_config.create_initial_state()
    state_dim = patient_state.get_state_dimension()
    patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

    print(f"Patient state dimension: {state_dim}")
    print(f"Patient initial trust: {patient_state.get_trust_level():.3f}")
    print(f"Patient initial satisfaction: {patient_state.get_satisfaction_level():.3f}")

    # Create therapist with evolved strategies
    print(f"Creating therapist with {therapist_population_size} strategies...")
    therapist = TherapistAgent(
        agent_id="therapeutic_agent",
        population_size=therapist_population_size,
        elite_ratio=0.3,
        mutation_strength=0.15,
    )

    # Create therapist wrapper for SAC integration
    therapist_wrapper = TherapistSACWrapper(therapist, state_dim)

    # Create therapeutic payoff calculator
    payoff_calculator = TherapeuticPayoffCalculator(alpha=4.0, beta=8.0)

    # Create therapeutic environment
    print("Creating therapeutic environment...")
    environment = TherapeuticEnvironment(
        therapist=therapist,
        patient_state=patient_state,
        payoff_calculator=payoff_calculator,
        evolution_frequency=evolution_frequency,
        max_steps_per_episode=60,  # Longer episodes for therapy
    )

    # Create trainer
    print("Creating SAC trainer...")
    trainer = SACTrainer(
        agent1=therapist_wrapper,  # Therapist
        agent2=patient_agent,  # Patient
        environment=environment,
        payoff_calculator=payoff_calculator,
        episodes_per_training=episodes,
        steps_per_episode=60,
        evaluation_frequency=50,
        save_frequency=200,
        training_frequency=1,
    )

    # Add comprehensive logging
    print("Initializing comprehensive logging...")
    logging_wrapper = LoggingTrainerWrapper(trainer, experiment_name)

    # Train the therapeutic interaction
    print(f"\nStarting therapeutic training...")
    print(f"Episodes: {episodes}")
    print(f"Evolution frequency: {evolution_frequency}")
    print(f"Expected therapy sessions: {episodes // 15}")
    print("-" * 50)

    # Run training with logging
    results = logging_wrapper.train_with_logging(
        f"./therapeutic_models/{experiment_name}"
    )

    # Get final therapeutic statistics
    final_stats = environment.get_therapeutic_stats()

    # Generate final report
    print("\n" + "=" * 70)
    print("THERAPEUTIC EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"Training completed successfully!")
    print(f"Total interactions: {final_stats['total_interactions']}")
    print(f"Therapy sessions: {final_stats['therapy_sessions']}")
    print(f"Final therapy phase: {final_stats['current_phase']}")
    print(f"Strategy evolution generations: {final_stats['evolution_generations']}")
    print(f"Best strategy fitness achieved: {final_stats['best_strategy_fitness']:.3f}")

    print(f"\nPatient Progress:")
    print(f"  Final warmth level: {final_stats['patient_final_warmth']:.3f}")
    print(f"  Final trust level: {final_stats['patient_final_trust']:.3f}")

    # Calculate success metrics
    warmth_improvement = (
        final_stats["patient_final_warmth"] - 0.2
    )  # Assuming started at ~0.2
    trust_improvement = final_stats["patient_final_trust"] - (-0.4)  # Started at -0.4

    print(f"  Warmth improvement: {warmth_improvement:.3f}")
    print(f"  Trust improvement: {trust_improvement:.3f}")

    # Success criteria
    therapeutic_success = (
        final_stats["patient_final_warmth"] > 0.6
        and final_stats["patient_final_trust"] > 0.3
    )

    print(f"\nTherapeutic Success: {'YES' if therapeutic_success else 'IN PROGRESS'}")

    if therapeutic_success:
        print("The therapist successfully guided the patient toward warm cooperation!")
    else:
        print("Patient showing progress but may need additional therapy sessions.")

    # Save comprehensive results
    def make_json_serializable(obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, "__dict__"):
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, "value"):  # Handle enums
            return obj.value
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # Convert everything else to string

    # Clean therapist evolution history for JSON
    clean_evolution_history = []
    for entry in therapist.evolution_history:
        clean_entry = {
            "generation": int(entry["generation"]),
            "best_fitness": float(entry["best_fitness"]),
            "average_fitness": float(entry["average_fitness"]),
            "best_strategy_summary": {
                "matching_intensity": float(entry["best_strategy"].matching_intensity),
                "trust_threshold": float(entry["best_strategy"].trust_threshold),
                "leading_step_size": float(entry["best_strategy"].leading_step_size),
                "warmth_target": float(entry["best_strategy"].warmth_target),
                "fitness_score": float(entry["best_strategy"].fitness_score),
            },
        }
        clean_evolution_history.append(clean_entry)

    results_data = {
        "experiment_name": experiment_name,
        "training_results": make_json_serializable(results),
        "therapeutic_stats": final_stats,
        "evolution_summary": {
            "total_generations": len(clean_evolution_history),
            "final_best_fitness": (
                clean_evolution_history[-1]["best_fitness"]
                if clean_evolution_history
                else 0.0
            ),
            "evolution_history": clean_evolution_history,
        },
        "final_strategy": {
            "matching_intensity": float(therapist.current_strategy.matching_intensity),
            "trust_threshold": float(therapist.current_strategy.trust_threshold),
            "leading_step_size": float(therapist.current_strategy.leading_step_size),
            "warmth_target": float(therapist.current_strategy.warmth_target),
            "fitness_score": float(therapist.current_strategy.fitness_score),
            "generation": int(therapist.current_strategy.generation),
        },
        "success_metrics": {
            "therapeutic_success": therapeutic_success,
            "warmth_improvement": float(warmth_improvement),
            "trust_improvement": float(trust_improvement),
        },
    }

    # Save to file
    results_file = f"./therapeutic_results_{experiment_name}.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Comprehensive logs and visualizations: {results['comprehensive_logs']}")
    print(f"Model checkpoints: ./therapeutic_models/{experiment_name}/")

    return results_data, environment, therapist, patient_agent


def run_multiple_therapeutic_experiments():
    """Run multiple therapeutic experiments with different configurations."""

    experiments = [
        {
            "name": "quick_intervention",
            "episodes": 300,
            "evolution_frequency": 50,
            "population_size": 15,
        },
        {
            "name": "intensive_therapy",
            "episodes": 600,
            "evolution_frequency": 75,
            "population_size": 25,
        },
        {
            "name": "long_term_treatment",
            "episodes": 800,
            "evolution_frequency": 100,
            "population_size": 30,
        },
    ]

    all_results = []

    for exp in experiments:
        print(f"\n\nStarting experiment: {exp['name']}")
        print("=" * 50)

        try:
            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name=exp["name"],
                episodes=exp["episodes"],
                evolution_frequency=exp["evolution_frequency"],
                therapist_population_size=exp["population_size"],
            )

            all_results.append(results)

        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")
            import traceback

            traceback.print_exc()

    # Compare results
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    for result in all_results:
        name = result["experiment_name"]
        stats = result["therapeutic_stats"]
        success = result["success_metrics"]

        print(f"\n{name}:")
        print(f"  Therapy Sessions: {stats['therapy_sessions']}")
        print(f"  Evolution Generations: {stats['evolution_generations']}")
        print(f"  Final Warmth: {stats['patient_final_warmth']:.3f}")
        print(f"  Final Trust: {stats['patient_final_trust']:.3f}")
        print(f"  Success: {'YES' if success['therapeutic_success'] else 'NO'}")

    return all_results


# =============================================================================
# EXPERIMENT CONFIGURATION - CHANGE THESE TO RUN DIFFERENT EXPERIMENTS
# =============================================================================

# Choose experiment type by changing this variable:
# "quick_test"     - Fast test (150 episodes, ~3-5 minutes)
# "single"         - Single experiment (500 episodes, ~10-15 minutes)
# "intensive"      - Intensive therapy (800 episodes, ~20-25 minutes)
# "comparison"     - Multiple experiments comparison (~30-45 minutes)

EXPERIMENT_TYPE = "single"  # <-- CHANGE THIS TO RUN DIFFERENT EXPERIMENTS

# Advanced configuration (optional to modify)
CUSTOM_CONFIG = {
    "experiment_name": "my_therapeutic_experiment",
    "episodes": 400,
    "evolution_frequency": 60,
    "therapist_population_size": 18,
}

# =============================================================================


def main():
    """Main function - runs experiment based on EXPERIMENT_TYPE setting above."""

    print("THERAPEUTIC AGENT TRAINING SYSTEM")
    print("=" * 50)
    print(f"Running experiment type: {EXPERIMENT_TYPE}")
    print("-" * 50)

    try:
        if EXPERIMENT_TYPE == "quick_test":
            print("Running QUICK TEST (fast verification)...")
            print("Episodes: 150, Population: 10, Evolution frequency: 30")
            print("Expected time: 3-5 minutes")
            print("-" * 50)

            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name="quick_test",
                episodes=150,
                evolution_frequency=30,
                therapist_population_size=10,
            )

        elif EXPERIMENT_TYPE == "single":
            print("Running SINGLE THERAPEUTIC EXPERIMENT...")
            print("Episodes: 500, Population: 20, Evolution frequency: 75")
            print("Expected time: 10-15 minutes")
            print("-" * 50)

            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name="single_therapy_session",
                episodes=500,
                evolution_frequency=75,
                therapist_population_size=20,
            )

        elif EXPERIMENT_TYPE == "intensive":
            print("Running INTENSIVE THERAPY EXPERIMENT...")
            print("Episodes: 800, Population: 25, Evolution frequency: 100")
            print("Expected time: 20-25 minutes")
            print("-" * 50)

            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name="intensive_therapy",
                episodes=800,
                evolution_frequency=100,
                therapist_population_size=25,
            )

        elif EXPERIMENT_TYPE == "comparison":
            print("Running MULTIPLE EXPERIMENT COMPARISON...")
            print("Will run 3 different configurations")
            print("Expected time: 30-45 minutes")
            print("-" * 50)

            all_results = run_multiple_therapeutic_experiments()

        elif EXPERIMENT_TYPE == "custom":
            print("Running CUSTOM CONFIGURATION...")
            config = CUSTOM_CONFIG
            print(f"Episodes: {config['episodes']}")
            print(f"Population: {config['therapist_population_size']}")
            print(f"Evolution frequency: {config['evolution_frequency']}")
            print("-" * 50)

            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name=config["experiment_name"],
                episodes=config["episodes"],
                evolution_frequency=config["evolution_frequency"],
                therapist_population_size=config["therapist_population_size"],
            )

        else:
            print(f"Unknown experiment type: {EXPERIMENT_TYPE}")
            print(
                "Valid options: 'quick_test', 'single', 'intensive', 'comparison', 'custom'"
            )
            print("Running default quick test...")
            print("-" * 50)

            results, env, therapist, patient = run_therapeutic_experiment(
                experiment_name="default_test",
                episodes=150,
                evolution_frequency=30,
                therapist_population_size=10,
            )

        print("\n" + "=" * 50)
        print("THERAPEUTIC TRAINING COMPLETED!")
        print("=" * 50)
        print("Generated files:")
        print("- Comprehensive logs and charts")
        print("- Model checkpoints")
        print("- Therapeutic progress visualizations")
        print("- Evolution history data")
        print("\nTo run a different experiment:")
        print("1. Change EXPERIMENT_TYPE at the top of this file")
        print("2. Run the script again")
        print("\nExperiment options:")
        print("- 'quick_test': Fast 3-5 minute test")
        print("- 'single': Standard 10-15 minute experiment")
        print("- 'intensive': Deep 20-25 minute therapy")
        print("- 'comparison': 30-45 minute multi-experiment")
        print("- 'custom': Use CUSTOM_CONFIG settings")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
