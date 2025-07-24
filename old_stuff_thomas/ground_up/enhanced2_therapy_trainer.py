"""
Integrated Visible Therapeutic Training
Combines visible evolution with your real SAC training and Gaussian payoffs.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, Tuple
import json
import copy
import random

# Import your existing modules
from agent_configuration import CompetitiveAgentConfig, BaseAgentConfig
from agent_state import InterpersonalAgentState
from sac_algorithm import SACAgent, SACTrainer
from interaction_environment import InterpersonalEnvironment
from gaussian_payoff_graph import calculate_warmth_payoff
from interfaces import PayoffCalculator
from comprehensive_logging import LoggingTrainerWrapper

# Import the working therapeutic system
from therapeutic_agent_system import (
    TherapistAgent,
    TherapistSACWrapper,
    TherapeuticPhase,
    TherapeuticStrategy,
)


class VisibleTherapeuticPayoffCalculator(PayoffCalculator):
    """
    Your Gaussian payoff calculator with enhanced therapist incentives.
    Shows exactly what payoffs each agent gets.
    """

    def __init__(
        self, alpha: float = 4.0, beta: float = 10.0, show_payoffs: bool = True
    ):
        self.alpha = alpha
        self.beta = beta
        self.show_payoffs = show_payoffs
        self.payoff_count = 0

    def calculate_payoff(
        self,
        therapist_action: float,
        patient_action: float,
        therapist_id: str,
        patient_id: str,
    ) -> Tuple[float, float]:
        """Calculate payoffs using your Gaussian system + therapist bonuses."""

        # Convert to warmth space
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Patient gets YOUR EXACT Gaussian payoffs (unchanged)
        patient_payoff = calculate_warmth_payoff(
            patient_warmth, therapist_warmth, self.alpha, self.beta
        )

        # Therapist base payoff (also Gaussian)
        therapist_base = calculate_warmth_payoff(
            therapist_warmth, patient_warmth, self.alpha, self.beta
        )

        # THERAPIST-ONLY enhancements for guiding toward warm-warm
        therapist_bonus = 0.0

        # 1. Massive bonus for patient warmth progress
        patient_warmth_reward = patient_warmth**2 * 8.0

        # 2. Huge bonus for warm-warm interactions
        if therapist_warmth >= 0.6 and patient_warmth >= 0.6:
            warm_warm_bonus = 15.0
            if therapist_warmth >= 0.8 and patient_warmth >= 0.8:
                warm_warm_bonus += 10.0
        else:
            warm_warm_bonus = 0.0

        # 3. Leadership bonus for appropriate leading
        if therapist_warmth > patient_warmth and patient_warmth > 0.3:
            leadership_bonus = (therapist_warmth - patient_warmth) * 5.0
        else:
            leadership_bonus = 0.0

        # 4. Breakthrough bonus for getting patient above thresholds
        breakthrough_bonus = 0.0
        if patient_warmth > 0.7:
            breakthrough_bonus = 20.0
        elif patient_warmth > 0.5:
            breakthrough_bonus = 8.0

        # Combine bonuses
        therapist_bonus = (
            patient_warmth_reward
            + warm_warm_bonus
            + leadership_bonus
            + breakthrough_bonus
        )

        # Final payoffs
        therapist_payoff = therapist_base + therapist_bonus

        # Show payoffs periodically for visibility
        self.payoff_count += 1
        if self.show_payoffs and self.payoff_count % 200 == 0:
            print(
                f"[PAYOFF SAMPLE] T_warmth:{therapist_warmth:.3f}, P_warmth:{patient_warmth:.3f}"
            )
            print(f"   Patient payoff: {patient_payoff:.2f} (pure Gaussian)")
            print(
                f"   Therapist base: {therapist_base:.2f}, bonus: {therapist_bonus:.2f}, total: {therapist_payoff:.2f}"
            )

        return therapist_payoff, patient_payoff


class VisibleTherapistAgent(TherapistAgent):
    """
    Enhanced therapist that shows evolution while working with real SAC training.
    """

    def __init__(self, agent_id: str = "visible_therapist", population_size: int = 8):
        super().__init__(agent_id, population_size)

        # Simple visible tracking
        self.strategy_names = []
        self.strategy_performance = {}
        self.generation_count = 0
        self.current_strategy_id = 0

        # Create descriptive names
        self._create_visible_names()

        print(f"\n🧬 VISIBLE THERAPIST POPULATION CREATED")
        print(f"Population Size: {population_size}")
        print("=" * 60)
        self._show_strategies()

    def _create_visible_names(self):
        """Create visible names based on strategy characteristics."""
        for i, strategy in enumerate(self.strategy_population):
            # Create name based on parameters
            if strategy.matching_intensity > 0.9:
                style = "Patient"
            elif strategy.leading_step_size > 0.2:
                style = "Bold"
            elif strategy.trust_threshold < 0.3:
                style = "Trusting"
            elif strategy.warmth_target > 0.8:
                style = "Ambitious"
            else:
                style = "Balanced"

            name = f"{style}_Therapist_{i+1}"
            self.strategy_names.append(name)

            # Initialize performance tracking
            self.strategy_performance[i] = {
                "name": name,
                "fitness_scores": [],
                "warm_warm_successes": 0,
                "total_interactions": 0,
                "generation": 0,
                "is_elite": False,
                "parents": [],
            }

    def _show_strategies(self):
        """Show current strategies with clear differences."""
        print("CURRENT STRATEGIES:")
        print("-" * 40)

        for i, strategy in enumerate(self.strategy_population):
            name = self.strategy_names[i]
            perf = self.strategy_performance[i]

            print(f"🧠 {name}")
            print(
                f"   Matching: {strategy.matching_intensity:.3f} ({'Precise' if strategy.matching_intensity > 0.9 else 'Flexible'})"
            )
            print(
                f"   Trust req: {strategy.trust_threshold:.3f} ({'Low' if strategy.trust_threshold < 0.3 else 'High'})"
            )
            print(
                f"   Leading: {strategy.leading_step_size:.3f} ({'Bold' if strategy.leading_step_size > 0.2 else 'Gentle'})"
            )
            print(
                f"   Target: {strategy.warmth_target:.3f} ({'High' if strategy.warmth_target > 0.8 else 'Moderate'})"
            )
            print(f"   Patience: {strategy.stabilization_patience} steps")

            if perf["fitness_scores"]:
                avg_fitness = np.mean(perf["fitness_scores"])
                print(f"   Avg fitness: {avg_fitness:.3f}")

            if perf["parents"]:
                print(f"   Parents: {', '.join(perf['parents'])}")

            print()

    def select_action(
        self, state: np.ndarray, patient_action: float, patient_trust: float
    ) -> float:
        """Select action while tracking which strategy is being used."""
        # Find current strategy ID
        for i, strategy in enumerate(self.strategy_population):
            if strategy == self.current_strategy:
                self.current_strategy_id = i
                break

        # Track interaction
        self.strategy_performance[self.current_strategy_id]["total_interactions"] += 1

        # Call parent method
        action = super().select_action(state, patient_action, patient_trust)

        return action

    def track_warm_warm_success(self, therapist_action: float, patient_action: float):
        """Track warm-warm interactions for current strategy."""
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        if therapist_warmth >= 0.6 and patient_warmth >= 0.6:
            self.strategy_performance[self.current_strategy_id][
                "warm_warm_successes"
            ] += 1

    def evolve_strategies(self, generation_size: int = 50):
        """Enhanced evolution with visible tracking."""
        self.generation_count += 1

        print(f"\n🧬 EVOLUTION GENERATION {self.generation_count}")
        print("=" * 60)

        # Evaluate all strategies using parent method first
        super().evolve_strategies(generation_size)

        # Then do visible tracking
        print(f"\n📊 STRATEGY PERFORMANCE ANALYSIS:")
        print("-" * 40)

        # Collect and sort results
        results = []
        for i, strategy in enumerate(self.strategy_population):
            name = (
                self.strategy_names[i]
                if i < len(self.strategy_names)
                else f"Strategy_{i}"
            )
            fitness = strategy.fitness_score
            perf = self.strategy_performance.get(
                i, {"warm_warm_successes": 0, "total_interactions": 1}
            )

            # Update performance
            if i in self.strategy_performance:
                self.strategy_performance[i]["fitness_scores"].append(fitness)

            results.append((i, fitness, name, perf))

        # Sort by fitness
        results.sort(key=lambda x: x[1], reverse=True)

        # Show rankings
        for rank, (strategy_id, fitness, name, perf) in enumerate(results):
            emoji = (
                "🥇"
                if rank == 0
                else "🥈" if rank == 1 else "🥉" if rank == 2 else "📊"
            )

            warm_warm_rate = 0.0
            if perf["total_interactions"] > 0:
                warm_warm_rate = (
                    perf["warm_warm_successes"] / perf["total_interactions"]
                ) * 100

            print(f"{emoji} #{rank+1}: {name}")
            print(f"    Fitness: {fitness:.3f}")
            print(f"    Warm-warm rate: {warm_warm_rate:.1f}%")
            print(f"    Interactions: {perf['total_interactions']}")
            print()

        # Update names and performance tracking for new generation
        self._update_after_evolution(results)

        print(f"✨ Generation {self.generation_count} complete!")
        print(f"   Best strategy: {results[0][2]}")
        print(f"   Best fitness: {results[0][1]:.3f}")

    def _update_after_evolution(self, results):
        """Update tracking after evolution."""
        # Keep track of which strategies survived
        elite_count = max(2, int(len(self.strategy_population) * 0.3))

        # Update names for new population
        new_names = []
        new_performance = {}

        for i in range(len(self.strategy_population)):
            if i < len(results):
                old_name = results[i][2]
                if i < elite_count:
                    # Elite survivor
                    new_name = old_name + "_evolved"
                    new_performance[i] = copy.deepcopy(
                        self.strategy_performance.get(results[i][0], {})
                    )
                    new_performance[i]["name"] = new_name
                    new_performance[i]["generation"] = self.generation_count
                    new_performance[i]["is_elite"] = True
                else:
                    # New offspring
                    new_name = f"Hybrid_{i-elite_count+1}_gen{self.generation_count}"
                    new_performance[i] = {
                        "name": new_name,
                        "fitness_scores": [],
                        "warm_warm_successes": 0,
                        "total_interactions": 0,
                        "generation": self.generation_count,
                        "is_elite": False,
                        "parents": [results[j][2] for j in range(min(2, elite_count))],
                    }

                new_names.append(new_name)
            else:
                new_name = f"NewStrategy_{i}"
                new_names.append(new_name)

        self.strategy_names = new_names
        self.strategy_performance = new_performance


class VisibleTherapeuticEnvironment(InterpersonalEnvironment):
    """
    Enhanced environment that works with your SAC training + visible evolution.
    """

    def __init__(
        self,
        therapist: VisibleTherapistAgent,
        patient_state: InterpersonalAgentState,
        payoff_calculator: VisibleTherapeuticPayoffCalculator,
        evolution_frequency: int = 100,
        max_steps_per_episode: int = 60,
    ):
        # Create dummy therapist state for parent class
        dummy_therapist_state = patient_state.__class__(
            memory_length=patient_state.memory_length,
            initial_trust=0.0,
            initial_satisfaction=0.0,
        )

        super().__init__(
            payoff_calculator=payoff_calculator,
            agent1_state=dummy_therapist_state,
            agent2_state=patient_state,
            agent1_id="visible_therapist",
            agent2_id="patient",
            max_steps_per_episode=max_steps_per_episode,
        )

        self.therapist = therapist
        self.patient_state = patient_state
        self.evolution_frequency = evolution_frequency
        self.total_interactions = 0
        self.therapy_sessions_completed = 0
        self.last_patient_action = 0.0

    def step(self, therapist_action: float, patient_action: float):
        """Execute step with visible tracking."""

        # Store for therapist
        self.therapist.last_patient_action = patient_action
        self.last_patient_action = patient_action

        # Track warm-warm for current strategy
        self.therapist.track_warm_warm_success(therapist_action, patient_action)

        # Calculate payoffs using YOUR Gaussian system
        therapist_payoff, patient_payoff = self.payoff_calculator.calculate_payoff(
            therapist_action, patient_action, "visible_therapist", "patient"
        )

        # Update patient state (YOUR existing system)
        self.patient_state.update_state(
            patient_action, therapist_action, patient_payoff
        )

        # Get next states
        patient_next_state = self.patient_state.get_state_vector()
        therapist_next_state = self._get_therapist_state()

        # Update counters
        self.current_step += 1
        self.total_interactions += 1

        # Check termination
        done = self._check_termination()

        # Evolve strategies periodically
        if (
            self.total_interactions % self.evolution_frequency == 0
            and self.total_interactions > 0
        ):
            print(f"\n🧬 EVOLVING STRATEGIES at interaction {self.total_interactions}")
            self.therapist.evolve_strategies(generation_size=40)
            self.therapist._show_strategies()

        return (
            therapist_next_state,
            patient_next_state,
            therapist_payoff,
            patient_payoff,
            done,
        )

    def reset(self):
        """Reset with session tracking."""
        self.patient_state.reset_state()
        self.current_step = 0
        self.episode_count += 1

        # New therapy session every 15 episodes
        if self.episode_count % 15 == 1:
            self.therapist._restart_therapy_session()
            self.therapy_sessions_completed += 1
            print(f"\n🏥 NEW THERAPY SESSION #{self.therapy_sessions_completed}")
            print(
                f"   Using strategy: {self.therapist.strategy_names[self.therapist.current_strategy_id]}"
            )

        # Return initial states
        patient_initial_state = self.patient_state.get_state_vector()
        therapist_initial_state = self._get_therapist_state()

        return therapist_initial_state, patient_initial_state

    def _get_therapist_state(self):
        """Create therapist state (same as your therapy_training.py)."""
        patient_trust = self.patient_state.get_trust_level()
        patient_satisfaction = self.patient_state.get_satisfaction_level()
        patient_warmth = (self.last_patient_action + 1) / 2

        # Encode therapeutic phase
        phase_encoding = {
            TherapeuticPhase.ASSESSMENT: 0.0,
            TherapeuticPhase.MATCHING: 0.2,
            TherapeuticPhase.LEADING: 0.4,
            TherapeuticPhase.STABILIZING: 0.6,
            TherapeuticPhase.ADVANCING: 0.8,
        }
        current_phase_encoding = phase_encoding.get(self.therapist.current_phase, 0.0)

        # Create state vector
        base_state = np.array(
            [
                patient_trust,
                patient_satisfaction,
                patient_warmth,
                current_phase_encoding,
                self.therapist.trust_trend,
                self.therapist.warmth_trend,
                self.therapist.therapy_step / 100.0,
                self.therapist.cycle_count / 10.0,
                self.therapist.current_strategy.warmth_target,
                self.therapist.current_strategy.trust_threshold,
            ],
            dtype=np.float32,
        )

        # Pad to match patient state dimension
        patient_state_dim = len(self.patient_state.get_state_vector())
        if len(base_state) < patient_state_dim:
            padding = np.zeros(patient_state_dim - len(base_state), dtype=np.float32)
            state = np.concatenate([base_state, padding])
        else:
            state = base_state[:patient_state_dim]

        return state

    def _check_termination(self):
        """Check termination (same logic as therapy_training.py)."""
        if self.current_step >= self.max_steps_per_episode:
            return True

        # Success criteria
        if hasattr(self.patient_state, "get_trust_level"):
            trust = self.patient_state.get_trust_level()
            patient_warmth = (self.last_patient_action + 1) / 2

            if trust > 0.6 and patient_warmth > 0.7:
                print(
                    f"[SUCCESS] Therapeutic breakthrough! Trust: {trust:.3f}, Warmth: {patient_warmth:.3f}"
                )
                return True

        return False


def run_visible_therapeutic_experiment(
    experiment_name: str = "visible_therapeutic_training",
    episodes: int = 400,
    evolution_frequency: int = 75,
    population_size: int = 8,
    alpha: float = 4.0,
    beta: float = 10.0,
):
    """
    Run therapeutic experiment with visible evolution using your real SAC + Gaussian system.
    """

    print("=" * 80)
    print(f"VISIBLE THERAPEUTIC TRAINING: {experiment_name}")
    print("Using your real SAC training + Gaussian payoffs + visible evolution")
    print("=" * 80)

    # Create resistant patient (EXACT same as therapy_training.py)
    patient_config = CompetitiveAgentConfig(
        initial_trust=-0.4,
        initial_satisfaction=-0.3,
        memory_length=80,
        lr_actor=5e-4,
        lr_critic=5e-4,
        alpha=0.25,
        noise_scale=0.08,
    )

    # Create patient agent (YOUR real SAC agent)
    patient_state = patient_config.create_initial_state()
    state_dim = patient_state.get_state_dimension()
    patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

    # Create visible therapist
    visible_therapist = VisibleTherapistAgent(
        agent_id="visible_therapeutic_agent",
        population_size=population_size,
    )

    # Create SAC wrapper (same as therapy_training.py)
    therapist_wrapper = TherapistSACWrapper(visible_therapist, state_dim)

    # Create visible payoff calculator (YOUR Gaussian system + bonuses)
    payoff_calculator = VisibleTherapeuticPayoffCalculator(
        alpha=alpha, beta=beta, show_payoffs=True
    )

    # Create visible environment
    environment = VisibleTherapeuticEnvironment(
        therapist=visible_therapist,
        patient_state=patient_state,
        payoff_calculator=payoff_calculator,
        evolution_frequency=evolution_frequency,
        max_steps_per_episode=60,
    )

    print(f"✓ Patient: Resistant (trust={patient_state.get_trust_level():.3f})")
    print(f"✓ Payoffs: YOUR Gaussian system (α={alpha}, β={beta}) + therapist bonuses")
    print(f"✓ Training: YOUR SAC algorithm")
    print(f"✓ Therapist: {population_size} visible evolving strategies")
    print(f"✓ Evolution: every {evolution_frequency} episodes")
    print("-" * 80)

    # Create trainer (YOUR real SAC trainer)
    trainer = SACTrainer(
        agent1=therapist_wrapper,
        agent2=patient_agent,
        environment=environment,
        payoff_calculator=payoff_calculator,
        episodes_per_training=episodes,
        steps_per_episode=60,
        evaluation_frequency=50,
        save_frequency=200,
        training_frequency=1,
    )

    # Add comprehensive logging (YOUR system)
    logging_wrapper = LoggingTrainerWrapper(trainer, experiment_name)

    # Train with visible evolution
    print(f"🚀 Starting visible therapeutic training...")
    print(f"   Episodes: {episodes}")
    print(f"   You'll see strategy evolution every {evolution_frequency} episodes")
    print(f"   Watch the console for strategy performance updates!")
    print("-" * 80)

    results = logging_wrapper.train_with_logging(f"./visible_models/{experiment_name}")

    # Final analysis
    print("\n" + "=" * 80)
    print("VISIBLE THERAPEUTIC TRAINING COMPLETED!")
    print("=" * 80)

    # Show final strategy performance
    visible_therapist._show_strategies()

    # Calculate results
    final_trust = patient_state.get_trust_level()
    final_warmth = (visible_therapist.last_patient_action + 1) / 2

    trust_improvement = final_trust - (-0.4)
    warmth_improvement = final_warmth - 0.2  # Estimated starting warmth

    therapeutic_success = final_trust > 0.4 and final_warmth > 0.6

    print(f"\n🎯 THERAPEUTIC RESULTS:")
    print(f"   Patient trust: -0.4 → {final_trust:.3f} (Δ{trust_improvement:+.3f})")
    print(f"   Patient warmth: ~0.2 → {final_warmth:.3f} (Δ{warmth_improvement:+.3f})")
    print(
        f"   Therapeutic success: {'✅ YES' if therapeutic_success else '📈 Progress'}"
    )
    print(f"   Therapy sessions: {environment.therapy_sessions_completed}")
    print(f"   Strategy generations: {visible_therapist.generation_count}")

    if therapeutic_success:
        print(f"\n🎉 SUCCESS! The visible evolution found strategies that work!")
    else:
        print(f"\n📈 Good progress! Evolution is learning better approaches.")

    print(f"\n📁 Generated Files:")
    print(f"   • Training logs: {results['comprehensive_logs']}")
    print(f"   • Model checkpoints: ./visible_models/{experiment_name}/")

    return {
        "results": results,
        "visible_therapist": visible_therapist,
        "patient_agent": patient_agent,
        "environment": environment,
        "therapeutic_success": therapeutic_success,
        "final_trust": final_trust,
        "final_warmth": final_warmth,
    }


def main():
    """Main function with different experiment options."""

    print("🧬 VISIBLE THERAPEUTIC EVOLUTION + YOUR SAC SYSTEM")
    print("=" * 60)
    print("This combines visible strategy evolution with your real:")
    print("• SAC machine learning")
    print("• Gaussian payoff matrix")
    print("• Comprehensive logging")
    print("• Model checkpoints")
    print("=" * 60)

    # Experiment configurations
    EXPERIMENTS = {
        "quick": {
            "episodes": 200,
            "evolution_frequency": 40,
            "population_size": 6,
            "alpha": 4.0,
            "beta": 10.0,
            "description": "Quick test (8-12 minutes)",
        },
        "standard": {
            "episodes": 400,
            "evolution_frequency": 75,
            "population_size": 8,
            "alpha": 4.0,
            "beta": 10.0,
            "description": "Standard experiment (20-25 minutes)",
        },
        "intensive": {
            "episodes": 600,
            "evolution_frequency": 100,
            "population_size": 12,
            "alpha": 6.0,
            "beta": 12.0,
            "description": "Intensive training (35-45 minutes)",
        },
    }

    # Select experiment
    experiment_type = "intensive"  # Change this to run different experiments

    config = EXPERIMENTS[experiment_type]
    experiment_name = f"visible_{experiment_type}_therapy"

    print(f"Running: {experiment_type.upper()} experiment")
    print(f"Description: {config['description']}")
    print(f"Episodes: {config['episodes']}")
    print(f"Population: {config['population_size']} evolving strategies")
    print(f"Gaussian params: α={config['alpha']}, β={config['beta']}")
    print(f"Evolution frequency: every {config['evolution_frequency']} episodes")
    print("-" * 60)

    try:
        results = run_visible_therapeutic_experiment(
            experiment_name=experiment_name,
            episodes=config["episodes"],
            evolution_frequency=config["evolution_frequency"],
            population_size=config["population_size"],
            alpha=config["alpha"],
            beta=config["beta"],
        )

        print(f"\n✨ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"You can now see exactly how population evolution works")
        print(f"with your real SAC training and Gaussian payoff system!")

        return results

    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
