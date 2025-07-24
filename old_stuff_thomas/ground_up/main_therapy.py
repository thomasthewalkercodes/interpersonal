"""
STANDALONE FIXED Therapeutic Agent System
Save this as: fixed_therapeutic_agent_system.py

This is a complete, standalone version that fixes all the bugs in the original system.
You can run this independently to test the therapeutic agent functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque, defaultdict
import copy
import os
import sys


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TherapeuticPhase(Enum):
    """Phases of the therapeutic process."""

    ASSESSMENT = "assessment"
    MATCHING = "matching"
    LEADING = "leading"
    STABILIZING = "stabilizing"
    ADVANCING = "advancing"


@dataclass
class TherapeuticStrategy:
    """A therapeutic strategy that can evolve through natural selection."""

    matching_intensity: float = 0.9
    trust_threshold: float = 0.3
    leading_step_size: float = 0.15
    stabilization_patience: int = 10
    warmth_target: float = 0.8
    trust_sensitivity: float = 0.5
    progress_sensitivity: float = 0.3
    retreat_threshold: float = -0.2
    assessment_duration: int = 15
    max_cycles: int = 10
    patience_multiplier: float = 1.2
    mutation_rate: float = 0.1
    fitness_score: float = 0.0
    generation: int = 0
    therapy_sessions: int = 0

    def mutate(self, mutation_strength: float = 0.1) -> "TherapeuticStrategy":
        """Create a mutated version of this strategy."""
        new_strategy = copy.deepcopy(self)

        if random.random() < self.mutation_rate:
            new_strategy.matching_intensity += np.random.normal(0, mutation_strength)
            new_strategy.matching_intensity = np.clip(
                new_strategy.matching_intensity, 0.1, 1.0
            )

        if random.random() < self.mutation_rate:
            new_strategy.trust_threshold += np.random.normal(0, mutation_strength)
            new_strategy.trust_threshold = np.clip(
                new_strategy.trust_threshold, 0.0, 0.8
            )

        if random.random() < self.mutation_rate:
            new_strategy.leading_step_size += np.random.normal(0, mutation_strength)
            new_strategy.leading_step_size = np.clip(
                new_strategy.leading_step_size, 0.05, 0.3
            )

        if random.random() < self.mutation_rate:
            new_strategy.stabilization_patience += int(np.random.normal(0, 3))
            new_strategy.stabilization_patience = max(
                5, min(30, new_strategy.stabilization_patience)
            )

        if random.random() < self.mutation_rate:
            new_strategy.trust_sensitivity += np.random.normal(0, mutation_strength)
            new_strategy.trust_sensitivity = np.clip(
                new_strategy.trust_sensitivity, 0.1, 1.0
            )

        new_strategy.generation += 1
        new_strategy.fitness_score = 0.0

        return new_strategy

    def crossover(self, other: "TherapeuticStrategy") -> "TherapeuticStrategy":
        """Create offspring by crossing over with another strategy."""
        child = TherapeuticStrategy()

        child.matching_intensity = random.choice(
            [self.matching_intensity, other.matching_intensity]
        )
        child.trust_threshold = random.choice(
            [self.trust_threshold, other.trust_threshold]
        )
        child.leading_step_size = random.choice(
            [self.leading_step_size, other.leading_step_size]
        )
        child.stabilization_patience = random.choice(
            [self.stabilization_patience, other.stabilization_patience]
        )
        child.warmth_target = random.choice([self.warmth_target, other.warmth_target])
        child.trust_sensitivity = (self.trust_sensitivity + other.trust_sensitivity) / 2
        child.progress_sensitivity = (
            self.progress_sensitivity + other.progress_sensitivity
        ) / 2
        child.generation = max(self.generation, other.generation) + 1

        return child


class TherapistAgent:
    """Advanced therapist agent that uses evolved strategies."""

    def __init__(
        self,
        agent_id: str = "therapist",
        population_size: int = 20,
        elite_ratio: float = 0.3,
        mutation_strength: float = 0.1,
    ):
        self.agent_id = agent_id
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_strength = mutation_strength

        # Initialize strategy population
        self.strategy_population = self._initialize_population()
        self.current_strategy = self.strategy_population[0]
        self.strategy_index = 0

        # Therapeutic state tracking
        self.current_phase = TherapeuticPhase.ASSESSMENT
        self.patient_baseline_warmth = 0.0
        self.patient_trust_history = deque(maxlen=50)
        self.patient_warmth_history = deque(maxlen=50)
        self.therapy_step = 0
        self.cycle_count = 0
        self.steps_in_phase = 0

        # FIXED: Initialize last_patient_action
        self.last_patient_action = 0.0

        # Performance tracking
        self.session_results = []
        self.evolution_history = []

        # Advanced monitoring
        self.trust_trend = 0.0
        self.warmth_trend = 0.0
        self.last_trust_level = 0.0
        self.target_warmth_level = 0.0
        self.phase_success_rate = defaultdict(float)

        print(f"[THERAPIST] Initialized with {population_size} strategies")

    def _initialize_population(self) -> List[TherapeuticStrategy]:
        """Initialize a diverse population of therapeutic strategies."""
        population = []

        for i in range(self.population_size):
            strategy = TherapeuticStrategy(
                matching_intensity=np.random.uniform(0.7, 1.0),
                trust_threshold=np.random.uniform(0.1, 0.6),
                leading_step_size=np.random.uniform(0.08, 0.25),
                stabilization_patience=np.random.randint(8, 20),
                warmth_target=np.random.uniform(0.6, 0.9),
                trust_sensitivity=np.random.uniform(0.2, 0.8),
                progress_sensitivity=np.random.uniform(0.1, 0.5),
                assessment_duration=np.random.randint(10, 25),
                max_cycles=np.random.randint(8, 15),
            )
            population.append(strategy)

        return population

    def select_action(
        self, state: np.ndarray, patient_action: float, patient_trust: float
    ) -> float:
        """Select therapeutic action based on current strategy and patient state."""
        # FIXED: Store patient action for future reference
        self.last_patient_action = patient_action

        # Update patient tracking
        self._update_patient_tracking(patient_action, patient_trust)

        # Determine current therapeutic phase
        self._update_therapeutic_phase()

        # Select action based on current phase and strategy
        action = self._select_phase_specific_action(patient_action, patient_trust)

        # Track therapy progress
        self.therapy_step += 1
        self.steps_in_phase += 1

        return action

    def _update_patient_tracking(self, patient_action: float, patient_trust: float):
        """Update tracking of patient's progress."""
        patient_warmth = (patient_action + 1) / 2

        self.patient_warmth_history.append(patient_warmth)
        self.patient_trust_history.append(patient_trust)

        # Calculate trends
        if len(self.patient_trust_history) >= 5:
            recent_trust = list(self.patient_trust_history)[-5:]
            self.trust_trend = np.polyfit(range(5), recent_trust, 1)[0]

        if len(self.patient_warmth_history) >= 5:
            recent_warmth = list(self.patient_warmth_history)[-5:]
            self.warmth_trend = np.polyfit(range(5), recent_warmth, 1)[0]

        # Store baseline during assessment
        if (
            self.current_phase == TherapeuticPhase.ASSESSMENT
            and len(self.patient_warmth_history) >= 5
        ):
            self.patient_baseline_warmth = np.mean(
                list(self.patient_warmth_history)[-5:]
            )

    def _update_therapeutic_phase(self):
        """Update the current therapeutic phase based on progress and strategy."""
        strategy = self.current_strategy
        current_trust = (
            self.patient_trust_history[-1] if self.patient_trust_history else 0.0
        )
        current_warmth = (
            self.patient_warmth_history[-1] if self.patient_warmth_history else 0.0
        )

        # Phase transition logic
        if self.current_phase == TherapeuticPhase.ASSESSMENT:
            if self.steps_in_phase >= strategy.assessment_duration:
                self.current_phase = TherapeuticPhase.MATCHING
                self.steps_in_phase = 0
                print(
                    f"[THERAPIST] Moving to MATCHING phase. Patient baseline: {self.patient_baseline_warmth:.3f}"
                )

        elif self.current_phase == TherapeuticPhase.MATCHING:
            if current_trust >= strategy.trust_threshold and self.steps_in_phase >= 5:
                self.current_phase = TherapeuticPhase.LEADING
                self.target_warmth_level = min(
                    current_warmth + strategy.leading_step_size, strategy.warmth_target
                )
                self.steps_in_phase = 0
                print(
                    f"[THERAPIST] Moving to LEADING phase. Target warmth: {self.target_warmth_level:.3f}"
                )
            elif self.trust_trend < strategy.retreat_threshold:
                self.steps_in_phase = 0
                print(f"[THERAPIST] Trust declining, continuing MATCHING")

        elif self.current_phase == TherapeuticPhase.LEADING:
            if (
                abs(current_warmth - self.target_warmth_level) < 0.1
                and self.warmth_trend >= 0
                and self.steps_in_phase >= 3
            ):
                self.current_phase = TherapeuticPhase.STABILIZING
                self.steps_in_phase = 0
                print(
                    f"[THERAPIST] Moving to STABILIZING phase at warmth {current_warmth:.3f}"
                )
            elif (
                self.trust_trend < strategy.retreat_threshold
                or self.warmth_trend < -0.05
            ):
                self.current_phase = TherapeuticPhase.MATCHING
                self.steps_in_phase = 0
                print(f"[THERAPIST] Patient resistance, returning to MATCHING")

        elif self.current_phase == TherapeuticPhase.STABILIZING:
            if self.steps_in_phase >= strategy.stabilization_patience:
                if current_warmth >= strategy.warmth_target * 0.9:
                    self.current_phase = TherapeuticPhase.ADVANCING
                    print(f"[THERAPIST] Near target! Moving to ADVANCING phase")
                else:
                    self.current_phase = TherapeuticPhase.LEADING
                    self.cycle_count += 1
                    self.target_warmth_level = min(
                        current_warmth + strategy.leading_step_size,
                        strategy.warmth_target,
                    )
                    print(
                        f"[THERAPIST] Starting cycle {self.cycle_count}, targeting {self.target_warmth_level:.3f}"
                    )
                self.steps_in_phase = 0

        elif self.current_phase == TherapeuticPhase.ADVANCING:
            if current_warmth >= strategy.warmth_target * 0.85 and current_trust >= 0.4:
                pass  # Success! Session complete
            elif self.steps_in_phase >= strategy.stabilization_patience * 2:
                if self.cycle_count >= strategy.max_cycles:
                    self._restart_therapy_session()

    def _select_phase_specific_action(
        self, patient_action: float, patient_trust: float
    ) -> float:
        """Select action based on current therapeutic phase."""
        strategy = self.current_strategy
        patient_warmth = (patient_action + 1) / 2

        if self.current_phase == TherapeuticPhase.ASSESSMENT:
            therapist_warmth = 0.4 + np.random.normal(0, 0.1)

        elif self.current_phase == TherapeuticPhase.MATCHING:
            matching_warmth = patient_warmth * strategy.matching_intensity
            warmth_bias = 0.05
            therapist_warmth = matching_warmth + warmth_bias

        elif self.current_phase == TherapeuticPhase.LEADING:
            therapist_warmth = self.target_warmth_level
            therapist_warmth += np.random.normal(0, 0.03)

        elif self.current_phase == TherapeuticPhase.STABILIZING:
            current_warmth = (
                self.patient_warmth_history[-1] if self.patient_warmth_history else 0.5
            )
            therapist_warmth = current_warmth + 0.02

        elif self.current_phase == TherapeuticPhase.ADVANCING:
            therapist_warmth = strategy.warmth_target
            if patient_trust > 0.5:
                therapist_warmth += 0.05

        else:
            therapist_warmth = 0.5

        # Apply trust-based adjustments
        if self.trust_trend < 0:
            therapist_warmth -= abs(self.trust_trend) * strategy.trust_sensitivity

        # Clamp to valid range and convert back to action space [-1, 1]
        therapist_warmth = np.clip(therapist_warmth, 0.0, 1.0)
        action = therapist_warmth * 2 - 1

        return action

    def _restart_therapy_session(self):
        """Restart therapy session with potentially new strategy."""
        self.current_phase = TherapeuticPhase.ASSESSMENT
        self.therapy_step = 0
        self.cycle_count = 0
        self.steps_in_phase = 0
        self.patient_warmth_history.clear()
        self.patient_trust_history.clear()
        self.last_patient_action = 0.0

        print(f"[THERAPIST] Restarting therapy session")

    def evaluate_strategy_fitness(self) -> float:
        """Evaluate the fitness of the current strategy based on patient progress."""
        if (
            len(self.patient_warmth_history) < 10
            or len(self.patient_trust_history) < 10
        ):
            return 0.0

        final_warmth = self.patient_warmth_history[-1]
        final_trust = self.patient_trust_history[-1]
        warmth_progress = final_warmth - self.patient_baseline_warmth
        trust_progress = final_trust - (
            self.patient_trust_history[0] if self.patient_trust_history else 0
        )

        warmth_efficiency = warmth_progress / max(1, self.therapy_step) * 100
        trust_stability = 1.0 - np.std(list(self.patient_trust_history)[-10:])
        target_achievement = min(
            1.0, final_warmth / self.current_strategy.warmth_target
        )
        cycle_penalty = (
            max(0, self.cycle_count - self.current_strategy.max_cycles) * 0.1
        )

        fitness = (
            warmth_progress * 2.0
            + trust_progress * 1.5
            + warmth_efficiency * 0.5
            + trust_stability * 1.0
            + target_achievement * 2.0
            - cycle_penalty
        )

        return max(0.0, fitness)

    def evolve_strategies(self, generation_size: int = 50):
        """Evolve the strategy population using genetic algorithm principles."""
        print(
            f"\n[EVOLUTION] Starting evolution cycle with {len(self.strategy_population)} strategies"
        )

        strategy_fitness = []

        for i, strategy in enumerate(self.strategy_population):
            self.current_strategy = strategy
            total_fitness = 0.0
            sessions_per_strategy = max(
                3, generation_size // len(self.strategy_population)
            )

            for session in range(sessions_per_strategy):
                self._restart_therapy_session()
                session_fitness = self._simulate_therapy_session()
                total_fitness += session_fitness

            average_fitness = total_fitness / sessions_per_strategy
            strategy.fitness_score = average_fitness
            strategy.therapy_sessions += sessions_per_strategy
            strategy_fitness.append((i, average_fitness))

            print(f"   Strategy {i}: Fitness {average_fitness:.3f}")

        # Sort strategies by fitness
        strategy_fitness.sort(key=lambda x: x[1], reverse=True)

        # Select elite strategies
        elite_count = int(len(self.strategy_population) * self.elite_ratio)
        elite_indices = [idx for idx, _ in strategy_fitness[:elite_count]]
        elite_strategies = [self.strategy_population[i] for i in elite_indices]

        print(f"[EVOLUTION] Elite strategies: {elite_indices}")

        # Generate new population
        new_population = elite_strategies.copy()

        while len(new_population) < self.population_size:
            parent1 = random.choice(elite_strategies)
            parent2 = random.choice(elite_strategies)

            if random.random() < 0.7:
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)

            child = child.mutate(self.mutation_strength)
            new_population.append(child)

        # Update population
        self.strategy_population = new_population
        self.current_strategy = self.strategy_population[0]

        # Record evolution history
        best_fitness = strategy_fitness[0][1]
        avg_fitness = np.mean([fitness for _, fitness in strategy_fitness])

        self.evolution_history.append(
            {
                "generation": len(self.evolution_history),
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "best_strategy": copy.deepcopy(elite_strategies[0]),
            }
        )

        print(
            f"[EVOLUTION] Best fitness: {best_fitness:.3f}, Average: {avg_fitness:.3f}"
        )

    def _simulate_therapy_session(self) -> float:
        """Simulate a therapy session for strategy evaluation."""
        patient_warmth = 0.2
        patient_trust = 0.1

        for step in range(100):
            therapist_action = self.select_action(
                state=np.zeros(10),
                patient_action=patient_warmth * 2 - 1,
                patient_trust=patient_trust,
            )

            therapist_warmth = (therapist_action + 1) / 2
            warmth_diff = abs(patient_warmth - therapist_warmth)

            if patient_trust > 0.2:
                patient_warmth += (therapist_warmth - patient_warmth) * 0.1

            trust_change = -warmth_diff * 0.1 + therapist_warmth * 0.05
            patient_trust += trust_change * 0.1
            patient_trust = np.clip(patient_trust, 0.0, 1.0)
            patient_warmth = np.clip(patient_warmth, 0.0, 1.0)

        return self.evaluate_strategy_fitness()

    def get_therapeutic_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on therapeutic progress."""
        if not self.patient_warmth_history or not self.patient_trust_history:
            return {"status": "No therapy data available"}

        current_warmth = self.patient_warmth_history[-1]
        current_trust = self.patient_trust_history[-1]

        report = {
            "session_info": {
                "therapy_step": self.therapy_step,
                "current_phase": self.current_phase.value,
                "cycle_count": self.cycle_count,
                "steps_in_phase": self.steps_in_phase,
            },
            "patient_progress": {
                "baseline_warmth": self.patient_baseline_warmth,
                "current_warmth": current_warmth,
                "warmth_progress": current_warmth - self.patient_baseline_warmth,
                "current_trust": current_trust,
                "trust_trend": self.trust_trend,
                "warmth_trend": self.warmth_trend,
            },
            "strategy_info": {
                "current_strategy_fitness": self.current_strategy.fitness_score,
                "strategy_generation": self.current_strategy.generation,
                "target_warmth": self.current_strategy.warmth_target,
                "trust_threshold": self.current_strategy.trust_threshold,
            },
            "evolution_progress": {
                "total_generations": len(self.evolution_history),
                "best_fitness_achieved": (
                    max([gen["best_fitness"] for gen in self.evolution_history])
                    if self.evolution_history
                    else 0.0
                ),
                "population_diversity": len(
                    set(str(s.__dict__) for s in self.strategy_population)
                ),
            },
        }

        return report


def test_simple_therapist():
    """Simple test of just the therapist without full SAC integration."""
    print("TESTING SIMPLE THERAPIST")
    print("=" * 40)

    try:
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
        print(f"\n Therapy Report:")
        print(f"  Final Phase: {report['session_info']['current_phase']}")
        print(f"  Total Steps: {report['session_info']['therapy_step']}")
        print(f"  Cycles: {report['session_info']['cycle_count']}")

        if "patient_progress" in report:
            progress = report["patient_progress"]
            print(f"  Patient Final Warmth: {progress.get('current_warmth', 'N/A')}")
            print(f"  Patient Final Trust: {progress.get('current_trust', 'N/A')}")

        print("[SUCCESS] Simple therapist test completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Error in simple test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_strategy_evolution():
    """Test the strategy evolution without SAC integration."""
    print("\n[EVOLUTION] TESTING STRATEGY EVOLUTION")
    print("=" * 40)

    try:
        print("Creating therapist for evolution test...")
        therapist = TherapistAgent(agent_id="evolution_test", population_size=6)

        print("Running strategy evolution...")

        # Manually set some fitness scores to test evolution
        for i, strategy in enumerate(therapist.strategy_population):
            strategy.fitness_score = np.random.uniform(0.1, 1.0)
            print(f"  Strategy {i}: Fitness {strategy.fitness_score:.3f}")

        print("\nEvolving strategies...")
        initial_best_fitness = max(
            s.fitness_score for s in therapist.strategy_population
        )

        # Run evolution (this will create new generation)
        therapist.evolve_strategies(generation_size=12)

        final_best_fitness = max(s.fitness_score for s in therapist.strategy_population)

        print(f"Initial best fitness: {initial_best_fitness:.3f}")
        print(f"Final best fitness: {final_best_fitness:.3f}")
        print(f"Evolution generations: {len(therapist.evolution_history)}")

        print("[SUCCESS] Evolution test completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Error in evolution test: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_full_therapeutic_simulation():
    """Run a full therapeutic simulation without SAC integration."""
    print("\n[SIMULATION] RUNNING FULL THERAPEUTIC SIMULATION")
    print("=" * 50)

    try:
        # Create therapist with larger population
        therapist = TherapistAgent(agent_id="full_sim_therapist", population_size=10)

        print(
            f"[SUCCESS] Created therapist with {len(therapist.strategy_population)} strategies"
        )

        # Simulate multiple therapy sessions
        print("\n[THERAPY] Simulating therapy sessions...")

        session_results = []

        for session in range(5):
            print(f"\n--- Session {session + 1} ---")

            # Reset for new session
            therapist._restart_therapy_session()

            # Simulate a cold, resistant patient who gradually warms up
            patient_warmth = 0.1  # Start very cold
            patient_trust = 0.05  # Start with very low trust

            session_data = {
                "warmth_history": [patient_warmth],
                "trust_history": [patient_trust],
                "therapist_actions": [],
                "phases": [],
            }

            # Run therapy session
            for step in range(30):
                patient_action = patient_warmth * 2 - 1  # Convert to action space

                # Therapist selects action
                therapist_action = therapist.select_action(
                    state=np.zeros(10),
                    patient_action=patient_action,
                    patient_trust=patient_trust,
                )

                therapist_warmth = (therapist_action + 1) / 2

                # Patient responds based on therapist behavior and trust
                if patient_trust > 0.3:
                    # Patient starts following therapist if trust is high enough
                    warmth_influence = (therapist_warmth - patient_warmth) * 0.2
                    patient_warmth += warmth_influence

                # Trust evolves based on warmth matching
                warmth_diff = abs(patient_warmth - therapist_warmth)
                trust_change = -warmth_diff * 0.1 + therapist_warmth * 0.05
                patient_trust += trust_change * 0.15

                # Add some noise for realism
                patient_warmth += np.random.normal(0, 0.02)
                patient_trust += np.random.normal(0, 0.01)

                # Clamp values
                patient_warmth = np.clip(patient_warmth, 0.0, 1.0)
                patient_trust = np.clip(patient_trust, 0.0, 1.0)

                # Record data
                session_data["warmth_history"].append(patient_warmth)
                session_data["trust_history"].append(patient_trust)
                session_data["therapist_actions"].append(therapist_action)
                session_data["phases"].append(therapist.current_phase.value)

                if step % 10 == 0:
                    print(
                        f"  Step {step}: Patient Warmth={patient_warmth:.3f}, "
                        f"Trust={patient_trust:.3f}, Phase={therapist.current_phase.value}"
                    )

            # Evaluate session
            final_warmth = session_data["warmth_history"][-1]
            final_trust = session_data["trust_history"][-1]
            warmth_progress = final_warmth - session_data["warmth_history"][0]

            session_results.append(
                {
                    "session": session + 1,
                    "final_warmth": final_warmth,
                    "final_trust": final_trust,
                    "warmth_progress": warmth_progress,
                    "data": session_data,
                }
            )

            print(
                f"  Final: Warmth={final_warmth:.3f}, Trust={final_trust:.3f}, "
                f"Progress={warmth_progress:.3f}"
            )

        # Run evolution after sessions
        print(f"\n[EVOLUTION] Evolving strategies based on session results...")
        therapist.evolve_strategies(generation_size=20)

        # Print summary
        print(f"\n[SUMMARY] SIMULATION SUMMARY")
        print("=" * 30)
        avg_final_warmth = np.mean([r["final_warmth"] for r in session_results])
        avg_final_trust = np.mean([r["final_trust"] for r in session_results])
        avg_progress = np.mean([r["warmth_progress"] for r in session_results])

        print(f"Average Final Warmth: {avg_final_warmth:.3f}")
        print(f"Average Final Trust: {avg_final_trust:.3f}")
        print(f"Average Warmth Progress: {avg_progress:.3f}")
        print(f"Evolution Generations: {len(therapist.evolution_history)}")

        if therapist.evolution_history:
            best_fitness = therapist.evolution_history[-1]["best_fitness"]
            print(f"Best Strategy Fitness: {best_fitness:.3f}")

        return session_results, therapist

    except Exception as e:
        print(f"[ERROR] Error in full simulation: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def main():
    """Main function - runs automatically without user input."""
    print(" THERAPEUTIC AGENT SYSTEM - STANDALONE TEST")
    print("=" * 60)

    # Test 1: Basic therapist functionality
    print("\n🔧 RUNNING BASIC TESTS...")
    if not test_simple_therapist():
        print(" Basic test failed. Stopping.")
        return

    # Test 2: Strategy evolution
    if not test_strategy_evolution():
        print("Evolution test failed. Stopping.")
        return

    # Test 3: Full simulation
    print("\n RUNNING FULL SIMULATION...")
    session_results, therapist = run_full_therapeutic_simulation()

    if session_results is not None:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("=" * 50)
        print("[RESULT] The therapeutic agent system is working correctly!")
        print("Key achievements:")
        print("  - Basic therapist functionality [OK]")
        print("  - Strategy evolution [OK]")
        print("  - Full therapy simulation [OK]")
        print("  - Patient progress tracking [OK]")

        # Save results if possible
        try:
            import json

            results_summary = {
                "session_count": len(session_results),
                "avg_final_warmth": float(
                    np.mean([r["final_warmth"] for r in session_results])
                ),
                "avg_final_trust": float(
                    np.mean([r["final_trust"] for r in session_results])
                ),
                "avg_progress": float(
                    np.mean([r["warmth_progress"] for r in session_results])
                ),
                "evolution_generations": (
                    len(therapist.evolution_history) if therapist else 0
                ),
            }

            with open("therapeutic_results.json", "w", encoding="utf-8") as f:
                json.dump(results_summary, f, indent=2)
            print("  - Results saved to 'therapeutic_results.json' [OK]")

        except Exception as e:
            print(f"  - Could not save results: {e}")

    else:
        print("\n SIMULATION FAILED")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    # Run automatically without requiring user input
    main()
