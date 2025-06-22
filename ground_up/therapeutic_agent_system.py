"""
Fixed Therapeutic Agent System - Windows Compatible
Removes Unicode characters that cause encoding issues on Windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import deque, defaultdict
import copy


class TherapeuticPhase(Enum):
    """Phases of the therapeutic process."""

    ASSESSMENT = "assessment"  # Understanding patient's baseline
    MATCHING = "matching"  # Matching patient's behavior to build trust
    LEADING = "leading"  # Gradually becoming warmer to guide patient
    STABILIZING = "stabilizing"  # Consolidating gains at current level
    ADVANCING = "advancing"  # Moving to next warmth level


@dataclass
class TherapeuticStrategy:
    """
    A therapeutic strategy that can evolve through natural selection.
    Represents the 'genes' of how a therapist approaches treatment.
    """

    # Core strategy parameters
    matching_intensity: float = 0.9  # How closely to match patient (0-1)
    trust_threshold: float = 0.3  # Trust level needed before leading
    leading_step_size: float = 0.15  # How much warmer to become when leading
    stabilization_patience: int = 10  # Steps to wait at each level
    warmth_target: float = 0.8  # Ultimate warmth goal

    # Adaptive parameters
    trust_sensitivity: float = 0.5  # How much trust changes affect strategy
    progress_sensitivity: float = 0.3  # How much progress rate affects strategy
    retreat_threshold: float = -0.2  # When to back off (negative trust change)

    # Timing parameters
    assessment_duration: int = 15  # Steps to assess patient baseline
    max_cycles: int = 10  # Maximum therapy cycles before restart
    patience_multiplier: float = 1.2  # How patience grows with failed attempts

    # Evolution parameters (for genetic algorithm)
    mutation_rate: float = 0.1
    fitness_score: float = 0.0
    generation: int = 0
    therapy_sessions: int = 0

    def mutate(self, mutation_strength: float = 0.1) -> "TherapeuticStrategy":
        """Create a mutated version of this strategy."""
        new_strategy = copy.deepcopy(self)

        # Mutate each parameter with some probability
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
        new_strategy.fitness_score = 0.0  # Reset fitness for new generation

        return new_strategy

    def crossover(self, other: "TherapeuticStrategy") -> "TherapeuticStrategy":
        """Create offspring by crossing over with another strategy."""
        child = TherapeuticStrategy()

        # Randomly inherit parameters from either parent
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

        # Average some parameters
        child.trust_sensitivity = (self.trust_sensitivity + other.trust_sensitivity) / 2
        child.progress_sensitivity = (
            self.progress_sensitivity + other.progress_sensitivity
        ) / 2

        child.generation = max(self.generation, other.generation) + 1
        return child


class TherapistAgent:
    """
    Advanced therapist agent that uses evolved strategies to guide patients
    toward warm-warm cooperation through natural selection of therapeutic approaches.
    """

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

        # Create diverse initial strategies
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
        """
        Select therapeutic action based on current strategy and patient state.

        Args:
            state: Therapist's internal state
            patient_action: Patient's most recent action
            patient_trust: Patient's current trust level

        Returns:
            Therapeutic action (warmth level)
        """
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
        # Convert action to warmth (assuming actions are in [-1, 1])
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
            # Move to leading when trust is sufficient
            if current_trust >= strategy.trust_threshold and self.steps_in_phase >= 5:
                self.current_phase = TherapeuticPhase.LEADING
                self.target_warmth_level = min(
                    current_warmth + strategy.leading_step_size, strategy.warmth_target
                )
                self.steps_in_phase = 0
                print(
                    f"[THERAPIST] Moving to LEADING phase. Target warmth: {self.target_warmth_level:.3f}"
                )

            # Retreat if trust is dropping significantly
            elif self.trust_trend < strategy.retreat_threshold:
                self.steps_in_phase = 0  # Stay in matching longer
                print(f"[THERAPIST] Trust declining, continuing MATCHING")

        elif self.current_phase == TherapeuticPhase.LEADING:
            # Move to stabilizing when patient starts following
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

            # Retreat if patient becomes more resistant
            elif (
                self.trust_trend < strategy.retreat_threshold
                or self.warmth_trend < -0.05
            ):
                self.current_phase = TherapeuticPhase.MATCHING
                self.steps_in_phase = 0
                print(f"[THERAPIST] Patient resistance, returning to MATCHING")

        elif self.current_phase == TherapeuticPhase.STABILIZING:
            if self.steps_in_phase >= strategy.stabilization_patience:
                # Check if we've reached the target
                if current_warmth >= strategy.warmth_target * 0.9:
                    self.current_phase = TherapeuticPhase.ADVANCING
                    print(f"[THERAPIST] Near target! Moving to ADVANCING phase")
                else:
                    # Start next cycle
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
            # Maintain warm behavior and monitor for stability
            if current_warmth >= strategy.warmth_target * 0.85 and current_trust >= 0.4:
                # Success! Session complete
                pass
            elif self.steps_in_phase >= strategy.stabilization_patience * 2:
                # Final push or restart
                if self.cycle_count >= strategy.max_cycles:
                    self._restart_therapy_session()

    def _select_phase_specific_action(
        self, patient_action: float, patient_trust: float
    ) -> float:
        """Select action based on current therapeutic phase."""
        strategy = self.current_strategy
        patient_warmth = (patient_action + 1) / 2

        if self.current_phase == TherapeuticPhase.ASSESSMENT:
            # Neutral behavior during assessment
            therapist_warmth = 0.4 + np.random.normal(0, 0.1)

        elif self.current_phase == TherapeuticPhase.MATCHING:
            # Match patient's behavior with slight warmth bias
            matching_warmth = patient_warmth * strategy.matching_intensity
            warmth_bias = 0.05  # Slight positive bias
            therapist_warmth = matching_warmth + warmth_bias

        elif self.current_phase == TherapeuticPhase.LEADING:
            # Move toward target warmth level
            therapist_warmth = self.target_warmth_level
            # Add slight variability to seem natural
            therapist_warmth += np.random.normal(0, 0.03)

        elif self.current_phase == TherapeuticPhase.STABILIZING:
            # Maintain current level with patience
            current_warmth = (
                self.patient_warmth_history[-1] if self.patient_warmth_history else 0.5
            )
            therapist_warmth = (
                current_warmth + 0.02
            )  # Slightly warmer to maintain leadership

        elif self.current_phase == TherapeuticPhase.ADVANCING:
            # Maintain warm, stable behavior
            therapist_warmth = strategy.warmth_target
            # Add trust-based adjustments
            if patient_trust > 0.5:
                therapist_warmth += 0.05  # Extra warmth for high trust

        else:
            therapist_warmth = 0.5  # Default neutral

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

        print(f"[THERAPIST] Restarting therapy session")

    def evaluate_strategy_fitness(self) -> float:
        """
        Evaluate the fitness of the current strategy based on patient progress.

        Returns:
            Fitness score (higher is better)
        """
        if (
            len(self.patient_warmth_history) < 10
            or len(self.patient_trust_history) < 10
        ):
            return 0.0

        # Calculate various success metrics
        final_warmth = self.patient_warmth_history[-1]
        final_trust = self.patient_trust_history[-1]
        warmth_progress = final_warmth - self.patient_baseline_warmth
        trust_progress = final_trust - (
            self.patient_trust_history[0] if self.patient_trust_history else 0
        )

        # Efficiency metrics
        warmth_efficiency = warmth_progress / max(1, self.therapy_step) * 100
        trust_stability = 1.0 - np.std(list(self.patient_trust_history)[-10:])

        # Goal achievement
        target_achievement = min(
            1.0, final_warmth / self.current_strategy.warmth_target
        )

        # Penalize excessive cycles
        cycle_penalty = (
            max(0, self.cycle_count - self.current_strategy.max_cycles) * 0.1
        )

        # Calculate composite fitness
        fitness = (
            warmth_progress * 2.0  # Primary goal
            + trust_progress * 1.5  # Trust building
            + warmth_efficiency * 0.5  # Efficiency
            + trust_stability * 1.0  # Stability
            + target_achievement * 2.0  # Goal achievement
            - cycle_penalty  # Penalty for inefficiency
        )

        return max(0.0, fitness)

    def evolve_strategies(self, generation_size: int = 50):
        """
        Evolve the strategy population using genetic algorithm principles.

        Args:
            generation_size: Number of therapy sessions to run per evolution cycle
        """
        print(
            f"\n[EVOLUTION] Starting evolution cycle with {len(self.strategy_population)} strategies"
        )

        # Evaluate all strategies
        strategy_fitness = []

        for i, strategy in enumerate(self.strategy_population):
            # Run therapy sessions with this strategy
            self.current_strategy = strategy

            # Simulate multiple sessions for robust evaluation
            total_fitness = 0.0
            sessions_per_strategy = max(
                3, generation_size // len(self.strategy_population)
            )

            for session in range(sessions_per_strategy):
                self._restart_therapy_session()
                # Simulate therapy session
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
        new_population = elite_strategies.copy()  # Keep elites

        while len(new_population) < self.population_size:
            # Select parents from elite strategies
            parent1 = random.choice(elite_strategies)
            parent2 = random.choice(elite_strategies)

            # Create offspring
            if random.random() < 0.7:  # Crossover probability
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)

            # Mutate
            child = child.mutate(self.mutation_strength)
            new_population.append(child)

        # Update population
        self.strategy_population = new_population
        self.current_strategy = self.strategy_population[0]  # Use best strategy

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
        """
        Simulate a therapy session for strategy evaluation.
        This is a simplified simulation - in practice, this would be integrated
        with your main SAC training loop.
        """
        # Simplified simulation of patient behavior
        patient_warmth = 0.2  # Start cold
        patient_trust = 0.1

        for step in range(100):  # Simulate 100 steps
            # Therapist action
            therapist_action = self.select_action(
                state=np.zeros(10),  # Dummy state
                patient_action=patient_warmth * 2 - 1,
                patient_trust=patient_trust,
            )

            therapist_warmth = (therapist_action + 1) / 2

            # Simplified patient response model
            warmth_diff = abs(patient_warmth - therapist_warmth)

            # Patient gradually matches therapist if trust is high enough
            if patient_trust > 0.2:
                patient_warmth += (therapist_warmth - patient_warmth) * 0.1

            # Trust evolves based on matching and therapist warmth
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


# Wrapper class for SAC integration
class TherapistSACWrapper:
    """
    Wrapper to make the TherapistAgent compatible with the SAC training system.
    """

    def __init__(self, therapist_agent: TherapistAgent, state_dim: int):
        self.therapist = therapist_agent
        self.state_dim = state_dim
        self.replay_buffer = []  # Dummy replay buffer
        self.last_patient_action = 0.0

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select action using therapist's strategy."""
        # Extract patient trust from state (assuming it's in the state vector)
        patient_trust = state[1] if len(state) > 1 else 0.0

        return self.therapist.select_action(
            state, self.last_patient_action, patient_trust
        )

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition (dummy implementation for compatibility)."""
        # Therapist doesn't learn from individual transitions like SAC
        # Instead, it evolves strategies based on overall session success
        pass

    def train_step(self):
        """Training step (dummy implementation)."""
        # Return dummy metrics for compatibility
        return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.1, "alpha_loss": 0.0}

    def save_model(self, filepath: str):
        """Save therapist strategies."""
        import json

        strategies_data = []
        for strategy in self.therapist.strategy_population:
            # Convert strategy to dictionary with all serializable fields
            strategy_dict = {
                "matching_intensity": float(strategy.matching_intensity),
                "trust_threshold": float(strategy.trust_threshold),
                "leading_step_size": float(strategy.leading_step_size),
                "stabilization_patience": int(strategy.stabilization_patience),
                "warmth_target": float(strategy.warmth_target),
                "trust_sensitivity": float(strategy.trust_sensitivity),
                "progress_sensitivity": float(strategy.progress_sensitivity),
                "retreat_threshold": float(strategy.retreat_threshold),
                "assessment_duration": int(strategy.assessment_duration),
                "max_cycles": int(strategy.max_cycles),
                "patience_multiplier": float(strategy.patience_multiplier),
                "mutation_rate": float(strategy.mutation_rate),
                "fitness_score": float(strategy.fitness_score),
                "generation": int(strategy.generation),
                "therapy_sessions": int(strategy.therapy_sessions),
            }
            strategies_data.append(strategy_dict)

        # Clean evolution history to be JSON serializable
        clean_evolution_history = []
        for entry in self.therapist.evolution_history:
            clean_entry = {
                "generation": int(entry["generation"]),
                "best_fitness": float(entry["best_fitness"]),
                "average_fitness": float(entry["average_fitness"]),
                # Convert best_strategy to dict instead of object
                "best_strategy": {
                    "matching_intensity": float(
                        entry["best_strategy"].matching_intensity
                    ),
                    "trust_threshold": float(entry["best_strategy"].trust_threshold),
                    "leading_step_size": float(
                        entry["best_strategy"].leading_step_size
                    ),
                    "stabilization_patience": int(
                        entry["best_strategy"].stabilization_patience
                    ),
                    "warmth_target": float(entry["best_strategy"].warmth_target),
                    "fitness_score": float(entry["best_strategy"].fitness_score),
                    "generation": int(entry["best_strategy"].generation),
                },
            }
            clean_evolution_history.append(clean_entry)

        save_data = {
            "strategies": strategies_data,
            "evolution_history": clean_evolution_history,
            "current_strategy_index": 0,
            "therapeutic_state": {
                "current_phase": self.therapist.current_phase.value,
                "therapy_step": int(self.therapist.therapy_step),
                "cycle_count": int(self.therapist.cycle_count),
                "patient_baseline_warmth": float(
                    self.therapist.patient_baseline_warmth
                ),
            },
        }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

    def load_model(self, filepath: str):
        """Load therapist strategies."""
        import json

        with open(filepath, "r") as f:
            save_data = json.load(f)

        # Reconstruct strategies from dictionaries
        strategies = []
        for strategy_data in save_data["strategies"]:
            strategy = TherapeuticStrategy(
                matching_intensity=strategy_data["matching_intensity"],
                trust_threshold=strategy_data["trust_threshold"],
                leading_step_size=strategy_data["leading_step_size"],
                stabilization_patience=strategy_data["stabilization_patience"],
                warmth_target=strategy_data["warmth_target"],
                trust_sensitivity=strategy_data.get("trust_sensitivity", 0.5),
                progress_sensitivity=strategy_data.get("progress_sensitivity", 0.3),
                retreat_threshold=strategy_data.get("retreat_threshold", -0.2),
                assessment_duration=strategy_data.get("assessment_duration", 15),
                max_cycles=strategy_data.get("max_cycles", 10),
                patience_multiplier=strategy_data.get("patience_multiplier", 1.2),
                mutation_rate=strategy_data.get("mutation_rate", 0.1),
                fitness_score=strategy_data["fitness_score"],
                generation=strategy_data["generation"],
                therapy_sessions=strategy_data.get("therapy_sessions", 0),
            )
            strategies.append(strategy)

        self.therapist.strategy_population = strategies
        self.therapist.evolution_history = save_data["evolution_history"]
        self.therapist.current_strategy = strategies[0]

        # Restore therapeutic state if available
        if "therapeutic_state" in save_data:
            state = save_data["therapeutic_state"]
            # Note: We can't directly restore the phase enum, but we could add logic here
            self.therapist.therapy_step = state.get("therapy_step", 0)
            self.therapist.cycle_count = state.get("cycle_count", 0)
            self.therapist.patient_baseline_warmth = state.get(
                "patient_baseline_warmth", 0.0
            )


def main():
    """Main function for testing the therapeutic agent system."""
    print("THERAPEUTIC AGENT SYSTEM - STANDALONE TEST")
    print("=" * 60)

    print("\n[TEST] Running basic tests...")

    # Test 1: Create therapist
    print("1. Creating therapist agent...")
    try:
        therapist = TherapistAgent(agent_id="test_therapist", population_size=5)
        print(
            f"   Created therapist with {len(therapist.strategy_population)} strategies"
        )
        print("   SUCCESS: Therapist creation")
    except Exception as e:
        print(f"   FAILED: Therapist creation - {e}")
        return False

    # Test 2: Test action selection
    print("\n2. Testing action selection...")
    try:
        dummy_state = np.zeros(10)
        patient_action = -0.5  # Cold patient
        patient_trust = 0.2

        action = therapist.select_action(dummy_state, patient_action, patient_trust)
        print(f"   Therapist action: {action:.3f}")
        print(f"   Current phase: {therapist.current_phase.value}")
        print("   SUCCESS: Action selection")
    except Exception as e:
        print(f"   FAILED: Action selection - {e}")
        return False

    # Test 3: Test strategy evolution
    print("\n3. Testing strategy evolution...")
    try:
        # Set some dummy fitness scores
        for i, strategy in enumerate(therapist.strategy_population):
            strategy.fitness_score = np.random.uniform(0.1, 1.0)

        therapist.evolve_strategies(generation_size=10)
        print(
            f"   Evolution completed. Generations: {len(therapist.evolution_history)}"
        )
        print("   SUCCESS: Strategy evolution")
    except Exception as e:
        print(f"   FAILED: Strategy evolution - {e}")
        return False

    # Test 4: Test therapeutic report
    print("\n4. Testing therapeutic report...")
    try:
        report = therapist.get_therapeutic_report()
        print(f"   Report keys: {list(report.keys())}")
        if "session_info" in report:
            print(f"   Current phase: {report['session_info']['current_phase']}")
        print("   SUCCESS: Therapeutic report")
    except Exception as e:
        print(f"   FAILED: Therapeutic report - {e}")
        return False

    # Test 5: Test SAC wrapper
    print("\n5. Testing SAC wrapper...")
    try:
        wrapper = TherapistSACWrapper(therapist, state_dim=10)
        state = np.random.randn(10)
        action = wrapper.select_action(state, training=True)
        print(f"   Wrapper action: {action:.3f}")
        print("   SUCCESS: SAC wrapper")
    except Exception as e:
        print(f"   FAILED: SAC wrapper - {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe therapeutic agent system is working correctly.")
    print("You can now integrate it with your main SAC training loop.")

    return True


if __name__ == "__main__":
    main()
