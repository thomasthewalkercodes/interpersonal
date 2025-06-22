"""
Therapeutic Agent System implementing natural selection for optimal therapy strategies.
The therapist agent evolves strategies to guide cold agents toward warm-warm cooperation.
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

                # Simulate therapy session (this would be integrated with your main training loop)
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

    def visualize_therapy_progress(self, save_path: str = "./therapy_analysis.png"):
        """Create visualization of therapy progress and evolution."""
        if not self.patient_warmth_history:
            print("No therapy data to visualize")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Patient warmth evolution
        axes[0, 0].plot(
            list(self.patient_warmth_history), label="Patient Warmth", linewidth=2
        )
        axes[0, 0].axhline(
            y=self.current_strategy.warmth_target,
            color="red",
            linestyle="--",
            label="Target",
        )
        axes[0, 0].set_title("Patient Warmth Evolution")
        axes[0, 0].set_xlabel("Therapy Step")
        axes[0, 0].set_ylabel("Warmth Level")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Trust evolution
        axes[0, 1].plot(
            list(self.patient_trust_history),
            label="Patient Trust",
            color="green",
            linewidth=2,
        )
        axes[0, 1].axhline(
            y=self.current_strategy.trust_threshold,
            color="orange",
            linestyle="--",
            label="Trust Threshold",
        )
        axes[0, 1].set_title("Trust Evolution")
        axes[0, 1].set_xlabel("Therapy Step")
        axes[0, 1].set_ylabel("Trust Level")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Evolution history
        if self.evolution_history:
            generations = [gen["generation"] for gen in self.evolution_history]
            best_fitness = [gen["best_fitness"] for gen in self.evolution_history]
            avg_fitness = [gen["average_fitness"] for gen in self.evolution_history]

            axes[0, 2].plot(
                generations, best_fitness, label="Best Fitness", linewidth=2
            )
            axes[0, 2].plot(
                generations, avg_fitness, label="Average Fitness", linewidth=2
            )
            axes[0, 2].set_title("Strategy Evolution")
            axes[0, 2].set_xlabel("Generation")
            axes[0, 2].set_ylabel("Fitness")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Phase progression
        # This would require tracking phase changes over time
        axes[1, 0].text(
            0.5,
            0.5,
            f"Current Phase:\n{self.current_phase.value.title()}\n\nCycle: {self.cycle_count}",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=14,
        )
        axes[1, 0].set_title("Therapeutic Phase")

        # 5. Strategy parameters
        strategy = self.current_strategy
        params_text = f"""Strategy Parameters:
Matching Intensity: {strategy.matching_intensity:.2f}
Trust Threshold: {strategy.trust_threshold:.2f}
Leading Step: {strategy.leading_step_size:.2f}
Patience: {strategy.stabilization_patience}
Target Warmth: {strategy.warmth_target:.2f}
Fitness: {strategy.fitness_score:.2f}"""

        axes[1, 1].text(
            0.1,
            0.9,
            params_text,
            ha="left",
            va="top",
            transform=axes[1, 1].transAxes,
            fontsize=10,
            fontfamily="monospace",
        )
        axes[1, 1].set_title("Current Strategy")
        axes[1, 1].axis("off")

        # 6. Progress summary
        if self.patient_warmth_history and self.patient_trust_history:
            progress_text = f"""Progress Summary:
Initial Warmth: {self.patient_baseline_warmth:.3f}
Current Warmth: {self.patient_warmth_history[-1]:.3f}
Warmth Gain: {self.patient_warmth_history[-1] - self.patient_baseline_warmth:.3f}

Current Trust: {self.patient_trust_history[-1]:.3f}
Trust Trend: {self.trust_trend:.4f}

Therapy Steps: {self.therapy_step}
Success Rate: {(self.patient_warmth_history[-1] / self.current_strategy.warmth_target * 100):.1f}%"""

            axes[1, 2].text(
                0.1,
                0.9,
                progress_text,
                ha="left",
                va="top",
                transform=axes[1, 2].transAxes,
                fontsize=10,
                fontfamily="monospace",
            )
            axes[1, 2].set_title("Session Summary")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[VISUALIZATION] Therapy progress saved to {save_path}")


# Integration functions
def create_therapist_patient_experiment(
    patient_config,
    therapist_population_size: int = 20,
    therapy_episodes: int = 500,
    evolution_frequency: int = 100,
):
    """
    Create an experiment with a therapist agent and a patient agent.

    Args:
        patient_config: Configuration for the patient agent
        therapist_population_size: Size of therapist strategy population
        therapy_episodes: Number of therapy episodes to run
        evolution_frequency: How often to evolve therapist strategies

    Returns:
        Tuple of (therapist, patient, environment, trainer)
    """
    from agent_state import InterpersonalAgentState
    from sac_algorithm import SACAgent
    from interaction_environment import InterpersonalEnvironment
    from gaussian_payoff_graph import calculate_warmth_payoff

    # Create therapist
    therapist = TherapistAgent(
        agent_id="therapist_agent", population_size=therapist_population_size
    )

    # Create patient agent
    patient_state = patient_config.create_initial_state()
    state_dim = patient_state.get_state_dimension()
    patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

    # Create therapist "agent" wrapper that integrates with SAC system
    therapist_sac_wrapper = TherapistSACWrapper(therapist, state_dim)

    # Create custom payoff calculator that encourages warm-warm outcomes
    payoff_calculator = TherapeuticPayoffCalculator()

    # Create environment
    environment = TherapeuticEnvironment(
        therapist=therapist,
        patient_state=patient_state,
        payoff_calculator=payoff_calculator,
        evolution_frequency=evolution_frequency,
    )

    return therapist, patient_agent, therapist_sac_wrapper, environment


class TherapistSACWrapper:
    """
    Wrapper to make the TherapistAgent compatible with the SAC training system.
    """

    def __init__(self, therapist_agent: TherapistAgent, state_dim: int):
        self.therapist = therapist_agent
        self.state_dim = state_dim
        self.replay_buffer = []  # Dummy replay buffer

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select action using therapist's strategy."""
        # Extract patient trust from state (assuming it's in the state vector)
        patient_trust = state[1] if len(state) > 1 else 0.0

        # Get last patient action (would need to be tracked separately)
        patient_action = getattr(self, "last_patient_action", 0.0)

        return self.therapist.select_action(state, patient_action, patient_trust)

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
            strategies_data.append(
                {
                    "matching_intensity": strategy.matching_intensity,
                    "trust_threshold": strategy.trust_threshold,
                    "leading_step_size": strategy.leading_step_size,
                    "stabilization_patience": strategy.stabilization_patience,
                    "warmth_target": strategy.warmth_target,
                    "fitness_score": strategy.fitness_score,
                    "generation": strategy.generation,
                }
            )

        save_data = {
            "strategies": strategies_data,
            "evolution_history": self.therapist.evolution_history,
            "current_strategy_index": 0,
        }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

    def load_model(self, filepath: str):
        """Load therapist strategies."""
        import json

        with open(filepath, "r") as f:
            save_data = json.load(f)

        # Reconstruct strategies
        strategies = []
        for strategy_data in save_data["strategies"]:
            strategy = TherapeuticStrategy(**strategy_data)
            strategies.append(strategy)

        self.therapist.strategy_population = strategies
        self.therapist.evolution_history = save_data["evolution_history"]
        self.therapist.current_strategy = strategies[0]


class TherapeuticPayoffCalculator:
    """
    Payoff calculator designed for therapeutic interactions.
    Rewards progress toward warm-warm cooperation.
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
        """Calculate payoffs with therapeutic objectives."""
        # Convert to warmth space
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Base payoff using Gaussian function
        from gaussian_payoff_graph import calculate_warmth_payoff

        base_therapist_payoff = calculate_warmth_payoff(
            therapist_warmth, patient_warmth, self.alpha, self.beta
        )
        base_patient_payoff = calculate_warmth_payoff(
            patient_warmth, therapist_warmth, self.alpha, self.beta
        )

        # Therapeutic modifications

        # 1. Reward therapist for patient progress toward warmth
        warmth_progress_bonus = (
            patient_warmth * 2.0
        )  # Strong incentive for patient warmth

        # 2. Reward mutual warmth highly
        mutual_warmth_bonus = min(therapist_warmth, patient_warmth) * 3.0

        # 3. Penalize therapist for being too cold (should model warmth)
        therapist_coldness_penalty = (1.0 - therapist_warmth) * 1.5

        # 4. Bonus for consistency in therapeutic direction
        # (This would require tracking over time - simplified here)
        consistency_bonus = 0.5 if therapist_warmth >= patient_warmth else 0.0

        # Calculate final payoffs
        therapist_payoff = (
            base_therapist_payoff
            + warmth_progress_bonus
            + mutual_warmth_bonus
            + consistency_bonus
            - therapist_coldness_penalty
        )

        # Patient gets standard payoff plus progress bonus
        patient_payoff = base_patient_payoff + mutual_warmth_bonus * 0.5

        return therapist_payoff, patient_payoff


class TherapeuticEnvironment:
    """
    Specialized environment for therapist-patient interactions with evolution.
    """

    def __init__(
        self,
        therapist: TherapistAgent,
        patient_state,
        payoff_calculator: TherapeuticPayoffCalculator,
        evolution_frequency: int = 100,
        max_steps_per_episode: int = 50,
    ):
        self.therapist = therapist
        self.patient_state = patient_state
        self.payoff_calculator = payoff_calculator
        self.evolution_frequency = evolution_frequency
        self.max_steps_per_episode = max_steps_per_episode

        self.current_step = 0
        self.episode_count = 0
        self.total_interactions = 0

        # Track therapeutic progress
        self.therapy_session_results = []

    def step(self, therapist_action: float, patient_action: float):
        """Execute one therapeutic interaction step."""
        # Store patient action for therapist
        self.therapist.therapist.last_patient_action = patient_action

        # Calculate payoffs
        therapist_payoff, patient_payoff = self.payoff_calculator.calculate_payoff(
            therapist_action, patient_action, "therapist", "patient"
        )

        # Update patient state
        self.patient_state.update_state(
            patient_action, therapist_action, patient_payoff
        )

        # Get next states
        patient_next_state = self.patient_state.get_state_vector()
        therapist_next_state = self._get_therapist_state()

        # Update step counters
        self.current_step += 1
        self.total_interactions += 1

        # Check for episode termination
        done = self._check_termination()

        # Evolve therapist strategies periodically
        if self.total_interactions % self.evolution_frequency == 0:
            print(
                f"\n[EVOLUTION] Evolving therapist strategies at interaction {self.total_interactions}"
            )
            self.therapist.evolve_strategies()

            # Generate progress report
            report = self.therapist.get_therapeutic_report()
            print(f"[THERAPY] Current progress: {report['patient_progress']}")

        return (
            therapist_next_state,
            patient_next_state,
            therapist_payoff,
            patient_payoff,
            done,
        )

    def reset(self):
        """Reset environment for new episode."""
        self.patient_state.reset_state()
        self.current_step = 0
        self.episode_count += 1

        # Reset therapist session if starting new therapy
        if self.episode_count % 10 == 1:  # New therapy session every 10 episodes
            self.therapist._restart_therapy_session()

        patient_initial_state = self.patient_state.get_state_vector()
        therapist_initial_state = self._get_therapist_state()

        return therapist_initial_state, patient_initial_state

    def _get_therapist_state(self):
        """Get therapist's state vector."""
        # Create state vector with therapeutic information
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

        # Create comprehensive state vector
        state = np.array(
            [
                patient_trust,
                patient_satisfaction,
                current_phase_encoding,
                self.therapist.trust_trend,
                self.therapist.warmth_trend,
                self.therapist.therapy_step / 100.0,  # Normalized step count
                self.therapist.cycle_count / 10.0,  # Normalized cycle count
                self.therapist.current_strategy.warmth_target,
                self.therapist.current_strategy.trust_threshold,
                len(self.therapist.patient_warmth_history)
                / 50.0,  # Normalized history length
            ],
            dtype=np.float32,
        )

        return state

    def _check_termination(self):
        """Check if episode should terminate."""
        if self.current_step >= self.max_steps_per_episode:
            return True

        # Check for therapeutic success
        if (
            hasattr(self.patient_state, "get_trust_level")
            and self.patient_state.get_trust_level() > 0.7
            and len(self.therapist.patient_warmth_history) > 0
            and self.therapist.patient_warmth_history[-1] > 0.8
        ):
            return True

        return False

    def get_state_dimension(self):
        """Get dimension of the state space."""
        return 10  # Fixed size based on _get_therapist_state

    def get_therapy_statistics(self):
        """Get comprehensive therapy statistics."""
        return {
            "total_interactions": self.total_interactions,
            "episode_count": self.episode_count,
            "therapist_report": self.therapist.get_therapeutic_report(),
            "current_strategy_fitness": self.therapist.current_strategy.fitness_score,
            "evolution_generations": len(self.therapist.evolution_history),
        }


# Example usage function
def run_therapeutic_experiment():
    """Example of how to run a therapeutic experiment."""
    from agent_configuration import CompetitiveAgentConfig  # Cold, resistant patient

    print(" STARTING THERAPEUTIC EXPERIMENT")
    print("=" * 50)

    # Create cold/resistant patient configuration
    patient_config = CompetitiveAgentConfig(
        initial_trust=-0.3,  # Start with distrust
        initial_satisfaction=-0.2,  # Start dissatisfied
        memory_length=100,  # Long memory of negative experiences
        noise_scale=0.05,  # Consistent cold behavior
    )

    # Create therapeutic experiment
    therapist, patient_agent, therapist_wrapper, environment = (
        create_therapist_patient_experiment(
            patient_config=patient_config,
            therapist_population_size=15,
            therapy_episodes=300,
            evolution_frequency=50,
        )
    )

    print(f"Therapist initialized with {len(therapist.strategy_population)} strategies")
    print(f" Patient configured as resistant/cold")

    # Create custom trainer for therapeutic interaction
    from sac_algorithm import SACTrainer

    trainer = SACTrainer(
        agent1=therapist_wrapper,  # Therapist
        agent2=patient_agent,  # Patient
        environment=environment,
        payoff_calculator=environment.payoff_calculator,
        episodes_per_training=300,
        steps_per_episode=50,
        evaluation_frequency=30,
        save_frequency=100,
    )

    # Train the therapeutic interaction
    print(" Starting therapeutic training...")
    results = trainer.train("./therapeutic_models")

    # Generate therapeutic analysis
    environment.therapist.visualize_therapy_progress("./therapeutic_progress.png")
    final_report = environment.get_therapy_statistics()

    print("\n THERAPEUTIC EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"Total interactions: {final_report['total_interactions']}")
    print(f"Episodes completed: {final_report['episode_count']}")
    print(f"Strategy generations: {final_report['evolution_generations']}")
    print(f"Best strategy fitness: {final_report['current_strategy_fitness']:.3f}")

    therapist_report = final_report["therapist_report"]
    if "patient_progress" in therapist_report:
        progress = therapist_report["patient_progress"]
        print(f"\nPatient Progress:")
        print(
            f"  Warmth: {progress['baseline_warmth']:.3f} → {progress['current_warmth']:.3f}"
        )
        print(f"  Trust: {progress['current_trust']:.3f}")
        print(f"  Progress: {progress['warmth_progress']:.3f}")

    return therapist, patient_agent, environment, results


if __name__ == "__main__":
    # Run example therapeutic experiment
    try:
        therapist, patient, environment, results = run_therapeutic_experiment()
        print("\n Therapeutic experiment completed successfully!")
    except Exception as e:
        print(f"\n Error in therapeutic experiment: {e}")
        import traceback

        traceback.print_exc()
