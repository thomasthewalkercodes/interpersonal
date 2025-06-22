"""
Enhanced Therapeutic System - Optimized for Maximum Warm-Warm Interactions
This version includes advanced techniques to push patients toward sustained warm cooperation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque, defaultdict
import copy
import json


class TherapeuticPhase(Enum):
    """Enhanced phases of the therapeutic process."""

    ASSESSMENT = "assessment"  # Understanding patient baseline
    TRUST_BUILDING = "trust_building"  # Focus purely on building trust
    MIRRORING = "mirroring"  # Advanced behavioral mirroring
    GRADUAL_LEADING = "gradual_leading"  # Very small incremental warmth increases
    WARMTH_ANCHORING = "warmth_anchoring"  # Lock in warmth gains
    COOPERATIVE_REINFORCEMENT = "cooperative_reinforcement"  # Reward patient warmth
    BREAKTHROUGH_PUSH = "breakthrough_push"  # Final push to full warmth
    MAINTENANCE = "maintenance"  # Sustain warm-warm interactions


@dataclass
class AdvancedTherapeuticStrategy:
    """
    Advanced therapeutic strategy with sophisticated parameters for maximum warm-warm outcomes.
    """

    # Core strategy parameters (enhanced)
    trust_building_intensity: float = 0.95  # How intensely to focus on trust first
    mirroring_precision: float = 0.98  # How precisely to mirror patient
    leading_micro_steps: float = 0.03  # Tiny incremental warmth increases
    warmth_acceleration: float = 1.05  # Multiplier for warmth gains
    patience_factor: float = 2.0  # How patient to be with progress

    # Advanced trust dynamics
    trust_momentum_factor: float = 0.85  # How to build on trust gains
    trust_recovery_rate: float = 0.7  # How quickly to recover from trust loss
    trust_threshold_adaptive: bool = True  # Whether to adapt trust thresholds

    # Sophisticated warmth progression
    warmth_gradient_steps: int = 15  # Number of micro-steps to target
    warmth_plateau_detection: float = 0.02  # Detect when patient plateaus
    warmth_breakthrough_trigger: float = 0.75  # When to attempt breakthrough
    warmth_target: float = 0.9  # Ultimate warmth goal (higher)

    # Reinforcement and reward mechanisms
    cooperative_bonus_multiplier: float = 2.5  # Bonus for patient cooperation
    warmth_momentum_bonus: float = 1.8  # Bonus for sustained warmth
    regression_penalty_mitigation: float = 0.6  # Reduce impact of temporary setbacks

    # Breakthrough and maintenance
    breakthrough_persistence: int = 25  # Steps to maintain breakthrough attempt
    maintenance_vigilance: float = 0.9  # How carefully to maintain gains
    relapse_prevention_strength: float = 0.8  # Strength of relapse prevention

    # Adaptive learning parameters
    strategy_adaptation_rate: float = 0.15  # How quickly to adapt strategy
    patient_model_learning: float = 0.2  # How well to learn patient patterns
    context_sensitivity: float = 0.7  # Sensitivity to context changes

    # Evolution and fitness
    fitness_score: float = 0.0
    generation: int = 0
    therapy_sessions: int = 0
    warm_warm_achievements: int = 0  # Count of successful warm-warm periods
    sustained_warmth_duration: int = 0  # Longest period of sustained warmth

    def to_dict(self):
        """Convert strategy to dictionary for JSON serialization."""
        return {
            k: (
                float(v)
                if isinstance(v, (int, float, np.floating, np.integer))
                else bool(v) if isinstance(v, bool) else v
            )
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create strategy from dictionary."""
        return cls(**data)

    def calculate_fitness(self, therapy_results: Dict) -> float:
        """
        Calculate fitness with heavy emphasis on warm-warm interactions.
        """
        base_fitness = therapy_results.get("warmth_progress", 0) * 3.0
        trust_fitness = therapy_results.get("trust_progress", 0) * 2.0

        # MAJOR bonus for warm-warm achievements
        warm_warm_bonus = therapy_results.get("warm_warm_periods", 0) * 10.0
        sustained_bonus = therapy_results.get("max_sustained_warmth", 0) * 5.0

        # Bonus for breaking through resistance
        breakthrough_bonus = therapy_results.get("breakthrough_achieved", False) * 15.0

        # Penalty for instability
        instability_penalty = therapy_results.get("warmth_variance", 0) * -2.0

        total_fitness = (
            base_fitness
            + trust_fitness
            + warm_warm_bonus
            + sustained_bonus
            + breakthrough_bonus
            + instability_penalty
        )

        return max(0.0, total_fitness)

    def mutate(self, mutation_strength: float = 0.08) -> "AdvancedTherapeuticStrategy":
        """Create a mutated version with focus on warm-warm optimization."""
        new_strategy = copy.deepcopy(self)

        # Mutate key parameters for warm-warm success
        mutation_params = [
            ("trust_building_intensity", 0.8, 1.0),
            ("mirroring_precision", 0.9, 1.0),
            ("leading_micro_steps", 0.01, 0.08),
            ("warmth_acceleration", 1.0, 1.3),
            ("patience_factor", 1.2, 3.0),
            ("cooperative_bonus_multiplier", 1.5, 4.0),
            ("breakthrough_persistence", 15, 40),
        ]

        for param_name, min_val, max_val in mutation_params:
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                current_val = getattr(new_strategy, param_name)
                noise = np.random.normal(0, mutation_strength)
                new_val = current_val + noise
                setattr(new_strategy, param_name, np.clip(new_val, min_val, max_val))

        new_strategy.generation += 1
        new_strategy.fitness_score = 0.0
        return new_strategy


class AdvancedTherapistAgent:
    """
    Advanced therapist agent specifically designed to maximize warm-warm interactions.
    Uses sophisticated psychological principles and adaptive learning.
    """

    def __init__(self, agent_id: str = "advanced_therapist", population_size: int = 25):
        self.agent_id = agent_id
        self.population_size = population_size

        # Initialize advanced strategy population
        self.strategy_population = self._initialize_advanced_population()
        self.current_strategy = self.strategy_population[0]

        # Enhanced therapeutic state tracking
        self.current_phase = TherapeuticPhase.ASSESSMENT
        self.patient_baseline_warmth = 0.0
        self.patient_trust_history = deque(maxlen=100)  # Longer memory
        self.patient_warmth_history = deque(maxlen=100)
        self.therapist_warmth_history = deque(maxlen=100)

        # Advanced progress tracking
        self.therapy_step = 0
        self.phase_step = 0
        self.warm_warm_periods = []  # Track periods of mutual warmth
        self.current_warm_warm_streak = 0
        self.max_warm_warm_streak = 0
        self.breakthrough_achieved = False

        # Sophisticated patient modeling
        self.patient_warmth_model = {
            "responsiveness": 0.5,  # How responsive patient is to therapist
            "trust_threshold": 0.3,  # Trust needed for patient to follow
            "warmth_ceiling": 0.8,  # Estimated max patient warmth
            "regression_tendency": 0.2,  # Tendency to regress
        }

        # Adaptive parameters
        self.trust_trend = 0.0
        self.warmth_trend = 0.0
        self.trust_momentum = 0.0
        self.warmth_momentum = 0.0

        # Performance metrics for warm-warm optimization
        self.session_metrics = {
            "warm_warm_percentage": 0.0,
            "sustained_warmth_duration": 0,
            "breakthrough_count": 0,
            "trust_stability": 0.0,
        }

        print(
            f"[ADVANCED THERAPIST] Initialized with {population_size} sophisticated strategies"
        )

    def _initialize_advanced_population(self) -> List[AdvancedTherapeuticStrategy]:
        """Initialize population with diverse advanced strategies."""
        population = []

        # Create strategies with different approaches to warm-warm optimization
        for i in range(self.population_size):
            if i < 5:  # Ultra-patient strategies
                strategy = AdvancedTherapeuticStrategy(
                    trust_building_intensity=np.random.uniform(0.9, 1.0),
                    mirroring_precision=np.random.uniform(0.95, 1.0),
                    leading_micro_steps=np.random.uniform(0.01, 0.03),
                    patience_factor=np.random.uniform(2.5, 4.0),
                    warmth_acceleration=np.random.uniform(1.0, 1.15),
                )
            elif i < 10:  # Breakthrough-focused strategies
                strategy = AdvancedTherapeuticStrategy(
                    trust_building_intensity=np.random.uniform(0.85, 0.95),
                    cooperative_bonus_multiplier=np.random.uniform(2.0, 4.0),
                    breakthrough_persistence=np.random.randint(20, 35),
                    warmth_acceleration=np.random.uniform(1.1, 1.4),
                )
            elif i < 15:  # Momentum-based strategies
                strategy = AdvancedTherapeuticStrategy(
                    warmth_momentum_bonus=np.random.uniform(1.5, 2.5),
                    trust_momentum_factor=np.random.uniform(0.7, 0.9),
                    strategy_adaptation_rate=np.random.uniform(0.1, 0.25),
                )
            else:  # Adaptive hybrid strategies
                strategy = AdvancedTherapeuticStrategy(
                    trust_building_intensity=np.random.uniform(0.85, 0.98),
                    mirroring_precision=np.random.uniform(0.9, 0.99),
                    leading_micro_steps=np.random.uniform(0.02, 0.06),
                    patient_model_learning=np.random.uniform(0.15, 0.3),
                    context_sensitivity=np.random.uniform(0.6, 0.8),
                )

            population.append(strategy)

        return population

    def select_action(
        self, state: np.ndarray, patient_action: float, patient_trust: float
    ) -> float:
        """
        Select therapeutic action using advanced warm-warm optimization.
        """
        # Update comprehensive tracking
        self._update_advanced_tracking(patient_action, patient_trust)

        # Update patient model
        self._update_patient_model(patient_action, patient_trust)

        # Determine optimal therapeutic phase
        self._update_advanced_therapeutic_phase()

        # Select action using sophisticated strategy
        action = self._select_advanced_therapeutic_action(patient_action, patient_trust)

        # Track warm-warm interactions
        self._track_warm_warm_interactions(action, patient_action)

        self.therapy_step += 1
        self.phase_step += 1

        return action

    def _update_advanced_tracking(self, patient_action: float, patient_trust: float):
        """Update advanced tracking with momentum and trend analysis."""
        patient_warmth = (patient_action + 1) / 2

        self.patient_warmth_history.append(patient_warmth)
        self.patient_trust_history.append(patient_trust)

        # Calculate sophisticated trends
        if len(self.patient_trust_history) >= 10:
            recent_trust = list(self.patient_trust_history)[-10:]
            self.trust_trend = np.polyfit(range(10), recent_trust, 1)[0]
            self.trust_momentum = self.trust_momentum * 0.9 + self.trust_trend * 0.1

        if len(self.patient_warmth_history) >= 10:
            recent_warmth = list(self.patient_warmth_history)[-10:]
            self.warmth_trend = np.polyfit(range(10), recent_warmth, 1)[0]
            self.warmth_momentum = self.warmth_momentum * 0.9 + self.warmth_trend * 0.1

        # Update baseline during assessment
        if (
            self.current_phase == TherapeuticPhase.ASSESSMENT
            and len(self.patient_warmth_history) >= 15
        ):
            self.patient_baseline_warmth = np.mean(
                list(self.patient_warmth_history)[-15:]
            )

    def _update_patient_model(self, patient_action: float, patient_trust: float):
        """Update sophisticated patient model for better prediction."""
        if (
            len(self.patient_warmth_history) < 5
            or len(self.therapist_warmth_history) < 5
        ):
            return

        strategy = self.current_strategy
        patient_warmth = (patient_action + 1) / 2

        # Estimate patient responsiveness
        if len(self.therapist_warmth_history) >= 2:
            therapist_change = (
                self.therapist_warmth_history[-1] - self.therapist_warmth_history[-2]
            )
            patient_change = patient_warmth - self.patient_warmth_history[-2]

            if abs(therapist_change) > 0.01:  # Avoid division by zero
                responsiveness = patient_change / therapist_change
                self.patient_warmth_model["responsiveness"] = (
                    self.patient_warmth_model["responsiveness"] * 0.9
                    + responsiveness * 0.1
                )

        # Estimate warmth ceiling
        recent_max_warmth = max(list(self.patient_warmth_history)[-20:])
        self.patient_warmth_model["warmth_ceiling"] = max(
            self.patient_warmth_model["warmth_ceiling"], recent_max_warmth * 1.1
        )

        # Update trust threshold based on when patient becomes responsive
        if (
            patient_trust > self.patient_warmth_model["trust_threshold"]
            and self.warmth_trend > 0
        ):
            self.patient_warmth_model["trust_threshold"] = (
                self.patient_warmth_model["trust_threshold"] * 0.95
                + patient_trust * 0.05
            )

    def _update_advanced_therapeutic_phase(self):
        """Update therapeutic phase with sophisticated logic."""
        strategy = self.current_strategy
        current_trust = (
            self.patient_trust_history[-1] if self.patient_trust_history else 0.0
        )
        current_warmth = (
            self.patient_warmth_history[-1] if self.patient_warmth_history else 0.0
        )

        # Phase transition logic optimized for warm-warm outcomes
        if self.current_phase == TherapeuticPhase.ASSESSMENT:
            if self.phase_step >= 20:  # Longer assessment for better baseline
                self.current_phase = TherapeuticPhase.TRUST_BUILDING
                self.phase_step = 0
                print(
                    f"[THERAPIST] Moving to TRUST_BUILDING. Baseline warmth: {self.patient_baseline_warmth:.3f}"
                )

        elif self.current_phase == TherapeuticPhase.TRUST_BUILDING:
            # Focus purely on trust until it's solid
            if current_trust >= 0.4 and self.trust_momentum > 0:
                self.current_phase = TherapeuticPhase.MIRRORING
                self.phase_step = 0
                print(
                    f"[THERAPIST] Trust established ({current_trust:.3f}), moving to MIRRORING"
                )

        elif self.current_phase == TherapeuticPhase.MIRRORING:
            # Mirror until patient shows readiness for leading
            if (
                current_trust >= 0.5
                and self.trust_momentum > -0.01
                and self.phase_step >= 15
            ):
                self.current_phase = TherapeuticPhase.GRADUAL_LEADING
                self.phase_step = 0
                print(
                    f"[THERAPIST] Patient ready for leading, trust: {current_trust:.3f}"
                )

        elif self.current_phase == TherapeuticPhase.GRADUAL_LEADING:
            # Gradual micro-increases in warmth
            if self.warmth_momentum > 0.02 and self.phase_step >= 20:
                self.current_phase = TherapeuticPhase.WARMTH_ANCHORING
                self.phase_step = 0
                print(f"[THERAPIST] Patient following warmth lead, anchoring gains")
            elif self.trust_trend < -0.05:  # Trust dropping
                self.current_phase = TherapeuticPhase.TRUST_BUILDING
                self.phase_step = 0
                print(f"[THERAPIST] Trust declining, returning to trust building")

        elif self.current_phase == TherapeuticPhase.WARMTH_ANCHORING:
            # Lock in current warmth level
            if current_warmth >= 0.6 and self.phase_step >= 15:
                self.current_phase = TherapeuticPhase.COOPERATIVE_REINFORCEMENT
                self.phase_step = 0
                print(
                    f"[THERAPIST] Good warmth level achieved, reinforcing cooperation"
                )

        elif self.current_phase == TherapeuticPhase.COOPERATIVE_REINFORCEMENT:
            # Reward and encourage mutual warmth
            if current_warmth >= strategy.warmth_breakthrough_trigger:
                self.current_phase = TherapeuticPhase.BREAKTHROUGH_PUSH
                self.phase_step = 0
                print(
                    f"[THERAPIST] Ready for breakthrough push! Warmth: {current_warmth:.3f}"
                )

        elif self.current_phase == TherapeuticPhase.BREAKTHROUGH_PUSH:
            # Push for maximum warmth
            if current_warmth >= strategy.warmth_target * 0.9:
                self.current_phase = TherapeuticPhase.MAINTENANCE
                self.breakthrough_achieved = True
                self.phase_step = 0
                print(
                    f"[BREAKTHROUGH] Achieved target warmth! Entering maintenance mode."
                )
            elif self.phase_step >= strategy.breakthrough_persistence:
                # If breakthrough attempt fails, return to reinforcement
                self.current_phase = TherapeuticPhase.COOPERATIVE_REINFORCEMENT
                self.phase_step = 0
                print(
                    f"[THERAPIST] Breakthrough attempt timeout, returning to reinforcement"
                )

    def _select_advanced_therapeutic_action(
        self, patient_action: float, patient_trust: float
    ) -> float:
        """Select action using advanced therapeutic principles."""
        strategy = self.current_strategy
        patient_warmth = (patient_action + 1) / 2

        if self.current_phase == TherapeuticPhase.ASSESSMENT:
            # Neutral but slightly positive assessment
            therapist_warmth = 0.45 + np.random.normal(0, 0.05)

        elif self.current_phase == TherapeuticPhase.TRUST_BUILDING:
            # Focus entirely on building trust through careful matching
            matching_warmth = patient_warmth * strategy.trust_building_intensity
            trust_bonus = min(
                0.1, patient_trust * 0.2
            )  # Small bonus for existing trust
            therapist_warmth = matching_warmth + trust_bonus

        elif self.current_phase == TherapeuticPhase.MIRRORING:
            # Precise mirroring with minimal leading
            therapist_warmth = patient_warmth * strategy.mirroring_precision
            therapist_warmth += strategy.leading_micro_steps  # Tiny lead

        elif self.current_phase == TherapeuticPhase.GRADUAL_LEADING:
            # Very small incremental increases
            target_warmth = min(
                patient_warmth
                + strategy.leading_micro_steps * (1 + self.trust_momentum),
                self.patient_warmth_model["warmth_ceiling"] * 0.8,
            )
            therapist_warmth = target_warmth

        elif self.current_phase == TherapeuticPhase.WARMTH_ANCHORING:
            # Maintain current level with slight encouragement
            therapist_warmth = patient_warmth + 0.02
            if self.warmth_momentum > 0:
                therapist_warmth += strategy.warmth_acceleration * 0.01

        elif self.current_phase == TherapeuticPhase.COOPERATIVE_REINFORCEMENT:
            # Reward patient warmth with bonus warmth
            base_warmth = patient_warmth + strategy.leading_micro_steps * 2
            cooperation_bonus = min(
                0.1, patient_warmth * strategy.cooperative_bonus_multiplier * 0.05
            )
            therapist_warmth = base_warmth + cooperation_bonus

        elif self.current_phase == TherapeuticPhase.BREAKTHROUGH_PUSH:
            # Push toward maximum warmth
            target = strategy.warmth_target
            current_therapist = (
                self.therapist_warmth_history[-1]
                if self.therapist_warmth_history
                else 0.5
            )
            # Gradual approach to target
            approach_rate = 0.05 * (1 + patient_trust)
            therapist_warmth = current_therapist + approach_rate
            therapist_warmth = min(therapist_warmth, target)

        elif self.current_phase == TherapeuticPhase.MAINTENANCE:
            # Maintain high warmth with patient
            therapist_warmth = min(strategy.warmth_target, patient_warmth + 0.05)
            # Add momentum bonus for sustained cooperation
            if self.current_warm_warm_streak > 10:
                therapist_warmth += 0.02

        else:
            therapist_warmth = 0.5  # Fallback

        # Apply trust-based adjustments
        if self.trust_trend < -0.02:  # Trust declining
            therapist_warmth -= abs(self.trust_trend) * 2.0  # Strong adjustment
        elif self.trust_momentum > 0.02:  # Trust building
            therapist_warmth += self.trust_momentum * strategy.warmth_acceleration

        # Apply patient model constraints
        max_appropriate = self.patient_warmth_model["warmth_ceiling"]
        if patient_trust < self.patient_warmth_model["trust_threshold"]:
            max_appropriate = patient_warmth + 0.05  # Don't lead too much without trust

        therapist_warmth = min(therapist_warmth, max_appropriate)

        # Clamp and convert to action space
        therapist_warmth = np.clip(therapist_warmth, 0.0, 1.0)
        action = therapist_warmth * 2 - 1

        # Store for patient model updates
        self.therapist_warmth_history.append(therapist_warmth)

        return action

    def _track_warm_warm_interactions(
        self, therapist_action: float, patient_action: float
    ):
        """Track periods of mutual warmth for optimization."""
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Define warm-warm threshold
        warm_threshold = 0.6

        if therapist_warmth >= warm_threshold and patient_warmth >= warm_threshold:
            self.current_warm_warm_streak += 1
            self.max_warm_warm_streak = max(
                self.max_warm_warm_streak, self.current_warm_warm_streak
            )
        else:
            if self.current_warm_warm_streak > 0:
                self.warm_warm_periods.append(self.current_warm_warm_streak)
            self.current_warm_warm_streak = 0

        # Update session metrics
        if self.therapy_step > 0:
            total_warm_warm_steps = (
                sum(self.warm_warm_periods) + self.current_warm_warm_streak
            )
            self.session_metrics["warm_warm_percentage"] = (
                total_warm_warm_steps / self.therapy_step
            )

    def evaluate_session_performance(self) -> Dict[str, float]:
        """Evaluate session with emphasis on warm-warm achievements."""
        if len(self.patient_warmth_history) < 10:
            return {"warmth_progress": 0, "warm_warm_periods": 0}

        # Basic progress metrics
        final_warmth = self.patient_warmth_history[-1]
        warmth_progress = final_warmth - self.patient_baseline_warmth

        final_trust = (
            self.patient_trust_history[-1] if self.patient_trust_history else 0
        )
        initial_trust = (
            self.patient_trust_history[0] if self.patient_trust_history else 0
        )
        trust_progress = final_trust - initial_trust

        # Warm-warm specific metrics
        total_warm_warm_steps = (
            sum(self.warm_warm_periods) + self.current_warm_warm_streak
        )
        max_sustained_warmth = (
            max(self.warm_warm_periods)
            if self.warm_warm_periods
            else self.current_warm_warm_streak
        )

        # Stability metrics
        warmth_variance = (
            np.var(list(self.patient_warmth_history)[-20:])
            if len(self.patient_warmth_history) >= 20
            else 0
        )

        return {
            "warmth_progress": warmth_progress,
            "trust_progress": trust_progress,
            "warm_warm_periods": total_warm_warm_steps,
            "max_sustained_warmth": max_sustained_warmth,
            "breakthrough_achieved": self.breakthrough_achieved,
            "warmth_variance": warmth_variance,
            "final_warmth": final_warmth,
            "final_trust": final_trust,
            "warm_warm_percentage": self.session_metrics["warm_warm_percentage"],
        }

    def evolve_strategies(self, generation_size: int = 50):
        """Evolve strategies with focus on warm-warm optimization."""
        print(f"\n[EVOLUTION] Evolving strategies for maximum warm-warm interactions")

        # Evaluate all strategies
        strategy_fitness = []

        for i, strategy in enumerate(self.strategy_population):
            self.current_strategy = strategy

            # Multiple sessions for robust evaluation
            total_fitness = 0.0
            sessions = max(3, generation_size // len(self.strategy_population))

            for session in range(sessions):
                self._reset_for_evaluation()
                session_results = self._simulate_advanced_therapy_session()
                fitness = strategy.calculate_fitness(session_results)
                total_fitness += fitness

                # Track warm-warm achievements
                strategy.warm_warm_achievements += session_results.get(
                    "warm_warm_periods", 0
                )
                strategy.sustained_warmth_duration = max(
                    strategy.sustained_warmth_duration,
                    session_results.get("max_sustained_warmth", 0),
                )

            avg_fitness = total_fitness / sessions
            strategy.fitness_score = avg_fitness
            strategy_fitness.append((i, avg_fitness))

            print(
                f"   Strategy {i}: Fitness {avg_fitness:.3f}, "
                f"Warm-warm: {strategy.warm_warm_achievements}, "
                f"Max sustained: {strategy.sustained_warmth_duration}"
            )

        # Evolution with elite selection
        strategy_fitness.sort(key=lambda x: x[1], reverse=True)
        elite_count = max(3, int(len(self.strategy_population) * 0.3))
        elite_strategies = [
            self.strategy_population[i] for i, _ in strategy_fitness[:elite_count]
        ]

        # Generate new population
        new_population = elite_strategies.copy()

        while len(new_population) < self.population_size:
            if random.random() < 0.6:  # Crossover
                parent1, parent2 = random.sample(elite_strategies, 2)
                child = self._crossover_strategies(parent1, parent2)
            else:  # Mutation
                parent = random.choice(elite_strategies)
                child = parent.mutate(0.1)

            new_population.append(child)

        self.strategy_population = new_population
        self.current_strategy = elite_strategies[0]  # Use best strategy

        # Log evolution
        best_fitness = strategy_fitness[0][1]
        print(f"[EVOLUTION] Best fitness: {best_fitness:.3f}")
        print(
            f"[EVOLUTION] Best strategy warm-warm achievements: {elite_strategies[0].warm_warm_achievements}"
        )

    def _crossover_strategies(
        self, parent1: AdvancedTherapeuticStrategy, parent2: AdvancedTherapeuticStrategy
    ) -> AdvancedTherapeuticStrategy:
        """Create offspring from two parent strategies."""
        child_params = {}

        # Mix parameters from both parents
        for key in parent1.__dict__:
            if key not in ["fitness_score", "generation", "therapy_sessions"]:
                if random.random() < 0.5:
                    child_params[key] = getattr(parent1, key)
                else:
                    child_params[key] = getattr(parent2, key)

        child = AdvancedTherapeuticStrategy(**child_params)
        child.generation = max(parent1.generation, parent2.generation) + 1
        return child

    def _reset_for_evaluation(self):
        """Reset state for strategy evaluation."""
        self.current_phase = TherapeuticPhase.ASSESSMENT
        self.therapy_step = 0
        self.phase_step = 0
        self.patient_warmth_history.clear()
        self.patient_trust_history.clear()
        self.therapist_warmth_history.clear()
        self.warm_warm_periods = []
        self.current_warm_warm_streak = 0
        self.breakthrough_achieved = False

    def _simulate_advanced_therapy_session(self) -> Dict[str, float]:
        """Simulate therapy session with sophisticated patient model."""
        # More realistic patient simulation
        patient_warmth = 0.15 + np.random.normal(0, 0.05)  # Start cold
        patient_trust = -0.2 + np.random.normal(0, 0.1)  # Start distrustful

        # Patient personality parameters
        patient_responsiveness = np.random.uniform(0.3, 0.8)
        patient_trust_building_rate = np.random.uniform(0.05, 0.15)
        patient_max_warmth = np.random.uniform(0.7, 0.95)
        patient_resistance_decay = np.random.uniform(0.02, 0.08)

        # Simulation loop
        for step in range(120):  # Longer sessions for breakthrough opportunities
            # Therapist action
            therapist_action = self.select_action(
                state=np.zeros(10),  # Dummy state
                patient_action=patient_warmth * 2 - 1,
                patient_trust=patient_trust,
            )

            therapist_warmth = (therapist_action + 1) / 2

            # Advanced patient response model
            warmth_diff = therapist_warmth - patient_warmth
            trust_diff = abs(patient_warmth - therapist_warmth)

            # Trust dynamics
            if trust_diff < 0.1:  # Therapist matching well
                trust_change = patient_trust_building_rate * (1 + patient_trust * 0.5)
            else:  # Mismatch creates trust issues
                trust_change = -trust_diff * 0.1

            # Add bonus for sustained positive interactions
            if patient_trust > 0.2 and therapist_warmth > 0.5:
                trust_change += 0.02

            patient_trust += trust_change
            patient_trust = np.clip(patient_trust, -0.5, 1.0)

            # Warmth dynamics - more sophisticated
            if patient_trust > 0.2:  # Patient will follow when trusting
                # Follow therapist with some responsiveness
                target_warmth = patient_warmth + warmth_diff * patient_responsiveness

                # Add breakthrough mechanics
                if therapist_warmth > 0.75 and patient_trust > 0.5:
                    # Breakthrough potential
                    breakthrough_factor = min(0.15, (therapist_warmth - 0.75) * 2)
                    target_warmth += breakthrough_factor

                # Gradual approach to target
                warmth_change = (target_warmth - patient_warmth) * 0.3

            else:  # Low trust - patient resists or stays cold
                warmth_change = -patient_resistance_decay * (1 - patient_trust)

            # Add momentum for sustained cooperation
            if len(self.patient_warmth_history) > 5:
                recent_trend = np.mean(
                    list(self.patient_warmth_history)[-5:]
                ) - np.mean(list(self.patient_warmth_history)[-10:-5])
                if recent_trend > 0:  # Positive momentum
                    warmth_change += recent_trend * 0.5

            patient_warmth += warmth_change
            patient_warmth = np.clip(patient_warmth, 0.0, min(patient_max_warmth, 1.0))

            # Add some noise for realism
            patient_warmth += np.random.normal(0, 0.02)
            patient_trust += np.random.normal(0, 0.01)

        # Evaluate session results
        return self.evaluate_session_performance()


# Enhanced SAC Wrapper for integration
class AdvancedTherapistSACWrapper:
    """
    Enhanced wrapper for SAC integration with advanced therapeutic capabilities.
    """

    def __init__(self, therapist_agent: AdvancedTherapistAgent, state_dim: int):
        self.therapist = therapist_agent
        self.state_dim = state_dim
        self.last_patient_action = 0.0
        self.session_step = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select action using advanced therapeutic strategy."""
        # Extract more information from state if available
        patient_trust = state[1] if len(state) > 1 else 0.0
        patient_satisfaction = state[2] if len(state) > 2 else 0.0

        # Use satisfaction as additional trust signal
        enhanced_trust = (patient_trust + patient_satisfaction * 0.5) / 1.5

        action = self.therapist.select_action(
            state, self.last_patient_action, enhanced_trust
        )
        self.session_step += 1

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store patient action for therapist modeling."""
        # Extract patient action from state if possible
        if len(state) > 0:
            # Estimate patient action from state changes
            self.last_patient_action = (
                state[0] * 2 - 1
            )  # Convert trust to action-like value

    def train_step(self):
        """Enhanced training step with strategy evolution."""
        # Periodically evolve strategies
        if self.session_step % 500 == 0 and self.session_step > 0:
            print(f"[EVOLUTION TRIGGER] Evolving at step {self.session_step}")
            self.therapist.evolve_strategies(generation_size=40)

        return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": 0.1, "alpha_loss": 0.0}

    def save_model(self, filepath: str):
        """Save advanced therapist with enhanced metrics."""
        try:
            # Convert strategies to serializable format
            strategies_data = [
                strategy.to_dict() for strategy in self.therapist.strategy_population
            ]

            # Enhanced save data
            save_data = {
                "strategies": strategies_data,
                "therapeutic_state": {
                    "current_phase": self.therapist.current_phase.value,
                    "therapy_step": int(self.therapist.therapy_step),
                    "breakthrough_achieved": bool(self.therapist.breakthrough_achieved),
                    "max_warm_warm_streak": int(self.therapist.max_warm_warm_streak),
                    "session_metrics": self.therapist.session_metrics,
                    "patient_model": self.therapist.patient_warmth_model,
                },
                "performance_summary": {
                    "best_strategy_fitness": max(
                        s.fitness_score for s in self.therapist.strategy_population
                    ),
                    "total_warm_warm_achievements": sum(
                        s.warm_warm_achievements
                        for s in self.therapist.strategy_population
                    ),
                    "max_sustained_duration": max(
                        s.sustained_warmth_duration
                        for s in self.therapist.strategy_population
                    ),
                },
            }

            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)

            print(f"[SAVE] Advanced therapist saved to {filepath}")

        except Exception as e:
            print(f"[ERROR] Save failed: {e}")

    def load_model(self, filepath: str):
        """Load advanced therapist model."""
        try:
            with open(filepath, "r") as f:
                save_data = json.load(f)

            # Reconstruct strategies
            strategies = [
                AdvancedTherapeuticStrategy.from_dict(data)
                for data in save_data["strategies"]
            ]

            self.therapist.strategy_population = strategies
            self.therapist.current_strategy = strategies[0]

            # Restore therapeutic state
            if "therapeutic_state" in save_data:
                state = save_data["therapeutic_state"]
                self.therapist.therapy_step = state.get("therapy_step", 0)
                self.therapist.breakthrough_achieved = state.get(
                    "breakthrough_achieved", False
                )
                self.therapist.max_warm_warm_streak = state.get(
                    "max_warm_warm_streak", 0
                )
                self.therapist.session_metrics = state.get("session_metrics", {})
                self.therapist.patient_warmth_model = state.get("patient_model", {})

            print(f"[LOAD] Advanced therapist loaded from {filepath}")

        except Exception as e:
            print(f"[ERROR] Load failed: {e}")


# Enhanced payoff calculator - ONLY modifies therapist incentives
class TherapistOptimizedPayoffCalculator:
    """
    Payoff calculator that gives the therapist enhanced incentives for guiding patients
    toward warmth, while keeping patient payoffs exactly as the original Gaussian system.
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
        """
        Calculate payoffs where:
        - Patient gets PURE Gaussian payoffs (unchanged)
        - Therapist gets enhanced incentives for patient progress
        """

        # Convert to warmth space
        therapist_warmth = (therapist_action + 1) / 2
        patient_warmth = (patient_action + 1) / 2

        # Import your original Gaussian payoff function
        from gaussian_payoff_graph import calculate_warmth_payoff

        # Patient gets PURE Gaussian payoffs - completely unchanged
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
        patient_warmth_reward = patient_warmth**2 * 8.0  # Quadratic reward

        # 2. Huge bonus for warm-warm interactions
        if therapist_warmth >= 0.6 and patient_warmth >= 0.6:
            warm_warm_bonus = 12.0
            # Extra for very warm
            if therapist_warmth >= 0.8 and patient_warmth >= 0.8:
                warm_warm_bonus += 8.0
        else:
            warm_warm_bonus = 0.0

        # 3. Leadership bonus for appropriate leading
        if therapist_warmth > patient_warmth and patient_warmth > 0.3:
            leadership_bonus = (therapist_warmth - patient_warmth) * 4.0
        else:
            leadership_bonus = 0.0

        # 4. Breakthrough bonus for getting patient above thresholds
        breakthrough_bonus = 0.0
        if patient_warmth > 0.7:
            breakthrough_bonus = 15.0
        elif patient_warmth > 0.5:
            breakthrough_bonus = 5.0

        # 5. Patience bonus for gradual approach (not rushing)
        patience_bonus = 0.0
        warmth_diff = abs(therapist_warmth - patient_warmth)
        if (
            warmth_diff < 0.15 and patient_warmth > 0.2
        ):  # Close matching when patient is responsive
            patience_bonus = 2.0

        # 6. Penalty for being too cold when patient is making progress
        coldness_penalty = 0.0
        if therapist_warmth < 0.4 and patient_warmth > 0.4:
            coldness_penalty = -3.0

        # Combine therapist bonuses
        therapist_bonus = (
            patient_warmth_reward
            + warm_warm_bonus
            + leadership_bonus
            + breakthrough_bonus
            + patience_bonus
            + coldness_penalty
        )

        # Final payoffs
        therapist_payoff = therapist_base + therapist_bonus
        # Patient payoff stays PURE Gaussian - no modifications

        return therapist_payoff, patient_payoff


def create_advanced_therapeutic_experiment(
    experiment_name: str = "advanced_warm_warm_therapy",
    episodes: int = 600,
    evolution_frequency: int = 60,
    population_size: int = 25,
):
    """
    Create an advanced therapeutic experiment where only the therapist is optimized,
    keeping the patient with pure Gaussian payoffs.
    """
    print("=" * 80)
    print(f"ADVANCED THERAPEUTIC EXPERIMENT: {experiment_name}")
    print("Therapist optimized for warm-warm, Patient uses pure Gaussian payoffs")
    print("=" * 80)

    # Import required modules
    from agent_configuration import CompetitiveAgentConfig
    from sac_algorithm import SACAgent
    from interaction_environment import InterpersonalEnvironment

    # Create resistant patient with YOUR original settings
    patient_config = CompetitiveAgentConfig(
        initial_trust=-0.3,  # Distrustful (your original setup)
        initial_satisfaction=-0.1,  # Dissatisfied (your original setup)
        memory_length=70,  # Long memory (your original setup)
        lr_actor=5e-4,  # Your original learning rates
        lr_critic=5e-4,
        alpha=0.3,  # Your original exploration
        noise_scale=0.15,  # Your original noise
    )

    # Create patient agent
    patient_state = patient_config.create_initial_state()
    state_dim = patient_state.get_state_dimension()
    patient_agent = SACAgent(state_dim, patient_config.get_sac_params())

    # Create advanced therapist
    therapist = AdvancedTherapistAgent(
        agent_id="advanced_therapist_pure_patient", population_size=population_size
    )

    # Create wrapper
    therapist_wrapper = AdvancedTherapistSACWrapper(therapist, state_dim)

    # Use therapist-only optimized payoff calculator
    # Patient gets PURE Gaussian, therapist gets optimization bonuses
    payoff_calculator = TherapistOptimizedPayoffCalculator(
        alpha=4.0, beta=10.0  # Your original alpha  # Your original beta
    )

    print(
        f"✓ Patient: Resistant + PURE Gaussian payoffs (alpha={payoff_calculator.alpha}, beta={payoff_calculator.beta})"
    )
    print(f"✓ Patient trust: {patient_state.get_trust_level():.3f}")
    print(f"✓ Patient satisfaction: {patient_state.get_satisfaction_level():.3f}")
    print(f"✓ Therapist: Advanced strategies + optimization bonuses")
    print(f"✓ Therapist population: {population_size} strategies")
    print(f"✓ Evolution frequency: every {evolution_frequency} episodes")
    print("-" * 80)

    return therapist, patient_agent, therapist_wrapper, payoff_calculator, patient_state


def main():
    """Test the advanced therapeutic system."""
    print("ADVANCED THERAPEUTIC SYSTEM - WARM-WARM OPTIMIZATION TEST")
    print("=" * 70)

    # Test enhanced components
    try:
        # Create advanced therapist
        therapist = AdvancedTherapistAgent(population_size=8)

        # Test action selection
        dummy_state = np.zeros(10)
        patient_action = -0.7  # Very cold patient
        patient_trust = -0.3  # Distrustful

        action = therapist.select_action(dummy_state, patient_action, patient_trust)

        print(f"✓ Advanced therapist created successfully")
        print(f"✓ Action selection working: {action:.3f}")
        print(f"✓ Current phase: {therapist.current_phase.value}")
        print(f"✓ Patient model: {therapist.patient_warmth_model}")

        # Test evolution
        for strategy in therapist.strategy_population[:3]:
            strategy.fitness_score = np.random.uniform(0.5, 2.0)

        therapist.evolve_strategies(generation_size=20)

        print(f"✓ Strategy evolution successful")
        print(
            f"✓ Best strategy fitness: {max(s.fitness_score for s in therapist.strategy_population):.3f}"
        )

        # Test payoff calculator
        payoff_calc = MaxWarmWarmPayoffCalculator()
        t_payoff, p_payoff = payoff_calc.calculate_payoff(
            0.8, 0.7, "therapist", "patient"
        )

        print(f"✓ Enhanced payoff calculator working")
        print(f"✓ Warm-warm payoffs: Therapist={t_payoff:.1f}, Patient={p_payoff:.1f}")

        print("\n" + "=" * 70)
        print("ADVANCED THERAPEUTIC SYSTEM READY!")
        print("=" * 70)
        print("Key enhancements for warm-warm optimization:")
        print("• 8 sophisticated therapeutic phases")
        print("• Advanced patient modeling and prediction")
        print("• Massive rewards for warm-warm interactions")
        print("• Breakthrough detection and maintenance")
        print("• Genetic evolution focused on cooperation success")
        print("• Momentum-based warmth building")
        print("• Trust-warmth coupling mechanisms")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
