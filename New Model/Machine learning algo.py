from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from Reinforcement_learning import (
    LearningConfig,
    WarmthLearningAgent,
    WarmthActionSpace,
)
from Configurations import run_simulation, SimulationConfig
from ml_algo_plot import plot_evolution_history


@dataclass
class OpponentConfig:
    """Configuration for opponent agents with extended parameters"""

    # Basic identification
    name: str = "Slow_Learner"

    # Core learning parameters
    learning_rate: float = 0.2
    discount_factor: float = 0.95
    exploration_rate: float = 0.1

    # Personality parameters
    prior_expectation: float = 1
    prior_strength: float = 10.0
    risk_sensitivity: float = 0.3

    # Memory and adaptation
    memory_length: int = 100
    forgetting_rate: float = 0.01


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization"""

    population_size: int = 20
    generations: int = 15
    elite_size: int = 4
    tournament_size: int = 10

    # Mutation settings
    base_mutation_rate: float = 0.3
    alpha_mutation_strength: float = 0.2
    other_mutation_strength: float = 0.2
    mutation_decay_rate: float = 0.995

    # Convergence settings
    smoothing_window: int = 5
    convergence_threshold: float = 0.01


@dataclass
class ParameterRanges:
    """Value ranges for evolvable parameters"""

    alpha: Tuple[float, float] = (0.1, 1)
    risk_sensitivity: Tuple[float, float] = (0.0, 1.0)
    prior_expectation: Tuple[float, float] = (0.0, 1.0)
    prior_strength: Tuple[float, float] = (1.0, 10)


class Individual:
    """Represents a single solution in the population"""

    def __init__(self, params: Dict[str, float]):
        self.params = params
        self.fitness: float = 0.0
        self.config = self._create_learning_config()

    def _create_learning_config(self) -> LearningConfig:
        return LearningConfig(
            alpha=self.params["alpha"],
            risk_sensitivity=self.params["risk_sensitivity"],
            prior_expectation=self.params["prior_expectation"],
            prior_strength=self.params["prior_strength"],
        )

    @classmethod
    def random(cls, param_ranges: Dict[str, Tuple[float, float]]) -> "Individual":
        """Create random individual within parameter ranges"""
        params = {
            key: np.random.uniform(ranges[0], ranges[1])
            for key, ranges in param_ranges.items()
            if isinstance(ranges, tuple)  # Only use tuple ranges
        }
        return cls(params)


class IMutationStrategy(ABC):
    """Interface for mutation strategies"""

    @abstractmethod
    def mutate(
        self, value: float, param_range: Tuple[float, float], strength: float
    ) -> float:
        pass


class ExponentialMutation(IMutationStrategy):
    """Exponential mutation for learning rate"""

    def mutate(
        self, value: float, param_range: Tuple[float, float], strength: float
    ) -> float:
        log_current = np.log(value)
        log_delta = np.random.normal(0, strength)
        new_value = np.exp(log_current + log_delta)
        return np.clip(new_value, param_range[0], param_range[1])


class LinearMutation(IMutationStrategy):
    """Linear mutation for other parameters"""

    def mutate(
        self, value: float, param_range: Tuple[float, float], strength: float
    ) -> float:
        param_range_size = param_range[1] - param_range[0]
        delta = np.random.normal(0, strength)
        new_value = value + delta * param_range_size
        return np.clip(new_value, param_range[0], param_range[1])


class GeneticOptimizer:
    """Handles evolutionary optimization"""

    def __init__(
        self, evolution_params: EvolutionConfig, parameter_ranges: ParameterRanges
    ):
        self.params = evolution_params
        self.ranges = parameter_ranges
        self.population: List[Individual] = []
        self.generation: int = 0
        self.best_history: List[Individual] = []
        self.mutation_strategies: Dict[str, IMutationStrategy] = {
            "alpha": ExponentialMutation(),
            "default": LinearMutation(),
        }
        self.current_mutation_rate = self.params.base_mutation_rate

    def initialize_population(self):
        """Initialize random population"""
        self.population = [
            Individual.random(self.ranges.__dict__)
            for _ in range(self.params.population_size)
        ]

    def evolve(
        self, opponents: List[WarmthLearningAgent], sim_config: SimulationConfig
    ) -> Individual:
        """Single generation evolution step"""
        self._evaluate_population(opponents, sim_config)
        self._sort_population()
        best = self._get_smoothed_best()
        self._adapt_mutation_rate()

        new_population = self._create_next_generation()
        self.population = new_population
        self.generation += 1

        return best

    def _evaluate_population(
        self, opponents: List[WarmthLearningAgent], sim_config: SimulationConfig
    ) -> None:
        for ind in self.population:
            ind.fitness = self._calculate_fitness(ind, opponents, sim_config)

    def _sort_population(self) -> None:
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_history.append(self.population[0])

    def _get_smoothed_best(self) -> Individual:
        if len(self.best_history) < self.params.smoothing_window:
            return self.population[0]

        recent_best = self.best_history[-self.params.smoothing_window :]
        smoothed_params = {}

        for param in self.ranges.__annotations__:
            smoothed_params[param] = np.mean([ind.params[param] for ind in recent_best])

        return Individual(smoothed_params)

    def _adapt_mutation_rate(self) -> None:
        """Adapts mutation rate based on population convergence"""
        if len(self.best_history) > self.params.smoothing_window:
            recent_fitness = [
                ind.fitness
                for ind in self.best_history[-self.params.smoothing_window :]
            ]
            fitness_variance = np.var(recent_fitness)

            if fitness_variance < self.params.convergence_threshold:
                self.current_mutation_rate *= self.params.mutation_decay_rate

    def _create_next_generation(self) -> List[Individual]:
        """Creates new population using selection and mutation"""
        new_population = self._preserve_elite()

        while len(new_population) < self.params.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            child = self._crossover(parent1, parent2)
            self._mutate_individual(child)
            new_population.append(child)

        return new_population

    def _preserve_elite(self) -> List[Individual]:
        """Preserves best individuals"""
        return self.population[: self.params.elite_size]

    def _tournament_select(self) -> Individual:
        """Selects individual using tournament selection"""
        tournament = np.random.choice(
            self.population, self.params.tournament_size, replace=False
        )
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Creates child using uniform crossover"""
        child_params = {}
        for param in self.ranges.__annotations__:
            if np.random.random() < 0.5:
                child_params[param] = parent1.params[param]
            else:
                child_params[param] = parent2.params[param]
        return Individual(child_params)

    def _mutate_individual(self, individual: Individual) -> None:
        """Applies mutation to an individual"""
        for param, value in individual.params.items():
            if np.random.random() < self.current_mutation_rate:
                param_range = getattr(self.ranges, param)
                strategy = self.mutation_strategies.get(
                    param, self.mutation_strategies["default"]
                )
                strength = (
                    self.params.alpha_mutation_strength
                    if param == "alpha"
                    else self.params.other_mutation_strength
                )

                individual.params[param] = strategy.mutate(value, param_range, strength)

    def _calculate_fitness(
        self,
        individual: Individual,
        opponents: List[WarmthLearningAgent],
        sim_config: SimulationConfig,
    ) -> float:
        """Calculate fitness score for an individual against all opponents"""
        total_points = 0
        action_space = WarmthActionSpace(n_bins=sim_config.n_bins)

        # Create agent from individual's parameters
        agent = WarmthLearningAgent(individual.config, action_space, "Evolving_Agent")

        # Play against each opponent
        for opp in opponents:
            payoffs1, _ = run_simulation(agent, opp, sim_config)
            total_points += sum(payoffs1)

        # Return average points across all opponents
        return total_points / len(opponents)


def evaluate_fitness(
    individual: Individual,
    opponents: List[WarmthLearningAgent],
    sim_config: SimulationConfig,
) -> float:
    """Run round-robin tournament and return total points"""
    total_points = 0
    action_space = WarmthActionSpace(n_bins=sim_config.n_bins)
    agent = WarmthLearningAgent(individual.config, action_space, "Evolving")

    # Play against each opponent
    for opp in opponents:
        payoffs1, _ = run_simulation(agent, opp, sim_config)
        total_points += sum(payoffs1)

    return total_points


def run_evolution(opponent_configs: List[OpponentConfig]):
    """Run evolution with specified opponent configurations"""
    # Setup configurations
    evolution_params = EvolutionConfig()
    parameter_ranges = ParameterRanges()
    sim_config = SimulationConfig(n_rounds=500)

    # Create opponent pool from configs
    opponents = create_opponent_pool(opponent_configs, sim_config)

    # Initialize optimizer with both required parameters
    optimizer = GeneticOptimizer(evolution_params, parameter_ranges)
    optimizer.initialize_population()

    history = []

    for gen in range(evolution_params.generations):
        best = optimizer.evolve(opponents, sim_config)
        history.append(
            {
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": np.mean([ind.fitness for ind in optimizer.population]),
                "best_params": best.params.copy(),
            }
        )

        if gen % 5 == 0:  # Only print every 5 generations
            print(f"Generation {gen}/{evolution_params.generations}")

    print("\nEvolution complete!")
    print("\nBest solution found:")
    print("Parameters:", {k: f"{v:.3f}" for k, v in best.params.items()})
    print(f"Fitness: {best.fitness:.2f}")

    return history


def create_opponent_pool(
    opponent_configs: List[OpponentConfig], sim_config: SimulationConfig
) -> List[WarmthLearningAgent]:
    """Create a pool of opponents from configurations"""
    action_space = WarmthActionSpace(n_bins=sim_config.n_bins)
    opponents = []

    for i, opp_config in enumerate(opponent_configs):
        config = LearningConfig(
            prior_expectation=opp_config.prior_expectation,
            prior_strength=opp_config.prior_strength,
            risk_sensitivity=opp_config.risk_sensitivity,
        )
        opponent = WarmthLearningAgent(config, action_space, f"{opp_config.name}_{i}")
        opponents.append(opponent)

    return opponents


if __name__ == "__main__":
    # Example opponent configurations
    opponent_configs = [
        OpponentConfig(
            name="Slow_Learner",
            learning_rate=0.1,
            prior_expectation=0.8,
            prior_strength=50.0,
            risk_sensitivity=0.7,
            memory_length=200,
        ),
        OpponentConfig(
            name="Quick_Adapter",
            learning_rate=0.1,
            prior_expectation=0.8,
            prior_strength=10.0,
            risk_sensitivity=0.2,
            memory_length=100,
        ),
        OpponentConfig(
            name="Risk_Taker",
            learning_rate=0.3,
            prior_expectation=0.9,
            risk_sensitivity=0.1,
            exploration_rate=0.2,
        ),
    ]

    # Run evolution with these opponents
    history = run_evolution(opponent_configs)

    # Plot results
    fig = plot_evolution_history(history)
    plt.show()
