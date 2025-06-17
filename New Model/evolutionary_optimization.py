import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
from Reinforcement_learning import (
    LearningConfig,
    WarmthLearningAgent,
    WarmthActionSpace,
)
from Configurations import run_simulation, SimulationConfig


@dataclass
class OpponentConfig:
    """Configuration for opponent agents with extended parameters"""

    # Basic parameters
    name: str = "Warm"

    # Learning parameters
    learning_rate: float = 0.2  # How quickly they learn (alpha)
    discount_factor: float = 0.95  # How much they value future rewards (gamma)
    exploration_rate: float = 0.1  # How much they explore (epsilon)

    # Personality traits
    prior_expectation: float = 0.5  # Initial warmth expectation
    prior_strength: float = 10.0  # Strength of initial beliefs
    risk_sensitivity: float = 0.3  # Aversion to rejection
    adaptation_speed: float = 0.1  # How quickly they change strategies

    # Memory parameters
    memory_length: int = 100  # How many past interactions they remember
    forgetting_rate: float = 0.01  # How quickly they forget old experiences


def create_opponent_pool(
    opponent_configs: List[OpponentConfig], sim_config: SimulationConfig
) -> List[WarmthLearningAgent]:
    """Create diverse opponents with different behavioral patterns"""
    action_space = WarmthActionSpace(n_bins=sim_config.n_bins)
    opponents = []

    for i, opp_config in enumerate(opponent_configs):
        config = LearningConfig(
            alpha=opp_config.learning_rate,
            gamma=opp_config.discount_factor,
            epsilon=opp_config.exploration_rate,
            risk_sensitivity=opp_config.risk_sensitivity,
            prior_expectation=opp_config.prior_expectation,
            prior_strength=opp_config.prior_strength,
        )
        opponent = WarmthLearningAgent(config, action_space, f"{opp_config.name}_{i}")
        opponent.memory_length = opp_config.memory_length
        opponent.forgetting_rate = opp_config.forgetting_rate
        opponents.append(opponent)

    return opponents


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization"""

    population_size: int = 30
    generations: int = 50
    tournament_rounds: int = 500
    mutation_rate: float = 0.15
    mutation_strength: float = 0.2
    elite_size: int = 2
    param_ranges: Dict[str, Tuple[float, float]] = None
    smoothing_window: int = 5  # Window for smoothing fitness history
    convergence_threshold: float = 0.01  # Threshold for detecting convergence

    def __post_init__(self):
        if self.param_ranges is None:
            self.param_ranges = {
                "alpha": (0.01, 0.5),
                "risk_sensitivity": (0.0, 1.0),
                "prior_expectation": (0.0, 1.0),
                "prior_strength": (1.0, 50.0),
            }


class Individual:
    """Represents a single agent configuration"""

    def __init__(self, params: Dict[str, float]):
        self.params = params
        self.fitness = 0.0
        self.config = LearningConfig(
            alpha=params["alpha"],
            risk_sensitivity=params["risk_sensitivity"],
            prior_expectation=params["prior_expectation"],
            prior_strength=params["prior_strength"],
        )

    @classmethod
    def random(cls, param_ranges: Dict) -> "Individual":
        """Create random individual within parameter ranges"""
        params = {
            key: np.random.uniform(ranges[0], ranges[1])
            for key, ranges in param_ranges.items()
        }
        return cls(params)


class GeneticOptimizer:
    def __init__(self, evolution_config: EvolutionConfig):
        self.config = evolution_config
        self.population: List[Individual] = []
        self.generation = 0
        self.best_history = []  # Track best individuals

    def initialize_population(self):
        """Create initial random population"""
        self.population = [
            Individual.random(self.config.param_ranges)
            for _ in range(self.config.population_size)
        ]

    def tournament_select(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection"""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create child using uniform crossover"""
        child_params = {}
        for param in self.config.param_ranges.keys():
            if np.random.random() < 0.5:
                child_params[param] = parent1.params[param]
            else:
                child_params[param] = parent2.params[param]
        return Individual(child_params)

    def adapt_mutation_rate(self):
        """Reduce mutation as population converges"""
        if len(self.best_history) > self.config.smoothing_window:
            recent_fitness = [
                ind.fitness
                for ind in self.best_history[-self.config.smoothing_window :]
            ]
            fitness_variance = np.var(recent_fitness)

            # Reduce mutation if population is converging
            if fitness_variance < self.config.convergence_threshold:
                self.config.mutation_rate *= 0.95
                self.config.mutation_strength *= 0.95

    def mutate(self, individual: Individual):
        """Mutate with adaptive strength"""
        for param, (min_val, max_val) in self.config.param_ranges.items():
            if np.random.random() < self.config.mutation_rate:
                # Calculate adaptive mutation strength
                param_range = max_val - min_val
                current_value = individual.params[param]

                # Smaller mutations for values near boundaries
                distance_to_boundary = (
                    min(current_value - min_val, max_val - current_value) / param_range
                )

                local_strength = self.config.mutation_strength * distance_to_boundary
                delta = np.random.normal(0, local_strength)
                new_value = current_value + delta * param_range
                individual.params[param] = np.clip(new_value, min_val, max_val)

    def evaluate_fitness(
        self,
        individual: Individual,
        opponents: List[WarmthLearningAgent],
        sim_config: SimulationConfig,
    ) -> float:
        """Evaluate individual's fitness against opponent pool"""
        total_points = 0
        action_space = WarmthActionSpace(n_bins=sim_config.n_bins)
        agent = WarmthLearningAgent(individual.config, action_space, "Evolving")

        for opp in opponents:
            payoffs1, _ = run_simulation(agent, opp, sim_config)
            total_points += sum(payoffs1)

        return total_points

    def evolve(
        self, opponents: List[WarmthLearningAgent], sim_config: SimulationConfig
    ) -> Individual:
        """Run one generation with parameter smoothing"""
        # Evaluate current population
        for ind in self.population:
            ind.fitness = self.evaluate_fitness(ind, opponents, sim_config)

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best = self.population[0]
        self.best_history.append(best)

        # Parameter smoothing for elite individuals
        if len(self.best_history) >= self.config.smoothing_window:
            recent_best = self.best_history[-self.config.smoothing_window :]
            for param in best.params:
                smoothed_value = np.mean([ind.params[param] for ind in recent_best])
                best.params[param] = smoothed_value

        # Create new population with increased elitism
        new_population = []
        new_population.extend(self.population[: self.config.elite_size])

        self.adapt_mutation_rate()

        # Fill rest of population
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_select(self.config.tournament_size)
            parent2 = self.tournament_select(self.config.tournament_size)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        return best


def run_evolution(opponent_configs: List[OpponentConfig]):
    """Run evolution with specified opponent configurations"""
    # Setup configurations
    evo_config = EvolutionConfig()
    sim_config = SimulationConfig(n_rounds=500)

    # Create opponent pool from configs
    opponents = create_opponent_pool(opponent_configs, sim_config)

    optimizer = GeneticOptimizer(evo_config)
    optimizer.initialize_population()

    history = []

    for gen in range(evo_config.generations):
        best = optimizer.evolve(opponents, sim_config)
        history.append(
            {
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": np.mean([ind.fitness for ind in optimizer.population]),
                "best_params": best.params.copy(),
            }
        )

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
    # Define opponent configurations
    opponent_configs = [
        OpponentConfig(
            prior_expectation=0.2,
            prior_strength=30.0,
            name="Cold",
        ),
        OpponentConfig(
            prior_expectation=0.8,
            prior_strength=30.0,
            name="Warm",
        ),
        OpponentConfig(
            prior_expectation=0.5,
            prior_strength=10.0,
            name="Flexible",
        ),
    ]

    # Run evolution with these opponents
    history = run_evolution(opponent_configs)

    # Plot results
    from ml_algo_plot import plot_evolution_history

    fig = plot_evolution_history(history)
    plt.show()
