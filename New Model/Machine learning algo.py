from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from Reinforcement_learning import (
    LearningConfig,
    WarmthLearningAgent,
    WarmthActionSpace,
)
from Configurations import run_simulation, SimulationConfig


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization"""

    population_size: int = 30
    generations: int = 50
    tournament_rounds: int = 500
    mutation_rate: float = 0.15
    mutation_strength: float = 0.2  # How much to mutate parameters
    elite_size: int = 2  # Number of best individuals to preserve

    # Parameter ranges for evolution
    param_ranges = {
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


class GeneticOptimizer:
    def __init__(self, evolution_config: EvolutionConfig):
        self.config = evolution_config
        self.population: List[Individual] = []
        self.generation = 0

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

    def mutate(self, individual: Individual):
        """Mutate individual parameters"""
        for param, (min_val, max_val) in self.config.param_ranges.items():
            if np.random.random() < self.config.mutation_rate:
                delta = np.random.normal(0, self.config.mutation_strength)
                new_value = individual.params[param] + delta * (max_val - min_val)
                individual.params[param] = np.clip(new_value, min_val, max_val)

    def evolve(
        self, opponents: List[WarmthLearningAgent], sim_config: SimulationConfig
    ):
        """Run one generation of evolution"""
        # Evaluate current population
        for ind in self.population:
            ind.fitness = evaluate_fitness(ind, opponents, sim_config)

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Keep track of best solution
        best = self.population[0]

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        new_population.extend(self.population[: self.config.elite_size])

        # Fill rest of population with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return best


def run_evolution():
    """Main function to run evolutionary optimization"""
    # Setup configurations
    evo_config = EvolutionConfig()
    sim_config = SimulationConfig(n_rounds=500)

    # Create opponent pool
    action_space = WarmthActionSpace(n_bins=sim_config.n_bins)
    opponents = [
        WarmthLearningAgent(
            LearningConfig(prior_expectation=0.2, prior_strength=30.0),
            action_space,
            "Cold Agent",
        ),
        WarmthLearningAgent(
            LearningConfig(prior_expectation=0.8, prior_strength=30.0),
            action_space,
            "Warm Agent",
        ),
    ]

    optimizer = GeneticOptimizer(evo_config)
    optimizer.initialize_population()

    history = []
    print("Starting evolution...")

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

        if gen % 5 == 0:  # Only print every 5 generations
            print(f"Generation {gen}/{evo_config.generations}")

    print("\nEvolution complete!")
    print("\nBest solution found:")
    print("Parameters:", {k: f"{v:.3f}" for k, v in best.params.items()})
    print(f"Fitness: {best.fitness:.2f}")

    return history


if __name__ == "__main__":
    history = run_evolution()

    # Plot results
    from ml_algo_plot import plot_evolution_history

    fig = plot_evolution_history(history)
    plt.show()
