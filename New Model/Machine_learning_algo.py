from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from Reinforcement_learning import (
    LearningConfig,
    WarmthLearningAgent,
    WarmthActionSpace,
)
from Configurations import SimulationConfig, run_simulation
from opponent_types import OpponentType, OPPONENT_TYPES
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
    prior_strength: Tuple[float, float] = (1.0, 2)


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

        # Add diversity tracking
        self.population_diversity = []

        # Adjust selection pressure
        self.selection_pressure = 2.0  # Higher values mean stronger selection
        self.min_diversity = 0.1  # Minimum population diversity threshold

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

        # Track diversity
        current_diversity = np.mean(
            [self._calculate_diversity(ind) for ind in self.population]
        )
        self.population_diversity.append(current_diversity)

        # Adjust selection pressure based on diversity
        if current_diversity < self.min_diversity:
            self.selection_pressure *= 0.9  # Reduce selection pressure
        else:
            self.selection_pressure = min(2.0, self.selection_pressure * 1.1)

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

    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Modified tournament selection with diversity preservation"""
        if self.generation < 5:  # Early generations
            # Use larger tournaments for more exploration
            tournament_size = max(2, tournament_size - 1)

        tournament = np.random.choice(self.population, tournament_size, replace=False)

        # Calculate diversity contribution
        diversity_scores = [self._calculate_diversity(ind) for ind in tournament]

        # Combine fitness and diversity
        combined_scores = [
            ind.fitness + self.min_diversity * div_score
            for ind, div_score in zip(tournament, diversity_scores)
        ]

        return tournament[np.argmax(combined_scores)]

    def _calculate_diversity(self, individual: Individual) -> float:
        """Calculate how different an individual is from population mean"""
        if not self.population:
            return 0.0

        mean_params = {}
        for param in individual.params:
            values = [ind.params[param] for ind in self.population]
            mean_params[param] = np.mean(values)

        # Calculate Euclidean distance from mean
        distance = (
            sum(
                ((individual.params[param] - mean_params[param]) ** 2)
                for param in individual.params
            )
            ** 0.5
        )

        return distance

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
        n_trials = 3  # Run multiple trials for more stable fitness
        action_space = WarmthActionSpace(n_bins=sim_config.n_bins)

        # Run multiple trials against each opponent
        for opponent in opponents:
            trial_points = []
            for _ in range(n_trials):
                # Create fresh agent for each trial to avoid learning interference
                agent = WarmthLearningAgent(
                    individual.config, action_space, "Evolving_Agent"
                )
                payoffs, _ = run_simulation(agent, opponent, sim_config)
                trial_points.append(sum(payoffs))

            # Use median score from trials against this opponent
            total_points += np.median(trial_points)

        # Calculate average points per opponent
        fitness = total_points / len(opponents)

        # Add stability bonus if performance is consistent
        if len(trial_points) > 1:
            stability = 1 / (1 + np.std(trial_points))
            fitness *= 1 + 0.1 * stability  # 10% bonus for stable performance

        return fitness


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


# Define standard opponent types
OPPONENT_TYPES = {
    "COLD_RIGID": OpponentType(
        name="Cold Rigid",
        description="Consistently cold with strong priors",
        learning_rate=0.1,
        prior_expectation=0.2,
        prior_strength=40.0,
        risk_sensitivity=0.7,
        exploration_rate=0.1,
    ),
    "WARM_FLEXIBLE": OpponentType(
        name="Warm Flexible",
        description="Initially warm but adapts quickly",
        learning_rate=0.3,
        prior_expectation=0.8,
        prior_strength=10.0,
        risk_sensitivity=0.3,
        exploration_rate=0.2,
    ),
    "CAUTIOUS_LEARNER": OpponentType(
        name="Cautious Learner",
        description="Starts neutral, learns slowly, avoids risks",
        learning_rate=0.15,
        prior_expectation=0.5,
        prior_strength=20.0,
        risk_sensitivity=0.6,
        exploration_rate=0.15,
    ),
    "ERRATIC": OpponentType(
        name="Erratic",
        description="Highly exploratory with weak priors",
        learning_rate=0.4,
        prior_expectation=0.5,
        prior_strength=5.0,
        risk_sensitivity=0.2,
        exploration_rate=0.4,
    ),
}

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


def plot_evolution_comparison(experiment_results: Dict[str, List[dict]]):
    """Visualize evolution results across different opponent mixes"""

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig)

    # Fitness evolution plot
    ax1 = fig.add_subplot(gs[0, :])
    for mix_name, history in experiment_results.items():
        generations = [h["generation"] for h in history]
        fitness = [h["best_fitness"] for h in history]
        ax1.plot(generations, fitness, label=f"{mix_name}", linewidth=2)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Evolution of Fitness Across Different Opponent Mixes")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Parameter evolution for each mix
    params = list(
        experiment_results[list(experiment_results.keys())[0]][0]["best_params"].keys()
    )

    for idx, param in enumerate(params):
        ax = fig.add_subplot(gs[1:, idx % 2])
        for mix_name, history in experiment_results.items():
            values = [h["best_params"][param] for h in history]
            ax.plot(generations, values, label=f"{mix_name}", linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{param} Value")
        ax.set_title(f"Evolution of {param}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Add final parameters comparison
    final_params = {
        mix_name: history[-1]["best_params"]
        for mix_name, history in experiment_results.items()
    }

    # Create comparison table
    table_data = []
    for param in params:
        row = [param] + [f"{final_params[mix][param]:.3f}" for mix in final_params]
        table_data.append(row)

    plt.figtext(
        0.5,
        0.02,
        tabulate(
            table_data,
            headers=["Parameter"] + list(final_params.keys()),
            tablefmt="grid",
        ),
        ha="center",
        fontfamily="monospace",
    )

    plt.tight_layout()
    return fig


def plot_opponent_distribution(experiment_results: Dict[str, List[dict]]):
    """Visualize opponent type distributions and corresponding best parameters"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot opponent distributions
    mix_data = []
    for mix_name, history in experiment_results.items():
        mix = history[0]["opponent_mix"]
        for opp_type, prop in mix.items():
            mix_data.append(
                {
                    "Mix": mix_name,
                    "Opponent Type": opp_type,
                    "Proportion": prop,
                }
            )

    sns.barplot(
        data=mix_data,
        x="Mix",
        y="Proportion",
        hue="Opponent Type",
        ax=axes[0],
    )
    axes[0].set_title("Opponent Type Distribution per Mix")

    # Plot final parameters for each mix
    param_data = []
    for mix_name, history in experiment_results.items():
        final_params = history[-1]["best_params"]
        for param, value in final_params.items():
            param_data.append(
                {
                    "Mix": mix_name,
                    "Parameter": param,
                    "Value": value,
                }
            )

    sns.barplot(
        data=param_data,
        x="Mix",
        y="Value",
        hue="Parameter",
        ax=axes[1],
    )
    axes[1].set_title("Best Parameters per Mix")

    plt.tight_layout()
    return fig
