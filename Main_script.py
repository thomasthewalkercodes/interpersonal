from .Nash.nash_calculator import solve_2x2_nash
import numpy as np

from .agent_thinking.Q_learning import (
    QLearningConfig,
    QLearningAgent,
    GameEnvironment,
)

# Example usage
A1 = np.array([[3, 1], [0, 2]])
A2 = np.array([[2, 1], [0, 3]])

# Get Nash equilibrium as initial strategy
nash_result = solve_2x2_nash(A1, A2)
p_init = nash_result["mixed_strategy"]["p_star"]
q_init = nash_result["mixed_strategy"]["q_star"]

# Configure agents
config1 = QLearningConfig(
    alpha=0.1,
    beta=1.0,
    gamma=0.88,
    rho=0.88,
    lambda_val=2.25,
    ema_weight=0.1,
    novelty_weight=0.1,
    novelty_decay=0.5,
    prior_weight=0.3,
    rand_explore=0.1,
)

config2 = QLearningConfig(
    alpha=0.1,
    beta=1.0,
    gamma=0.88,
    rho=0.88,
    lambda_val=2.25,
    ema_weight=0.1,
    novelty_weight=0.1,
    novelty_decay=0.5,
    prior_weight=0.3,
    rand_explore=0.1,
)

# Create agents with Nash equilibrium as initial strategy
agent1 = QLearningAgent(config1, p_init)
agent2 = QLearningAgent(config2, q_init)

# Create game environment
game = GameEnvironment(A1, A2, agent1, agent2)

# Run simulation
n_rounds = 1000
for _ in range(n_rounds):
    action1, action2, payoff1, payoff2 = game.step()
