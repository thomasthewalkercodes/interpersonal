from Reinforcement_learning import (
    LearningConfig,
    WarmthActionSpace,
    WarmthLearningAgent,
    interact,
)

# Create configurations
config1 = LearningConfig(alpha=0.1, epsilon=0.2)
config2 = LearningConfig(alpha=0.1, epsilon=0.2)

# Create action spaces
action_space = WarmthActionSpace(n_bins=10)

# Create agents
agent1 = WarmthLearningAgent(config1, action_space, "Agent1")
agent2 = WarmthLearningAgent(config2, action_space, "Agent2")

# Run interaction
payoff1, payoff2 = interact(agent1, agent2)
print(f"Payoffs: {payoff1:.3f}, {payoff2:.3f}")
