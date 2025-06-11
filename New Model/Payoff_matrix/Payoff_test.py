from gaussian_payoff import calculate_warmth_payoff

# Example usage
agent1_warmth = 0.8  # Agent 1 being quite warm
agent2_warmth = 0.3  # Agent 2 being rather cold

payoff = calculate_warmth_payoff(agent1_warmth, agent2_warmth)
print(f"Payoff for interaction: {payoff:.3f}")

# You can also adjust sensitivity parameters
custom_payoff = calculate_warmth_payoff(
    w1=0.7,
    w2=0.6,
    alpha=3,  # More sensitive to mismatches
    beta=4,  # Less penalty for rejection
)
