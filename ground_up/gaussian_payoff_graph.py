import numpy as np

# alpha = More sensitive to mismatches
# beta = Less penalty for rejection
"""Calculate payoff for warmth interaction between two agents.
    Args:
        w1: warmth of first agent (0-1)
        w2: warmth of second agent (0-1)
        alpha: mismatch penalty factor
        beta: risk factor weight

    Returns:
        float: payoff value
"""


def calculate_warmth_payoff(
    w1: float, w2: float, alpha: float = 4, beta: float = 10
) -> float:
    # Ensure inputs are in valid range
    w1 = np.clip(w1, 0, 1)
    w2 = np.clip(w2, 0, 1)

    # Calculate payoff components
    mismatch = (w1 - w2) ** 2
    warmth_bonus = (w1 + w2) / 2  # Reward for mutual warmth
    risk_factor = w1
    penalty = risk_factor * mismatch * beta
    base_payoff = np.exp(-alpha * mismatch)

    # Add warmth bonus to base payoff
    return base_payoff + warmth_bonus - penalty


# This function here is to visualize the payoff function as a heatmap.
# It is not used in the main simulation but can be called separately to see how the payoff evolves with different warmth values.
def plot_payoff_heatmap(alpha: float = 2, beta: float = 5):
    """Separate function to visualize the payoff function"""
    import matplotlib.pyplot as plt

    w1 = np.linspace(0, 1, 100)
    w2 = np.linspace(0, 1, 100)
    W1, W2 = np.meshgrid(w1, w2)

    P = np.zeros_like(W1)
    for i in range(len(w1)):
        for j in range(len(w2)):
            P[i, j] = calculate_warmth_payoff(W1[i, j], W2[i, j], alpha, beta)

    plt.figure(figsize=(6, 5))
    plt.imshow(P, extent=[0, 1, 0, 1], origin="lower", cmap="bwr")
    plt.colorbar(label="Payoff")
    plt.xlabel("Agent 1's warmth")
    plt.ylabel("Agent 2's warmth")
    plt.title("Warmth Interaction Payoff")
    plt.show()


if __name__ == "__main__":
    # Only show plot when running this file directly
    plot_payoff_heatmap()
