import numpy as np

from nash_calculator import solve_2x2_nash

# Example usage
if __name__ == "__main__":
    # Create example payoff matrices
    A1 = np.array([[3, 1], [0, 2]])  # Player 1's payoff matrix
    A2 = np.array([[2, 1], [0, 3]])  # Player 2's payoff matrix

    # Solve for Nash equilibria
    result = solve_2x2_nash(A1, A2)

    # Display results
    print("\nResults Summary:")
    print("Pure Strategy Equilibria:", result["pure_equilibria"])

    mixed = result["mixed_strategy"]
    if mixed["valid"]:
        print("\nMixed Strategy Equilibrium:")
        print(f"Player 1 plays Up with probability: {mixed['p_star']:.4f}")
        print(f"Player 2 plays Left with probability: {mixed['q_star']:.4f}")
    else:
        print("\nNo valid mixed strategy equilibrium found")
