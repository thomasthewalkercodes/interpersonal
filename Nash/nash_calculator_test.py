import numpy as np
from nash_calculator import solve_2x2_nash


def test_nash_equilibrium():
    """Test Nash equilibrium calculator with different game types"""
    # Test cases with different game types
    games = {
        "Prisoner's Dilemma": {
            "A1": np.array([[3, 0], [5, 1]]),
            "A2": np.array([[3, 5], [0, 1]]),
        },
        "Battle of Sexes": {
            "A1": np.array([[3, 0], [0, 2]]),
            "A2": np.array([[2, 0], [0, 3]]),
        },
        "Matching Pennies": {
            "A1": np.array([[1, -1], [-1, 1]]),
            "A2": np.array([[-1, 1], [1, -1]]),
        },
    }

    # Run tests for each game
    for game_name, matrices in games.items():
        print(f"\n=== Testing {game_name} ===")
        result = solve_2x2_nash(matrices["A1"], matrices["A2"])

        if result:
            print(f"Nash Equilibrium found:")
            print(f"p* (Up probability): {result['mixed_strategy']['p_star']:.4f}")
            print(f"q* (Left probability): {result['mixed_strategy']['q_star']:.4f}")
        else:
            print("No Nash equilibrium found")

        print("=" * 40)


if __name__ == "__main__":
    test_nash_equilibrium()
