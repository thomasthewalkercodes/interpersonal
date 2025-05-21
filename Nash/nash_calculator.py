import nashpy as nash
import numpy as np
from typing import Dict, Optional


def solve_2x2_nash(A1: np.ndarray, A2: np.ndarray) -> Optional[Dict]:
    """Calculate Nash equilibrium for 2x2 game using nashpy"""
    try:
        # Create game and find equilibria
        game = nash.Game(A1, A2)
        equilibria = list(game.support_enumeration())

        if not equilibria:
            print("Error: Failed to calculate Nash equilibrium")
            return None

        # Extract probabilities from first equilibrium
        p_star, q_star = equilibria[0]

        # Display game payoff matrices
        print("\nCombined Payoff Matrix (Player 1, Player 2):")
        print("        Left   Right")
        print(f"Up    ({A1[0,0]}, {A2[0,0]})  ({A1[0,1]}, {A2[0,1]})")
        print(f"Down  ({A1[1,0]}, {A2[1,0]})  ({A1[1,1]}, {A2[1,1]})")
        print()

        return {"mixed_strategy": {"p_star": p_star[0], "q_star": q_star[0]}}

    except Exception as e:
        print(f"Error calculating Nash equilibrium: {str(e)}")
        return None
