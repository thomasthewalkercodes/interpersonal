# pip install numpy
# pip install pandas

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class PayoffMatrix:
    """Class to represent the payoff matrices for both players"""

    player1: np.ndarray  # Matrix for Player 1
    player2: np.ndarray  # Matrix for Player 2


class PayoffDisplay:
    """Class responsible for displaying payoff matrices"""

    @staticmethod
    def display_combined_matrix(payoff: PayoffMatrix) -> None:
        print("Combined Payoff Matrix (Player 1, Player 2):")
        combined = [
            [f"({payoff.player1[i, j]}, {payoff.player2[i, j]})" for j in range(2)]
            for i in range(2)
        ]
        df = pd.DataFrame(combined, index=["Up", "Down"], columns=["Left", "Right"])
        print(df)
        print()


class PureStrategyAnalyzer:
    """Class responsible for finding pure strategy Nash equilibria"""

    def __init__(self, payoff: PayoffMatrix):
        self.payoff = payoff
        self.pure_equilibria = np.zeros((2, 2), dtype=bool)
        self._analyze()

    def _analyze(self) -> None:
        a, b = self.payoff.player1[0, 0], self.payoff.player1[0, 1]
        c, d = self.payoff.player1[1, 0], self.payoff.player1[1, 1]
        w, x = self.payoff.player2[0, 0], self.payoff.player2[0, 1]
        y, z = self.payoff.player2[1, 0], self.payoff.player2[1, 1]

        self.pure_equilibria[0, 0] = (a >= c) and (w >= x)  # Up, Left
        self.pure_equilibria[0, 1] = (b >= d) and (x >= w)  # Up, Right
        self.pure_equilibria[1, 0] = (c >= a) and (y >= z)  # Down, Left
        self.pure_equilibria[1, 1] = (d >= b) and (z >= y)  # Down, Right

    def get_pure_equilibria(self) -> List[str]:
        strategies = [
            ("Up", "Left"),
            ("Up", "Right"),
            ("Down", "Left"),
            ("Down", "Right"),
        ]
        return [
            f"({s1}, {s2})"
            for (s1, s2), is_eq in zip(strategies, self.pure_equilibria.flatten())
            if is_eq
        ]


class MixedStrategyAnalyzer:
    """Class responsible for finding mixed strategy Nash equilibria"""

    def __init__(self, payoff: PayoffMatrix):
        self.payoff = payoff
        self.p_star: Optional[float] = None
        self.q_star: Optional[float] = None
        self._calculate_probabilities()

    def _calculate_probabilities(self) -> None:
        """Calculate mixed strategy Nash equilibrium probabilities"""
        try:

            a, b = self.payoff.player1[0, 0], self.payoff.player1[0, 1]
            c, d = self.payoff.player1[1, 0], self.payoff.player1[1, 1]

            w, x = self.payoff.player2[0, 0], self.payoff.player2[0, 1]
            y, z = self.payoff.player2[1, 0], self.payoff.player2[1, 1]
            # Calculate denominators
            denom1 = (a - b) - (c - d)
            denom2 = (w - y) - (x - z)

            # Calculate numerators
            num1 = d - b
            num2 = z - y

            # Calculate q* (probability of Left for Player 2)
            if abs(denom1) < 1e-10:
                self.q_star = 0.5  # Use uniform probability when indifferent
            else:
                self.q_star = num1 / denom1
                if not (0 <= self.q_star <= 1):
                    self.q_star = 0.5

            # Calculate p* (probability of Up for Player 1)
            if abs(denom2) < 1e-10:
                self.p_star = 0.5
            else:
                self.p_star = num2 / denom2  # Should be 1/2 = 0.5
                if not (0 <= self.p_star <= 1):
                    self.p_star = 0.5

            # Debug output
            print(f"Debug - Payoffs:")
            print(f"P1: [[{a},{b}], [{c},{d}]]")
            print(f"P2: [[{w},{x}], [{y},{z}]]")
            print(f"Calculations:")
            print(f"p*: num={num2}, denom={denom2}, result={self.p_star:.4f}")
            print(f"q*: num={num1}, denom={denom1}, result={self.q_star:.4f}")

        except Exception as e:
            print(f"Error calculating Nash equilibrium: {str(e)}")
            self.p_star = None
            self.q_star = None

    def is_valid_mixed_strategy(self) -> bool:
        if self.p_star is None or self.q_star is None:
            return False
        return (0 <= self.p_star <= 1) and (0 <= self.q_star <= 1)

    def calculate_expected_payoffs(self) -> Tuple[float, float]:
        if not self.is_valid_mixed_strategy():
            return 0.0, 0.0

        a, b = self.payoff.player1[0, 0], self.payoff.player1[0, 1]
        c, d = self.payoff.player1[1, 0], self.payoff.player1[1, 1]
        w, x = self.payoff.player2[0, 0], self.payoff.player2[0, 1]
        y, z = self.payoff.player2[1, 0], self.payoff.player2[1, 1]

        p1_payoff = (
            self.p_star * self.q_star * a
            + self.p_star * (1 - self.q_star) * b
            + (1 - self.p_star) * self.q_star * c
            + (1 - self.p_star) * (1 - self.q_star) * d
        )

        p2_payoff = (
            self.p_star * self.q_star * w
            + self.p_star * (1 - self.q_star) * x
            + (1 - self.p_star) * self.q_star * y
            + (1 - self.p_star) * (1 - self.q_star) * z
        )

        return round(p1_payoff, 4), round(p2_payoff, 4)


class NashEquilibriumSolver:
    def __init__(self, payoff: PayoffMatrix):
        self.payoff = payoff
        self.display = PayoffDisplay()
        self.pure_analyzer = PureStrategyAnalyzer(payoff)
        self.mixed_analyzer = MixedStrategyAnalyzer(payoff)

    def solve(self) -> Dict:
        self.display.display_combined_matrix(self.payoff)
        pure_equilibria = self.pure_analyzer.get_pure_equilibria()

        # If pure strategy equilibria exist, return those instead of mixed
        if pure_equilibria:
            # Convert pure strategy equilibrium to probabilities
            # Take the first pure equilibrium if multiple exist
            strategy = pure_equilibria[0].strip("()").split(", ")
            p_star = 1.0 if strategy[0] == "Up" else 0.0
            q_star = 1.0 if strategy[1] == "Left" else 0.0

            return {
                "pure_equilibria": pure_equilibria,
                "mixed_strategy": {
                    "valid": True,
                    "p_star": p_star,
                    "q_star": q_star,
                    "is_pure": True,
                },
            }

        # Only calculate mixed strategy if no pure strategy equilibria exist
        return {
            "pure_equilibria": [],
            "mixed_strategy": {
                "valid": self.mixed_analyzer.is_valid_mixed_strategy(),
                "p_star": self.mixed_analyzer.p_star,
                "q_star": self.mixed_analyzer.q_star,
                "is_pure": False,
            },
        }


def solve_2x2_nash(A1: np.ndarray, A2: np.ndarray) -> dict:
    """Calculate Nash equilibrium for 2x2 game"""
    try:
        # Check matrix dimensions
        if A1.shape != (2, 2) or A2.shape != (2, 2):
            raise ValueError("Invalid matrix dimensions")

        payoff = PayoffMatrix(A1, A2)
        solver = NashEquilibriumSolver(payoff)
        result = solver.solve()

        # Add debug output for equilibrium type
        if result["pure_equilibria"]:
            print("Found pure strategy Nash equilibrium:")
            print(f"Pure equilibria: {result['pure_equilibria']}")
        else:
            print("No pure strategy equilibrium found, using mixed strategy.")

        return {
            "mixed_strategy": {
                "p_star": result["mixed_strategy"]["p_star"],
                "q_star": result["mixed_strategy"]["q_star"],
            }
        }
    except Exception as e:
        print(f"Error calculating Nash equilibrium: {str(e)}")
        return None
