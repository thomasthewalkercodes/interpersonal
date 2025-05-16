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
            [
                f"({payoff.player1[i, j]}, {payoff.player2[i, j]})"
                for j in range(2)
            ]
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
        a, b = self.payoff.player1[0, 0], self.payoff.player1[0, 1]
        c, d = self.payoff.player1[1, 0], self.payoff.player1[1, 1]
        w, x = self.payoff.player2[0, 0], self.payoff.player2[0, 1]
        y, z = self.payoff.player2[1, 0], self.payoff.player2[1, 1]

        denom1 = (a - b) - (c - d)
        denom2 = (w - y) - (x - z)

        self.q_star = (d - b) / denom1 if abs(denom1) >= 1e-10 else None
        self.p_star = (z - y) / denom2 if abs(denom2) >= 1e-10 else None

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
        return {
            "pure_equilibria": pure_equilibria,
            "mixed_strategy": {
                "valid": self.mixed_analyzer.is_valid_mixed_strategy(),
                "p_star": self.mixed_analyzer.p_star,
                "q_star": self.mixed_analyzer.q_star,
            },
        }


def solve_2x2_nash(A1: np.ndarray, A2: np.ndarray) -> Dict:
    payoff = PayoffMatrix(A1, A2)
    solver = NashEquilibriumSolver(payoff)
    return solver.solve()
