# Abstract classes

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from gaussian_payoff_graph import calculate_warmth_payoff


class PayoffCalculator:
    def gauss(Theself, Theother) -> float:
        """Calculate Gaussian payoff based on warmth values"""
        return calculate_warmth_payoff(Theself, Theother, Theself.alpha, Theother.beta)
