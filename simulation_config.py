"""
Configuration file for Circumplex Model Simulation
This file contains all the parameters that can be easily modified
"""

import numpy as np
from dataclasses import dataclass

# ============================================================================
# QUICK SETTINGS - MODIFY THESE AT THE TOP
# ============================================================================

# SELECT PRESET (choose one: 'balanced', 'borderline', 'avoidant', 'dominant', 'cyclical', 'custom')
PRESET = "avoidant"

# NUMBER OF SIMULATION STEPS
N_STEPS = 300

# ============================================================================
# OCTANT DEFINITIONS
# ============================================================================
self.conflict_elevation = 0.2  # Baseline conflict for all octants
self.conflict_amplitude = 0.8  # Strength of variation
self.conflict_angular_shift = np.pi  # Peak opposite the last behavior

# Names of the 8 octants in the circumplex model
OCTANT_NAMES = [
    "Dominant",  # 0 - Top (90°)
    "Warm-Dominant",  # 1 - Top-Right (45°)
    "Warm",  # 2 - Right (0°)
    "Warm-Submissive",  # 3 - Bottom-Right (315°)
    "Submissive",  # 4 - Bottom (270°)
    "Cold-Submissive",  # 5 - Bottom-Left (225°)
    "Cold",  # 6 - Left (180°)
    "Cold-Dominant",  # 7 - Top-Left (135°)
]

# Opposite pairs (these behaviors conflict with each other)
OCTANT_PAIRS = [
    (0, 4),  # Dominant <-> Submissive
    (1, 5),  # Warm-Dominant <-> Cold-Submissive
    (2, 6),  # Warm <-> Cold
    (3, 7),  # Warm-Submissive <-> Cold-Dominant
]

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    "balanced": {
        "name": "Balanced Personality",
        "description": "Equal availability across all behaviors",
        "availabilities": {
            "Dominant": 0.5,
            "Warm-Dominant": 0.5,
            "Warm": 0.5,
            "Warm-Submissive": 0.5,
            "Submissive": 0.5,
            "Cold-Submissive": 0.5,
            "Cold": 0.5,
            "Cold-Dominant": 0.5,
        },
        "conflicts": {
            "Dominant <-> Submissive": 0.3,
            "Warm-Dominant <-> Cold-Submissive": 0.3,
            "Warm <-> Cold": 0.3,
            "Warm-Submissive <-> Cold-Dominant": 0.3,
        },
        "adjacency_boost": 1.2,
        "learning_rate": 0.1,
    },
    "borderline": {
        "name": "Borderline Pattern",
        "description": "High warm/cold availability with high conflict (vacillating pattern)",
        "availabilities": {
            "Dominant": 0.5,
            "Warm-Dominant": 0.4,
            "Warm": 0.1,
            "Warm-Submissive": 0.4,
            "Submissive": 0.5,
            "Cold-Submissive": 0.4,
            "Cold": 0.1,
            "Cold-Dominant": 0.4,
        },
        "conflicts": {
            "Dominant <-> Submissive": 0.2,
            "Warm-Dominant <-> Cold-Submissive": 0.6,
            "Warm <-> Cold": 0.9,  # Very high conflict between warm and cold
            "Warm-Submissive <-> Cold-Dominant": 0.6,
        },
        "adjacency_boost": 1.1,
        "learning_rate": 0.05,
    },
    "avoidant": {
        "name": "Avoidant Pattern",
        "description": "High cold-submissive tendencies, avoiding warm interactions",
        "availabilities": {
            "Dominant": 0.2,
            "Warm-Dominant": 0.3,
            "Warm": 0.9,
            "Warm-Submissive": 0.3,
            "Submissive": 0.2,
            "Cold-Submissive": 0.3,
            "Cold": 0.9,
            "Cold-Dominant": 0.3,
        },
        "conflicts": {
            "Dominant <-> Submissive": 0.2,
            "Warm-Dominant <-> Cold-Submissive": 0.5,
            "Warm <-> Cold": 1.0,
            "Warm-Submissive <-> Cold-Dominant": 0.5,
        },
        "adjacency_boost": 1.0,
        "learning_rate": 0.1,
    },
    "dominant": {
        "name": "Dominant Pattern",
        "description": "High dominance behaviors, low submissiveness",
        "availabilities": {
            "Dominant": 0.9,
            "Warm-Dominant": 0.7,
            "Warm": 0.3,
            "Warm-Submissive": 0.1,
            "Submissive": 0.1,
            "Cold-Submissive": 0.1,
            "Cold": 0.3,
            "Cold-Dominant": 0.7,
        },
        "conflicts": {
            "Dominant <-> Submissive": 0.6,  # High conflict with submissive behaviors
            "Warm-Dominant <-> Cold-Submissive": 0.4,
            "Warm <-> Cold": 0.3,
            "Warm-Submissive <-> Cold-Dominant": 0.4,
        },
        "adjacency_boost": 1.2,
        "learning_rate": 0.15,
    },
    "cyclical": {
        "name": "Cyclical Pattern",
        "description": "Alternating availability creating unstable patterns",
        "availabilities": {
            "Dominant": 0.6,
            "Warm-Dominant": 0.2,
            "Warm": 0.6,
            "Warm-Submissive": 0.2,
            "Submissive": 0.6,
            "Cold-Submissive": 0.2,
            "Cold": 0.6,
            "Cold-Dominant": 0.2,
        },
        "conflicts": {
            "Dominant <-> Submissive": 0.7,
            "Warm-Dominant <-> Cold-Submissive": 0.7,
            "Warm <-> Cold": 0.7,
            "Warm-Submissive <-> Cold-Dominant": 0.7,
        },
        "adjacency_boost": 1.0,
        "learning_rate": 0.2,
    },
}

# ============================================================================
# CUSTOM CONFIGURATION (used when PRESET = 'custom')
# ============================================================================

CUSTOM_CONFIG = {
    "name": "Custom Configuration",
    "description": "User-defined parameters",
    "availabilities": {
        "Dominant": 0.5,  # Modify these values
        "Warm-Dominant": 0.5,
        "Warm": 0.5,
        "Warm-Submissive": 0.5,
        "Submissive": 0.5,
        "Cold-Submissive": 0.5,
        "Cold": 0.5,
        "Cold-Dominant": 0.5,
    },
    "conflicts": {
        "Dominant <-> Submissive": 0.3,  # Modify these values
        "Warm-Dominant <-> Cold-Submissive": 0.3,
        "Warm <-> Cold": 0.3,
        "Warm-Submissive <-> Cold-Dominant": 0.3,
    },
    "adjacency_boost": 1.2,
    "learning_rate": 0.1,
}

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================


@dataclass
class CircumplexConfig:
    """Configuration container for the simulation"""

    def __init__(self):
        # Select configuration based on PRESET
        if PRESET == "custom":
            config = CUSTOM_CONFIG
        elif PRESET in PRESETS:
            config = PRESETS[PRESET]
        else:
            print(f"Warning: Unknown preset '{PRESET}'. Using 'balanced' instead.")
            config = PRESETS["balanced"]

        # Store configuration name and description
        self.preset_name = config["name"]
        self.description = config["description"]

        # Convert availability dictionary to numpy array in correct order
        self.availabilities = np.array(
            [config["availabilities"][name] for name in OCTANT_NAMES]
        )

        # Convert conflict dictionary to numpy array
        conflict_keys = [
            "Dominant <-> Submissive",
            "Warm-Dominant <-> Cold-Submissive",
            "Warm <-> Cold",
            "Warm-Submissive <-> Cold-Dominant",
        ]
        self.octant_conflicts = np.array(
            [config["conflicts"][key] for key in conflict_keys]
        )

        # Other parameters
        self.adjacency_boost = config["adjacency_boost"]
        self.learning_rate = config["learning_rate"]

        # Number of steps
        self.n_steps = N_STEPS

        # Validate configuration
        self._validate()

        # Print which preset is being used
        print(f"\n{'='*60}")
        print(f"Using preset: {self.preset_name}")
        print(f"Description: {self.description}")
        print(f"Simulation steps: {self.n_steps}")
        print(f"{'='*60}\n")

    def _validate(self):
        """Validate configuration values"""
        # Check availabilities are in valid range
        if not all(0 <= a <= 1 for a in self.availabilities):
            raise ValueError("All availabilities must be between 0 and 1")

        # Check conflicts are in valid range
        if not all(0 <= c <= 1 for c in self.octant_conflicts):
            raise ValueError("All conflicts must be between 0 and 1")

        # Check adjacency boost is positive
        if self.adjacency_boost <= 0:
            raise ValueError("Adjacency boost must be positive")

        # Check learning rate is in valid range
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1")

        # Check n_steps is positive
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")

    def summary(self):
        """Print configuration summary"""
        print("\n" + "=" * 50)
        print(f"CONFIGURATION: {self.preset_name}")
        print("=" * 50)

        print("\nAvailabilities:")
        for i, name in enumerate(OCTANT_NAMES):
            bar = "#" * int(self.availabilities[i] * 20)  # Changed from "█" to "#"
            print(f"  {name:20s}: {self.availabilities[i]:.2f} {bar}")

        print("\nConflicts:")
        for i, (oct1, oct2) in enumerate(OCTANT_PAIRS):
            bar = "#" * int(self.octant_conflicts[i] * 20)  # Changed from "█" to "#"
            print(
                f"  {OCTANT_NAMES[oct1]:12s} <-> {OCTANT_NAMES[oct2]:12s}: {self.octant_conflicts[i]:.2f} {bar}"
            )

        print(f"\nAdjacency Boost: {self.adjacency_boost}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Simulation Steps: {self.n_steps}")


# ============================================================================
# Quick test when running this file directly
# ============================================================================

if __name__ == "__main__":
    print("Testing configuration...")
    config = CircumplexConfig()
    config.summary()
