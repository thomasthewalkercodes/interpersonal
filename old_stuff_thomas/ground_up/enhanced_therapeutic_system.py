"""
FIXED Visible Therapeutic Evolution System
Simple, robust version that shows exactly how population evolution works.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
from collections import deque, defaultdict
import copy
import json

# Import your working modules
from therapeutic_agent_system import (
    TherapistAgent,
    TherapeuticStrategy,
    TherapeuticPhase,
)


class SimpleVisibleTherapist(TherapistAgent):
    """
    Simplified visible therapist that clearly shows what each strategy does.
    """

    def __init__(self, agent_id: str = "visible_therapist", population_size: int = 6):
        super().__init__(agent_id, population_size)

        # Simple strategy tracking
        self.strategy_performance = {}
        self.strategy_names = []
        self.generation_count = 0

        # Create simple, descriptive names
        self._create_strategy_names()

        print(f"\n🧬 THERAPIST POPULATION CREATED")
        print("=" * 50)
        self._show_initial_strategies()

    def _create_strategy_names(self):
        """Create simple, descriptive names for strategies."""
        base_names = ["Patient", "Bold", "Gradual", "Trusting", "Ambitious", "Balanced"]

        for i in range(len(self.strategy_population)):
            if i < len(base_names):
                name = f"{base_names[i]}_Therapist"
            else:
                name = f"Hybrid_{i}"

            self.strategy_names.append(name)

            # Initialize performance tracking
            self.strategy_performance[i] = {
                "name": name,
                "fitness_history": [],
                "sessions_tested": 0,
                "warm_warm_successes": 0,
                "breakthroughs": 0,
                "avg_patient_progress": 0.0,
                "generation": 0,
                "is_elite": False,
                "parents": [],
            }

    def _show_initial_strategies(self):
        """Show initial strategy characteristics."""
        print("INITIAL STRATEGIES:")
        print("-" * 30)

        for i, strategy in enumerate(self.strategy_population):
            name = self.strategy_names[i]
            print(f"🧠 {name}")
            print(f"   Matching intensity: {strategy.matching_intensity:.3f}")
            print(f"   Trust threshold: {strategy.trust_threshold:.3f}")
            print(f"   Leading step size: {strategy.leading_step_size:.3f}")
            print(f"   Warmth target: {strategy.warmth_target:.3f}")
            print(f"   Patience: {strategy.stabilization_patience} steps")

            # Classify approach
            if strategy.matching_intensity > 0.9:
                approach = "Patient Mirroring"
            elif strategy.leading_step_size > 0.2:
                approach = "Bold Leadership"
            elif strategy.trust_threshold < 0.3:
                approach = "Quick Trust"
            else:
                approach = "Balanced"

            print(f"   Style: {approach}")
            print()

    def evolve_strategies(self, generation_size: int = 30):
        """Simplified evolution with clear tracking."""
        self.generation_count += 1

        print(f"\n🧬 EVOLUTION GENERATION {self.generation_count}")
        print("=" * 50)
        print("Testing all strategies...")

        # Test each strategy
        results = []

        for i, strategy in enumerate(self.strategy_population):
            name = self.strategy_names[i]
            print(f"\n📊 Testing {name}...")

            # Test strategy multiple times
            self.current_strategy = strategy
            total_fitness = 0.0
            warm_warm_count = 0
            breakthrough_count = 0
            progress_total = 0.0

            sessions = max(3, generation_size // len(self.strategy_population))

            for session in range(sessions):
                self._restart_therapy_session()
                fitness, stats = self._test_strategy()

                total_fitness += fitness
                if stats["warm_warm_periods"] > 3:
                    warm_warm_count += 1
                breakthrough_count += stats["breakthroughs"]
                progress_total += stats["patient_progress"]

            avg_fitness = total_fitness / sessions
            avg_progress = progress_total / sessions

            # Update performance tracking
            perf = self.strategy_performance[i]
            perf["fitness_history"].append(avg_fitness)
            perf["sessions_tested"] += sessions
            perf["warm_warm_successes"] += warm_warm_count
            perf["breakthroughs"] += breakthrough_count
            perf["avg_patient_progress"] = avg_progress

            strategy.fitness_score = avg_fitness
            results.append((i, avg_fitness, name))

            print(f"   💯 Fitness: {avg_fitness:.3f}")
            print(f"   🤝 Warm-warm sessions: {warm_warm_count}/{sessions}")
            print(f"   ⚡ Total breakthroughs: {breakthrough_count}")
            print(f"   📈 Avg progress: {avg_progress:.3f}")

        # Sort by fitness
        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🏆 STRATEGY RANKINGS:")
        print("-" * 30)
        for rank, (strategy_id, fitness, name) in enumerate(results):
            emoji = (
                "🥇"
                if rank == 0
                else "🥈" if rank == 1 else "🥉" if rank == 2 else "📊"
            )
            print(f"{emoji} #{rank+1}: {name} (Fitness: {fitness:.3f})")

        # Evolution: keep top strategies
        elite_count = max(2, len(self.strategy_population) // 3)
        elite_indices = [result[0] for result in results[:elite_count]]
        elite_strategies = [self.strategy_population[i] for i in elite_indices]
        elite_names = [self.strategy_names[i] for i in elite_indices]

        print(f"\n🌟 SURVIVORS (Top {elite_count}):")
        for i, name in enumerate(elite_names):
            print(f"   ✅ {name}")
            self.strategy_performance[elite_indices[i]]["is_elite"] = True

        # Create new generation
        print(f"\n🧪 CREATING NEW GENERATION...")

        new_population = elite_strategies.copy()
        new_names = [name + "_evolved" for name in elite_names]
        new_performance = {}

        # Copy performance data for survivors
        for i, old_idx in enumerate(elite_indices):
            new_performance[i] = copy.deepcopy(self.strategy_performance[old_idx])
            new_performance[i]["name"] = new_names[i]
            new_performance[i]["generation"] = self.generation_count

        # Create offspring to fill population
        offspring_count = 0
        while len(new_population) < self.population_size:
            # Select two random elite parents
            parent1 = random.choice(elite_strategies)
            parent2 = random.choice(elite_strategies)

            parent1_idx = elite_strategies.index(parent1)
            parent2_idx = elite_strategies.index(parent2)
            parent1_name = elite_names[parent1_idx]
            parent2_name = elite_names[parent2_idx]

            # Create offspring
            if random.random() < 0.7:  # Crossover
                child = parent1.crossover(parent2)
                child_name = f"Hybrid_{offspring_count+1}"
                parents = [parent1_name, parent2_name]
            else:  # Mutation
                child = parent1.mutate(0.15)
                child_name = f"Evolved_{offspring_count+1}"
                parents = [parent1_name]

            new_population.append(child)
            new_names.append(child_name)

            # Initialize tracking for new strategy
            new_idx = len(new_population) - 1
            new_performance[new_idx] = {
                "name": child_name,
                "fitness_history": [],
                "sessions_tested": 0,
                "warm_warm_successes": 0,
                "breakthroughs": 0,
                "avg_patient_progress": 0.0,
                "generation": self.generation_count,
                "is_elite": False,
                "parents": parents,
            }

            offspring_count += 1
            print(f"   🆕 Created {child_name} (parents: {', '.join(parents)})")

        # Update population
        self.strategy_population = new_population
        self.strategy_names = new_names
        self.strategy_performance = new_performance
        self.current_strategy = self.strategy_population[0]  # Use best strategy

        # Call parent method for compatibility
        try:
            # Store old evolution history before calling parent
            old_evolution_history = self.evolution_history.copy()
            super().evolve_strategies(generation_size)
            # Restore our evolution history
            self.evolution_history = old_evolution_history
        except:
            pass  # Skip if parent method has issues

        print(f"\n✨ GENERATION {self.generation_count} COMPLETE!")
        print(f"   👑 Best strategy: {new_names[0]}")
        print(f"   🧬 Survivors: {elite_count}")
        print(f"   🆕 New offspring: {offspring_count}")

    def _test_strategy(self) -> Tuple[float, Dict]:
        """Test a strategy and return fitness and statistics."""
        # Simple patient simulation
        patient_warmth = 0.2
        patient_trust = 0.1

        warm_warm_periods = 0
        breakthroughs = 0
        initial_warmth = patient_warmth

        for step in range(80):
            # Get therapist action
            therapist_action = self.select_action(
                np.zeros(10), patient_warmth * 2 - 1, patient_trust
            )

            therapist_warmth = (therapist_action + 1) / 2

            # Count warm-warm interactions
            if therapist_warmth > 0.6 and patient_warmth > 0.6:
                warm_warm_periods += 1

            # Count breakthroughs
            if patient_warmth > 0.7 and patient_trust > 0.5:
                breakthroughs += 1

            # Simple patient response
            if patient_trust > 0.2:
                # Patient follows therapist when trust is high
                influence = (therapist_warmth - patient_warmth) * 0.12
                patient_warmth += influence

            # Trust dynamics
            warmth_match = 1.0 - abs(therapist_warmth - patient_warmth)
            trust_change = warmth_match * 0.05 + therapist_warmth * 0.03 - 0.02
            patient_trust += trust_change

            # Add some noise
            patient_warmth += np.random.normal(0, 0.02)
            patient_trust += np.random.normal(0, 0.01)

            # Clip values
            patient_warmth = np.clip(patient_warmth, 0.0, 1.0)
            patient_trust = np.clip(patient_trust, 0.0, 1.0)

        # Calculate fitness
        progress = patient_warmth - initial_warmth
        cooperation_bonus = warm_warm_periods * 0.1
        breakthrough_bonus = breakthroughs * 0.5

        fitness = progress * 3.0 + cooperation_bonus + breakthrough_bonus

        stats = {
            "warm_warm_periods": warm_warm_periods,
            "breakthroughs": breakthroughs,
            "patient_progress": progress,
            "final_warmth": patient_warmth,
            "final_trust": patient_trust,
        }

        return fitness, stats

    def show_current_population(self):
        """Show current population with performance."""
        print(f"\n🧬 CURRENT POPULATION (Generation {self.generation_count})")
        print("=" * 60)

        # Sort by average fitness
        sorted_strategies = []
        for i in range(len(self.strategy_population)):
            perf = self.strategy_performance[i]
            avg_fitness = (
                np.mean(perf["fitness_history"]) if perf["fitness_history"] else 0.0
            )
            sorted_strategies.append((i, avg_fitness, perf))

        sorted_strategies.sort(key=lambda x: x[1], reverse=True)

        for rank, (strategy_id, avg_fitness, perf) in enumerate(sorted_strategies):
            emoji = "👑" if rank == 0 else "⭐" if perf["is_elite"] else "🆕"

            print(f"{emoji} #{rank+1}: {perf['name']}")
            print(f"    Average fitness: {avg_fitness:.3f}")
            print(f"    Sessions tested: {perf['sessions_tested']}")
            if perf["sessions_tested"] > 0:
                success_rate = (
                    perf["warm_warm_successes"] / perf["sessions_tested"]
                ) * 100
                print(f"    Warm-warm success: {success_rate:.1f}%")
                print(f"    Total breakthroughs: {perf['breakthroughs']}")
            print(f"    Generation: {perf['generation']}")
            if perf["parents"]:
                print(f"    Parents: {', '.join(perf['parents'])}")
            print()

    def show_evolution_summary(self):
        """Show summary of evolution progress."""
        if self.generation_count == 0:
            print("No evolution has occurred yet.")
            return

        print(f"\n📈 EVOLUTION SUMMARY")
        print("=" * 40)
        print(f"Generations: {self.generation_count}")

        # Get fitness progression
        all_fitnesses = []
        for gen in range(1, self.generation_count + 1):
            gen_fitnesses = []
            for perf in self.strategy_performance.values():
                if len(perf["fitness_history"]) >= gen:
                    gen_fitnesses.append(perf["fitness_history"][gen - 1])
            if gen_fitnesses:
                all_fitnesses.append(max(gen_fitnesses))

        if len(all_fitnesses) > 1:
            improvement = all_fitnesses[-1] - all_fitnesses[0]
            print(f"Best fitness improvement: {improvement:.3f}")
            print(f"Initial best: {all_fitnesses[0]:.3f}")
            print(f"Current best: {all_fitnesses[-1]:.3f}")

        # Show family tree
        print(f"\nFamily Tree:")
        for perf in self.strategy_performance.values():
            if perf["parents"]:
                print(f"  {perf['name']} ← {', '.join(perf['parents'])}")


def simple_evolution_demo():
    """Run a simple, clear demonstration of evolution."""
    print("🎯 SIMPLE EVOLUTION DEMO")
    print("=" * 50)
    print("This shows exactly how population evolution works!")
    print("Each therapist has different strategies, they compete,")
    print("and the best ones survive to create improved offspring.")
    print("=" * 50)

    # Create simple population
    therapist = SimpleVisibleTherapist(population_size=6)

    input("\n👆 Press Enter to run first evolution...")

    # First evolution
    therapist.evolve_strategies(generation_size=18)  # 3 tests per strategy

    input("\n👆 Press Enter to see evolved population...")
    therapist.show_current_population()

    input("\n👆 Press Enter to run second evolution...")

    # Second evolution
    therapist.evolve_strategies(generation_size=24)  # 4 tests per strategy

    print("\n📊 FINAL RESULTS:")
    therapist.show_current_population()
    therapist.show_evolution_summary()

    return therapist


def compare_population_sizes():
    """Compare different population sizes to show the effect."""
    print("\n🔬 POPULATION SIZE COMPARISON")
    print("=" * 50)
    print("Testing different population sizes to show the effect...")

    sizes = [4, 8, 12]
    results = {}

    for size in sizes:
        print(f"\n🧪 Testing population size {size}...")
        therapist = SimpleVisibleTherapist(population_size=size)

        # Run two quick evolution cycles
        therapist.evolve_strategies(generation_size=size * 3)
        therapist.evolve_strategies(generation_size=size * 3)

        # Get best fitness
        best_fitness = 0.0
        for perf in therapist.strategy_performance.values():
            if perf["fitness_history"]:
                best_fitness = max(best_fitness, max(perf["fitness_history"]))

        results[size] = {
            "best_fitness": best_fitness,
            "diversity": len(
                set(
                    perf["name"].split("_")[0]
                    for perf in therapist.strategy_performance.values()
                )
            ),
            "total_tests": sum(
                perf["sessions_tested"]
                for perf in therapist.strategy_performance.values()
            ),
        }

        print(f"   Best fitness achieved: {best_fitness:.3f}")
        print(f"   Strategy diversity: {results[size]['diversity']}")
        print(f"   Total tests run: {results[size]['total_tests']}")

    print(f"\n📊 POPULATION SIZE COMPARISON RESULTS:")
    print("-" * 40)
    for size, result in results.items():
        print(
            f"Size {size}: Best={result['best_fitness']:.3f}, Diversity={result['diversity']}, Tests={result['total_tests']}"
        )

    print(f"\n💡 INSIGHTS:")
    print("• Larger populations explore more strategies")
    print("• More diversity leads to better solutions")
    print("• But larger populations need more testing time")
    print("• Sweet spot is usually 15-25 for good balance")


def main():
    """Main demo function."""
    print("🧬 UNDERSTANDING THERAPEUTIC POPULATION EVOLUTION")
    print("=" * 60)
    print("This demo shows EXACTLY what population size does in")
    print("the therapeutic agent system and how evolution works!")
    print("=" * 60)

    # Run simple demo
    therapist = simple_evolution_demo()

    print(f"\n🤔 WANT TO COMPARE POPULATION SIZES?")
    try:
        choice = input("Compare different population sizes? (y/n): ").strip().lower()
        if choice in ["y", "yes"]:
            compare_population_sizes()
    except KeyboardInterrupt:
        print("\nSkipping comparison...")

    print(f"\n✨ KEY TAKEAWAYS:")
    print("=" * 30)
    print("🧬 Population = Different therapeutic strategies")
    print("🏆 Competition = Testing who helps patients most")
    print("🔄 Evolution = Best strategies create improved offspring")
    print("📈 Result = Better warm-warm cooperation over time")
    print("🎯 Population size = Exploration vs efficiency tradeoff")

    print(f"\nNow you can see the 'invisible' evolution happening!")
    return therapist


if __name__ == "__main__":
    main()
