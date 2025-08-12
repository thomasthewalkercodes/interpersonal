from interpersonal_dynamics_simulation import InterPersonalSimulation
from dynamical_analysis_extension import enhance_existing_simulation

# Your existing simulation setup (unchanged)
sim = InterPersonalSimulation()
sim.add_agent("Agent_Balanced", "balanced")
sim.add_agent("Agent_Anxious", "anxious")
sim.add_agent("Agent_Narcissistic", "narcissistic")

# Run simulation as usual
sim.run_simulation(n_rounds=200)

# NEW: Enhance with dynamical analysis
enhanced_sim = enhance_existing_simulation(sim)
results = enhanced_sim.analyze_dynamics()

# Access results
print(results["summary"])
print("Top 5 hypotheses:")
for h in results["hypotheses"][:5]:
    print(f"  • {h}")
