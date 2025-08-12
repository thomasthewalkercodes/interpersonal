from dynamical_analysis_extension import DynamicalSystemsAnalyzer

from interpersonal_dynamics_simulation import InterPersonalSimulation


# Run your existing simulation
sim = InterPersonalSimulation()
sim.add_agent("Agent_Balanced", "depressed")
sim.add_agent("Agent_Balanced2", "aggressive")

sim.run_simulation(n_rounds=200)

# After running your simulation
data = sim.get_results_dataframe()
analyzer = DynamicalSystemsAnalyzer(data=data)

# Generate comprehensive analysis
fig = analyzer.plot_comprehensive_dynamics(save_path="analysis.png")
metrics = analyzer.compute_advanced_metrics()
hypotheses = analyzer.generate_research_hypotheses()
summary = analyzer.create_publication_summary("report.txt")

# Access specific metrics
for dyad, metrics in metrics.items():
    print(f"{dyad}: synchrony = {metrics.get('mood_synchrony', 'N/A'):.3f}")
