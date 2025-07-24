# interpersonal
Computer Simulation of interpersonal theory

# QLearning
Discretization:

Converts continuous warmth values (0 to 1) into discrete bins
Necessary because Q-learning uses a discrete Q-table
Example: With 10 bins:
0.0-0.1 → bin 0
0.1-0.2 → bin 1
etc.

Action Sampling:

Provides random actions for exploration
Returns random values between 0 and 1
Used when agent decides to explore rather than exploit

State Space Management:

Defines how many discrete states the agent can recognize
More bins = finer granularity but larger Q-table
Fewer bins = coarser but faster learning