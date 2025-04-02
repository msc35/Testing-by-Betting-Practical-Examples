import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_patients = 100  # Set to 100 observations
true_success_rate = 0.51  # True success rate (adjust as needed)
outcomes = np.random.choice(['Success', 'Failure'], size=n_patients, p=[true_success_rate, 1-true_success_rate])

# Define probabilities for null and alternative hypotheses
P_success = 0.5  # Null hypothesis: 50% success (placebo)
Q_success = 0.55  # Alternative hypothesis: Slightly higher success rate
P_failure = 1 - P_success
Q_failure = 1 - Q_success

# Initialize tracking variables
cumulative_score = 1  # Start neutral
cumulative_scores = [cumulative_score]
observed_successes = 0
p_values = []

# Compute betting scores and p-values dynamically
for i, outcome in enumerate(outcomes):
    if outcome == 'Success':
        S = Q_success / P_success  # Betting score update
        observed_successes += 1
    else:
        S = Q_failure / P_failure

    cumulative_score *= S  # Multiply to update score
    cumulative_scores.append(cumulative_score)

    # Perform binomial test at each step
    p_val = binomtest(observed_successes, i+1, P_success, alternative='greater').pvalue
    p_values.append(p_val)

# Plot Betting Scores and p-values
fig, ax1 = plt.subplots(figsize=(12, 6))

# Betting Score Plot (Left y-axis)
ax1.plot(range(n_patients+1), cumulative_scores, linestyle='-', marker='o', color='b', alpha=0.8, label="Cumulative Betting Score")
ax1.axhline(y=1, color='r', linestyle='--', label="Neutral Evidence")
ax1.set_xlabel("Number of Patients")
ax1.set_ylabel("Cumulative Betting Score", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# P-Value Plot (Right y-axis)
ax2 = ax1.twinx()
ax2.plot(range(1, n_patients+1), p_values, linestyle='-', marker='x', color='g', alpha=0.8, label="P-value")
ax2.axhline(y=0.05, color='purple', linestyle='--', label="Significance Threshold (0.05)")
ax2.set_ylabel("P-value", color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Legends
fig.tight_layout()
fig.legend(loc="upper right")
plt.title(f"Betting Scores vs P-values (Q = {Q_success}, 100 Observations)")
plt.show()
