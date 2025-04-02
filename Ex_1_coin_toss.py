import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest


# Define probabilities
P_heads = 0.5  # Null hypothesis (fair coin)
Q_heads = 0.6  # Alternative hypothesis (biased coin)
P_tails = 1 - P_heads
Q_tails = 1 - Q_heads

# Simulate coin tosses
np.random.seed(32123)  # For reproducibility
n_tosses = 100  # Number of coin flips
true_bias = 0.7  # True probability of heads (unknown to the test)
tosses = np.random.choice(['H', 'T'], size=n_tosses, p=[true_bias, 1-true_bias])

# Calculate betting scores
betting_scores = []
cumulative_score = 1  # Start with 1 (neutral)
cumulative_scores = [cumulative_score]

for toss in tosses:
    if toss == 'H':
        S = Q_heads / P_heads
    else:
        S = Q_tails / P_tails
    
    cumulative_score *= S
    betting_scores.append(S)
    cumulative_scores.append(cumulative_score)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(n_tosses+1), cumulative_scores, marker='o', linestyle='-', color='b', alpha=0.7)
plt.axhline(y=1, color='r', linestyle='--', label="Neutral Evidence")
plt.xlabel("Number of Tosses")
plt.ylabel("Cumulative Betting Score")
plt.title("Cumulative Betting Score Over Coin Tosses")
plt.legend()
plt.show()



# Initialize variables for p-value calculations
observed_heads = 0
p_values = []

# Compute p-values at each step
for i, toss in enumerate(tosses):
    if toss == 'H':
        observed_heads += 1
    
    # Perform a binomial test at each step
    p_val = binomtest(observed_heads, i+1, P_heads, alternative='two-sided').pvalue
    p_values.append(p_val)

# Plot Betting Scores and p-values
fig, ax1 = plt.subplots(figsize=(10, 5))

# Betting Score Plot (Left y-axis)
ax1.plot(range(n_tosses+1), cumulative_scores, marker='o', linestyle='-', color='b', alpha=0.7, label="Cumulative Betting Score")
ax1.axhline(y=1, color='r', linestyle='--', label="Neutral Evidence")
ax1.set_xlabel("Number of Tosses")
ax1.set_ylabel("Cumulative Betting Score", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# P-Value Plot (Right y-axis)
ax2 = ax1.twinx()
ax2.plot(range(1, n_tosses+1), p_values, marker='x', linestyle='-', color='g', alpha=0.7, label="P-value")
ax2.axhline(y=0.05, color='purple', linestyle='--', label="Significance Threshold (0.05)")
ax2.set_ylabel("P-value", color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Legends
fig.tight_layout()
fig.legend(loc="upper right")
plt.title("Betting Scores vs P-values Over Coin Tosses")
plt.show()
