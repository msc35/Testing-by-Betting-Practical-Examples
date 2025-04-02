import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, binomtest
from statsmodels.stats.multitest import multipletests

# Set random seed for reproducibility
np.random.seed(15)

# Define parameters
n_tests = 20  # Number of hypothesis tests
n_samples_per_group = 100  # Sample size per group
true_effect_tests = 10 # Half the tests have real effects
alpha = 0.05  # Significance threshold

# Generate synthetic data for each test
p_values = []
betting_scores = []
null_p_values = []  # P-values for true nulls
alternative_p_values = []  # P-values for true effects

for i in range(n_tests):
    if i < true_effect_tests:
        # True effect: Sample from different distributions
        group_1 = np.random.normal(loc=0.5, scale=1, size=n_samples_per_group)
        group_2 = np.random.normal(loc=0.8, scale=1, size=n_samples_per_group)
    else:
        # Null hypothesis: Sample from the same distribution
        group_1 = np.random.normal(loc=0.5, scale=1, size=n_samples_per_group)
        group_2 = np.random.normal(loc=0.5, scale=1, size=n_samples_per_group)

    # Perform t-test
    t_stat, p_val = ttest_ind(group_1, group_2)
    p_values.append(p_val)

    # Calculate betting score
    P = 0.50  # Null probability (hypothetical)
    Q = 0.55  # Alternative probability
    S = Q / P if p_val < 0.05 else P / Q  # Adjust betting score based on evidence
    betting_scores.append(S)

    # Store p-values separately for later analysis
    if i < true_effect_tests:
        alternative_p_values.append(p_val)
    else:
        null_p_values.append(p_val)

# Apply multiple testing corrections
bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
fdr_corrected = multipletests(p_values, method='fdr_bh')[1]

# Convert betting scores to cumulative betting score
cumulative_betting_score = np.cumprod(betting_scores)

# Plot p-values and multiple testing corrections
plt.figure(figsize=(12, 6))
plt.scatter(range(1, n_tests + 1), p_values, color='g', label="Raw P-values")
plt.scatter(range(1, n_tests + 1), bonferroni_corrected, color='r', label="Bonferroni Corrected")
plt.scatter(range(1, n_tests + 1), fdr_corrected, color='b', label="FDR Corrected")
plt.axhline(y=alpha, color='black', linestyle='--', label="Alpha = 0.05")
plt.xlabel("Hypothesis Test Number")
plt.ylabel("P-value")
plt.legend()
plt.title("P-values and Multiple Comparison Corrections")
plt.show()

# Plot cumulative betting score
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_tests + 1), cumulative_betting_score, linestyle='-', marker='o', color='purple', label="Cumulative Betting Score")
plt.axhline(y=1, color='r', linestyle='--', label="Neutral Evidence (1)")
plt.xlabel("Hypothesis Test Number")
plt.ylabel("Cumulative Betting Score")
plt.legend()
plt.title("Cumulative Betting Score Across Multiple Tests")
plt.show()
