import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

# Load Boston Housing Dataset from OpenML
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Define outcome variable (MEDV - Median Home Value) and predictors
X = df.drop(columns=["MEDV"])  # Features
y = df["MEDV"]  # House price

# Standardize numeric features for better regression stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Add intercept term for OLS regression
X_scaled.insert(0, "Intercept", 1)  # Explicitly adding intercept

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train full OLS model
full_model = sm.OLS(y_train, X_train).fit()

# Compute log-likelihood of full model
log_likelihood_full = full_model.llf

# Extract p-values and coefficients
p_values = full_model.pvalues
coefficients = full_model.params

# Compute Betting Scores for Each Feature (Leave-One-Out Approach)
betting_scores = []
for feature in X.columns:
    X_train_reduced = X_train.drop(columns=[feature])  # Remove one feature
    reduced_model = sm.OLS(y_train, X_train_reduced).fit()
    
    # Compute log-likelihood of reduced model
    log_likelihood_reduced = reduced_model.llf

    # Compute betting score for this feature (Corrected Direction)
    S = np.exp(log_likelihood_full - log_likelihood_reduced)
    betting_scores.append(S)

# Compute General Betting Score
general_betting_score = np.exp(log_likelihood_full)  # Measures overall model strength

# Convert results to DataFrame
results_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients[1:],  # Skip the intercept
    "P-Value": p_values[1:],  # Skip the intercept
    "Betting_Score": betting_scores
})

# Append General Model Results
general_results = pd.DataFrame({
    "Feature": ["General Model"],
    "Coefficient": [None],
    "P-Value": [full_model.f_pvalue],  # Model-wide p-value
    "Betting_Score": [general_betting_score]
})

results_df = pd.concat([results_df, general_results], ignore_index=True)

# Print the results table
print("\n\U0001F3E1 Boston Housing Regression Results (OLS with Betting Scores & P-values):")
print(results_df.to_string(index=False))

# 📊 Bar Plot for Betting Scores (Log-Scaled for Better Visualization)
plt.figure(figsize=(12, 6))
sns.barplot(x="Feature", y=np.log1p(results_df[:-1]["Betting_Score"]), data=results_df[:-1], palette="Blues_r")  # Exclude General Model row
plt.axhline(y=np.log1p(1), color='r', linestyle='--', label="Neutral Betting Score (log(1))")
plt.xlabel("Feature")
plt.ylabel("Log-Scaled Betting Score")
plt.xticks(rotation=90)
plt.legend()
plt.title("Betting Scores for Boston Housing Predictors (Log Scale)")
plt.show()
