import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# MAE per fold (ใหม่)
df = pd.DataFrame({
    "Random Forest": [221.0989, 184.3111, 237.1839, 213.1226, 171.6968],
    "XGBoost": [225.7870, 197.7079, 255.4876, 228.8657, 187.4416],
    "Extra Trees": [222.9002, 181.6161, 232.6464, 212.2192, 174.1859],
    "SVR": [829.2170, 887.6108, 898.2137, 887.0496, 855.8903],
    "Linear Regression": [278.6218, 244.3515, 282.5577, 257.2065, 233.3579]
})

# Friedman test
stat, p_friedman = friedmanchisquare(
    df["Random Forest"],
    df["XGBoost"],
    df["Extra Trees"],
    df["SVR"],
    df["Linear Regression"]
)

print("Friedman test statistic =", stat)
print("Friedman p-value =", p_friedman)

# Wilcoxon test เปรียบเทียบกับ baseline "Extra Trees"
baseline = "Extra Trees"
comparisons = []
p_values = []

for col in df.columns:
    if col != baseline:
        stat, p = wilcoxon(df[baseline], df[col])
        comparisons.append(f"ET vs {col}")
        p_values.append(p)

# Holm correction
reject, pvals_corrected, _, _ = multipletests(
    p_values, alpha=0.05, method='holm'
)

results = pd.DataFrame({
    "Comparison": comparisons,
    "Raw p-value": p_values,
    "Holm-adjusted p-value": pvals_corrected,
    "Significant (α=0.05)": reject
})

print(results)
