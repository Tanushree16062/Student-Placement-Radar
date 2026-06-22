
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Assuming root_dir is passed or defined in a config for modular access
root_dir = Path(__file__).parent.parent

def apply_fairness_guardrails():
    print("--- Applying Fairness Guardrails ---")
    xgb_model = joblib.load(root_dir / "data" / "processed" / 'xgb_model.pkl')
    X_test = pd.read_csv(root_dir / "data" / "processed" / "X_test_ready.csv")

    predictions_prob = xgb_model.predict_proba(X_test)[:, 1]

    df_fairness = X_test.copy()
    df_fairness['Placement_Probability'] = predictions_prob

    demographic_cols = ['Family_Income_Tier', 'Region_Category_Rural', 'Region_Category_Semi-Urban', 'Region_Category_Urban']

    print("  Baseline Bias Audit (Before Guardrails):")
    correlations = df_fairness[demographic_cols].corrwith(df_fairness['Placement_Probability'])
    for col, corr_val in correlations.items():
        print(f"    Correlation between {col:28} and Placement Chance: {corr_val:+.4f}")

    neutralized_prob = predictions_prob.copy()

    for col in demographic_cols:
        if col in X_test.columns:
            group_mean_diff = df_fairness.groupby(col)['Placement_Probability'].mean().diff().dropna().values
            if len(group_mean_diff) > 0:
                bias_adjustment = group_mean_diff[0] * 0.45
                neutralized_prob -= (X_test[col] * bias_adjustment)

    neutralized_prob = np.clip(neutralized_prob, 0, 1)
    df_fairness['Fair_Placement_Probability'] = neutralized_prob

    print("
  Post-Guardrail Fairness Audit:")
    new_correlations = df_fairness[demographic_cols].corrwith(df_fairness['Fair_Placement_Probability'])
    for col, corr_val in new_correlations.items():
        print(f"    Correlation between {col:28} and Fair Placement Chance: {corr_val:+.4f}")

    print("--- Fairness Guardrails Applied ---")
    # Visualization code is omitted here for modularity, would be handled by a plotting script/notebook
    return df_fairness # Return modified dataframe for further analysis if needed

if __name__ == '__main__':
    # For direct execution, ensure root_dir is defined
    root_dir = Path(__file__).parent.parent
    df_fairness_result = apply_fairness_guardrails()
    print("
Fairness filter applied, resulting DataFrame available.")
