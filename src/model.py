import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming root_dir is passed or defined in a config for modular access
root_dir = Path(__file__).parent.parent

def train_and_evaluate_models():
    print("--- Starting Model Training and Evaluation ---")
    X_train = pd.read_csv(root_dir / "data" / "processed" / "X_train_ready.csv")
    X_test = pd.read_csv(root_dir / "data" / "processed" / "X_test_ready.csv")
    y_train = pd.read_csv(root_dir / "data" / "processed" / "y_train_ready.csv").values.ravel()
    y_test = pd.read_csv(root_dir / "data" / "processed" / "y_test_ready.csv").values.ravel()

    models = {
        "Naïve Bayes (Baseline)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Engine (Primary)": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
    }

    performance_metrics = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        performance_metrics[name] = {"Accuracy": f"{acc*100:.2f}%", "Precision": f"{prec*100:.2f}%", "Recall": f"{rec*100:.2f}%", "F1-Score": f"{f1*100:.2f}%"}
    df_comparison = pd.DataFrame(performance_metrics).T
    print("--- Model Benchmarking Complete ---")
    print(df_comparison)
    return models["XGBoost Engine (Primary)"], X_train, y_train # Return the best model and training data

def perform_cross_validation(model, X_train, y_train):
    print("--- Running 5-Fold Cross-Validation ---")
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring_metrics)
    cv_results = {
        "Accuracy": np.mean(scores['test_accuracy']) * 100,
        "Precision": np.mean(scores['test_precision']) * 100,
        "Recall": np.mean(scores['test_recall']) * 100,
        "F1-Score": np.mean(scores['test_f1']) * 100
    }
    print("--- Cross-Validation Complete ---")
    print(pd.DataFrame([cv_results], index=[model.__class__.__name__]))
    return cv_results

def setup_shap_explainer(xgb_model, X_train):
    print("--- Setting up SHAP Explainer ---")
    joblib.dump(xgb_model, root_dir / "data" / "processed" / 'xgb_model.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    joblib.dump(explainer, root_dir / "data" / "processed" / 'explainer.pkl')
    print("  Saved xgb_model.pkl and explainer.pkl to core_files/data/processed/")
    return explainer

def simulate_streaming_update(xgb_model, num_new_students=200, random_seed=101, phase=1):
    print(f"--- Simulating Streaming Update (Phase {phase}) ---")
    np.random.seed(random_seed)
    X_stream = pd.read_csv(root_dir / "data" / "processed" / "X_test_ready.csv").head(num_new_students).copy()

    if phase == 1: # Market shift towards Certifications
        new_log_odds = (X_stream['Certifications_Count'] * 3.5) + (X_stream['Coding_Skills_Score'] * 1.5)
    elif phase == 2: # Market shift towards Internships
        new_log_odds = (X_stream['Internships_Count'] * 5.0) + (X_stream['Coding_Skills_Score'] * 1.0)

    prob_stream = 1 / (1 + np.exp(-new_log_odds))
    y_stream = np.where(prob_stream > 0.5, 1, 0)

    xgb_model.fit(X_stream, y_stream, xgb_model=xgb_model.get_booster())
    joblib.dump(xgb_model, root_dir / "data" / "processed" / 'xgb_model_updated.pkl')
    print(f"  Model updated and saved as xgb_model_updated.pkl to core_files/data/processed/")
    return xgb_model


if __name__ == '__main__':
    # For direct execution, ensure root_dir is defined
    root_dir = Path(__file__).parent.parent
    
    xgb_model, X_train, y_train = train_and_evaluate_models()
    perform_cross_validation(xgb_model, X_train, y_train)
    explainer = setup_shap_explainer(xgb_model, X_train)
    
    # Simulate dynamic adaptability
    print("
--- Visualizing Dynamic Adaptability ---")
    # Initial model is already xgb_model
    # Simulate market shift 1 (Certifications)
    xgb_model_after_shift1 = simulate_streaming_update(xgb_model, phase=1)
    
    # Simulate market shift 2 (Internships)
    xgb_model_after_shift2 = simulate_streaming_update(xgb_model_after_shift1, phase=2, random_seed=202)

    # Placeholder for visualizations - these would typically be in a separate notebook or dashboard logic
    print("
Visualizations for feature importance before/after shifts would go here (e.g., using matplotlib/seaborn).")
    print("Consider running these parts in a separate notebook that imports functions from src/model.py")
