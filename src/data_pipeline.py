
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Used for saving/loading the scaler object

def generate_placement_dataset(num_students=2500, random_seed=42):
    np.random.seed(random_seed)

    # Academic Parameters
    cgpa = np.round(np.random.uniform(5.5, 9.8, num_students), 2)
    grade_velocity = np.round(np.random.uniform(-0.6, 0.6, num_students), 2)
    backlogs = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.75, 0.15, 0.07, 0.03])

    # Market Readiness Parameters
    coding_skills_score = np.random.randint(40, 100, size=num_students)
    aptitude_score = np.random.randint(45, 100, size=num_students)
    projects_count = np.random.choice([0, 1, 2, 3, 4], size=num_students, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    certifications = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.4, 0.4, 0.15, 0.05])
    internships = np.random.choice([0, 1, 2], size=num_students, p=[0.6, 0.3, 0.1])

    # Soft Skills & Behavioral Parameters
    communication_score = np.random.randint(40, 100, size=num_students)
    attendance_rate = np.round(np.random.uniform(65, 98, num_students), 1)

    # Socioeconomic Context
    family_income_tier = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.3, 0.5, 0.2])
    region_category = np.random.choice(['Rural', 'Semi-Urban', 'Urban'], size=num_students, p=[0.35, 0.45, 0.2])
    college_tier = np.random.choice([1, 2, 3], size=num_students, p=[0.15, 0.45, 0.4])

    # Base Placement Calculation
    log_odds = (
        (cgpa - 7.0) * 1.5 +
        (grade_velocity * 2.0) -
        (backlogs * 1.8) +
        ((coding_skills_score - 60) / 10) * 0.8 +
        ((aptitude_score - 60) / 10) * 0.5 +
        (projects_count * 0.4) +
        (certifications * 0.5) +
        (internships * 0.9) +
        ((communication_score - 60) / 10) * 0.7 +
        ((attendance_rate - 75) / 10) * 0.3
    )

    # Introduce controlled historical bias
    bias_factor = np.where(family_income_tier == 'Low', -0.5, 0) + np.where(region_category == 'Rural', -0.4, 0)
    log_odds += bias_factor

    # Convert log-odds to placement status (0 or 1)
    prob = 1 / (1 + np.exp(-log_odds))
    placement_status = np.where(prob > 0.5, 1, 0)

    # Compile all generated data into a pandas DataFrame.
    df = pd.DataFrame({
        'CGPA': cgpa,
        'Grade_Velocity': grade_velocity,
        'Backlogs': backlogs,
        'Coding_Skills_Score': coding_skills_score,
        'Aptitude_Score': aptitude_score,
        'Projects_Count': projects_count,
        'Certifications_Count': certifications,
        'Internships_Count': internships,
        'Communication_Score': communication_score,
        'Attendance_Rate': attendance_rate,
        'Family_Income_Tier': family_income_tier,
        'Region_Category': region_category,
        'College_Tier': college_tier,
        'Placement_Status': placement_status
    })
    df['student_id'] = np.arange(1, num_students + 1)

    return df

def run_data_pipeline():
    print("--- Running Data Pipeline ---")
    # Generate raw data
    df_raw = generate_placement_dataset()
    df_raw.to_csv(root_dir / "data" / "raw" / "student_placement_records.csv", index=False)
    print("  Generated and saved raw data to core_files/data/raw/student_placement_records.csv")

    # Load raw data and perform initial inspection (simplified for script)
    df = pd.read_csv(root_dir / "data" / "raw" / "student_placement_records.csv")
    print(f"  Loaded dataset with {df.shape[0]} rows, {df.shape[1]} columns.")

    # Ordinal Encoding for Ordered Categorical Features:
    income_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Family_Income_Tier'] = df['Family_Income_Tier'].map(income_map)

    # One-Hot Encoding for Unordered Categorical Features:
    df = pd.get_dummies(df, columns=['Region_Category'], drop_first=False)

    # Separate Features (X) from the Target Label (y)
    X = df.drop(columns=['Placement_Status'])
    y = df['Placement_Status']

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save unscaled training and testing feature sets for display
    X_train.to_csv(root_dir / "data" / "processed" / "X_train_display.csv", index=False)
    X_test.to_csv(root_dir / "data" / "processed" / "X_test_display.csv", index=False)
    print("  Saved X_train_display.csv and X_test_display.csv to core_files/data/processed/")

    # Define the list of numerical columns that require scaling.
    columns_to_scale = [
        'CGPA', 'Grade_Velocity', 'Backlogs', 'Coding_Skills_Score',
        'Aptitude_Score', 'Projects_Count', 'Certifications_Count',
        'Internships_Count', 'Communication_Score', 'Attendance_Rate'
    ]

    # Create copies of the feature sets, excluding the 'student_id' for scaling.
    X_train_scaled = X_train.drop(columns=['student_id']).copy()
    X_test_scaled = X_test.drop(columns=['student_id']).copy()

    # Initialize the StandardScaler.
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing datasets.
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])

    # Save the processed (scaled) feature arrays and target labels locally.
    X_train_scaled.to_csv(root_dir / "data" / "processed" / "X_train_ready.csv", index=False)
    X_test_scaled.to_csv(root_dir / "data" / "processed" / "X_test_ready.csv", index=False)
    y_train.to_csv(root_dir / "data" / "processed" / "y_train_ready.csv", index=False)
    y_test.to_csv(root_dir / "data" / "processed" / "y_test_ready.csv", index=False)
    print("  Saved scaled data and target labels to core_files/data/processed/")

    # Save the fitted scaler object.
    joblib.dump(scaler, root_dir / "data" / "processed" / 'scaler.pkl')
    print("  Saved scaler.pkl to core_files/data/processed/")
    print("--- Data Pipeline Complete ---")

if __name__ == '__main__':
    # For direct execution, ensure root_dir is defined
    root_dir = Path(__file__).parent.parent
    run_data_pipeline()
