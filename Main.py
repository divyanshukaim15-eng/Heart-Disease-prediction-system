import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys 

class HeartDiseasePredictor:
    """
    A high-precision Ensemble Predictor (Random Forest + Gradient Boosting)
    designed to maximize accuracy on heart disease risk factors.
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['Age', 'Sex', 'ChestPainType', 'BP', 'Cholesterol', 'MaxHeartRate', 'ExerciseAngina', 'FastingBloodSugar']

    def generate_large_synthetic_data(self, n_samples=50000):
        """
        Generates a massive synthetic dataset (50,000 samples) with 
        refined signal-to-noise ratio for better model training.
        """
        print(f"1. Generating massive synthetic medical data with {n_samples} samples...")
        np.random.seed(42)
        
        # 1. Age: Random between 20 and 90
        age = np.random.randint(20, 91, n_samples)
        
        # 2. Sex: 0 = Female, 1 = Male 
        sex = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        
        # 3. Chest Pain Type: 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
        cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.2, 0.3, 0.4])
        
        # 4. Resting Blood Pressure (BP)
        trestbps = np.clip((np.random.normal(130, 15, n_samples) + (sex * 5) + (age * 0.2)).astype(int), 90, 220) 
        
        # 5. Cholesterol
        chol = np.clip((np.random.normal(240, 40, n_samples) + (age * 0.5)).astype(int), 120, 600)

        # 6. Max Heart Rate (thalach)
        max_hr = np.clip(220 - age + np.random.normal(0, 10, n_samples), 60, 220).astype(int)

        # 7. Exercise Induced Angina (exang): 0=No, 1=Yes
        exang = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        
        # 8. Fasting Blood Sugar (fbs): 1 if > 120 mg/dl, 0 otherwise
        fbs = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20]) 
        
        # --- REFINED RISK LOGIC ---
        risk_score = (
            (age * 0.05) + (sex * 1.5) + (chol * 0.004) + (trestbps * 0.01) + 
            (exang * 3.0) - (max_hr * 0.015) - (cp * 0.5) + (fbs * 3.5)
        )
        
        # Reduced noise (from 1.0 to 0.8) makes patterns clearer to learn, improving accuracy
        noise = np.random.normal(0, 0.8, n_samples)
        threshold = np.percentile(risk_score, 60) 
        target = (risk_score + noise > threshold).astype(int)
        
        data = pd.DataFrame({
            'Age': age, 'Sex': sex, 'ChestPainType': cp, 'BP': trestbps, 
            'Cholesterol': chol, 'MaxHeartRate': max_hr, 'ExerciseAngina': exang, 
            'FastingBloodSugar': fbs,
            'HeartDisease': target
        })
        
        return data

    def train(self, df):
        print("2. Preprocessing and splitting data...")
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("3. Training Ensemble Model (Random Forest + Gradient Boosting)...")
        print("   This may take a moment as we are training two advanced models simultaneously.")
        
        # Model 1: Random Forest (Good at handling variance)
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        
        # Model 2: Gradient Boosting (Good at handling bias and hard cases)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        
        # Ensemble: Soft Voting (Averages the probabilities of both models)
        self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("4. Evaluating Ensemble Performance...")
        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n   >>> Ensemble Model Accuracy: {accuracy * 100:.2f}%")
        print("\n   >>> Classification Report:")
        print(classification_report(y_test, predictions))

    def predict_patient(self, patient_data):
        """ Predicts heart disease for a single patient dictionary. """
        if self.model is None:
            print("Error: Model not trained yet.")
            return None, None

        # Ensure all required features are present and in order
        input_df = pd.DataFrame([patient_data], columns=self.feature_names)
        
        # Scale the input
        input_scaled = self.scaler.transform(input_df)
        
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]
        
        return prediction, probability

# --- User Input Helper Function ---
def get_user_input():
    """ Collects and validates patient data from the user. """
    print("\n--- Mandatory Patient Data Input ---")
    data = {}

    def get_validated_input(prompt, value_type=int, range_min=None, range_max=None, mapping=None):
        while True:
            try:
                user_input = input(prompt)
                value = value_type(user_input)
                
                if mapping is not None and value not in mapping:
                    raise ValueError("Invalid option selected.")

                if range_min is not None and value < range_min:
                    raise ValueError(f"Value must be at least {range_min}.")
                
                if range_max is not None and value > range_max:
                    raise ValueError(f"Value must be at most {range_max}.")
                
                return value
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
            except Exception:
                print("An unexpected error occurred. Please try again.")

    data['Age'] = get_validated_input("Enter Age (20-90): ", range_min=20, range_max=90)
    data['Sex'] = get_validated_input("Enter Sex (0 for Female, 1 for Male): ", mapping={0, 1})
    
    cp_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
    print("\nChest Pain Types:")
    for key, val in cp_map.items():
        print(f"  {key}: {val}")
    data['ChestPainType'] = get_validated_input("Enter Chest Pain Type (0-3): ", range_min=0, range_max=3)
    
    data['BP'] = get_validated_input("Enter Resting Blood Pressure (BP) in mmHg (90-220): ", range_min=90, range_max=220)
    data['Cholesterol'] = get_validated_input("Enter Cholesterol in mg/dl (120-600): ", range_min=120, range_max=600)
    data['MaxHeartRate'] = get_validated_input("Enter Maximum Heart Rate Achieved (60-220): ", range_min=60, range_max=220)
    data['ExerciseAngina'] = get_validated_input("Exercise Induced Angina (0 for No, 1 for Yes): ", mapping={0, 1})
    
    print("\nFasting Blood Sugar (FBS):")
    print(" (A value > 120 mg/dl is often considered high risk/diabetic, used as a binary feature.)")
    fbs_value = get_validated_input("Enter Fasting Blood Sugar in mg/dl (e.g., 95 or 140): ", range_min=60, range_max=300)
    
    data['FastingBloodSugar'] = 1 if fbs_value > 120 else 0
    
    return data

def main():
    print("--- High-Precision Heart Disease Predictor (Ensemble v4.2) ---")
    
    predictor = HeartDiseasePredictor()
    
    # 1. Generate and Train Model (Using increased 50k dataset)
    df = predictor.generate_large_synthetic_data(n_samples=50000)
    
    # --- SAVE CSV ---
    csv_filename = "heart_disease_dataset_50k.csv"
    print(f"   > Saving generated dataset to '{csv_filename}'...")
    df.to_csv(csv_filename, index=False)
    print("   > Save complete. You can find this file in your current folder.")
    # ----------------
    
    predictor.train(df) 
    
    # 2. Get User Input
    print("\n" + "="*50)
    print("Starting Patient Diagnostic Interface")
    print("="*50)
    
    patient_data = get_user_input()
    
    # 3. Predict
    pred, prob = predictor.predict_patient(patient_data)
    
    if pred is not None:
        result_text = "POSITIVE (HIGH RISK of Heart Disease)" if pred == 1 else "NEGATIVE (LOW RISK of Heart Disease)"
        color = "\033[91m" if pred == 1 else "\033[92m" 
        reset = "\033[0m"
        
        print("\n" + "="*50)
        print(f"   PATIENT DIAGNOSTIC RESULT:")
        print(f"   Model Strategy: Ensemble (Gradient Boosting + Random Forest)")
        print(f"\n   >>> {color}{result_text}{reset}")
        print(f"   >>> Probability of Disease (Risk Score): {prob * 100:.2f}%")
        print("="*50)
        
        if prob > 0.7:
             print("   Interpretation: The model indicates a very strong likelihood of heart disease.")
        elif prob > 0.5:
             print("   Interpretation: The model suggests a probable risk of heart disease.")
        elif prob < 0.3:
             print("   Interpretation: The model indicates a low risk based on the input features.")
        else:
             print("   Interpretation: The model shows moderate, inconclusive risk. Further testing is advised.")
             
        print("\nDisclaimer: This is a statistical prediction and not a substitute for professional medical diagnosis.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction session terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        sys.exit(1)