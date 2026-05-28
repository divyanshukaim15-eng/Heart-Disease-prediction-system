🫀 Heart Disease Predictor — Ensemble v4.2
A high-precision, command-line heart disease risk predictor built with an ensemble of Random Forest and Gradient Boosting classifiers. The model is trained on a large synthetic dataset (50,000 samples) and accepts real patient data at runtime to produce a risk score and diagnosis.

Disclaimer: This tool is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or advice.


Features

Ensemble Model — Combines Random Forest and Gradient Boosting via soft voting for improved accuracy
Large Synthetic Training Set — Generates 50,000 samples with realistic clinical risk logic
Interactive CLI — Guided, validated input collection for patient data
Risk Scoring — Returns both a binary prediction and a continuous probability score
Dataset Export — Saves the generated training data to a CSV file for inspection or reuse


Model Architecture
ComponentAlgorithmRoleModel 1Random Forest (200 estimators)Handles variance, robust to outliersModel 2Gradient Boosting (200 estimators, lr=0.1)Handles bias, excels on hard casesEnsembleSoft Voting ClassifierAverages predicted probabilities
Features are standardized with StandardScaler before training and inference.

Input Features
FeatureTypeRange / OptionsAgeInteger20 – 90SexBinary0 = Female, 1 = MaleChest Pain TypeCategorical0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = AsymptomaticResting Blood Pressure (BP)Integer (mmHg)90 – 220CholesterolInteger (mg/dl)120 – 600Max Heart RateInteger (bpm)60 – 220Exercise Induced AnginaBinary0 = No, 1 = YesFasting Blood SugarInteger (mg/dl)60 – 300 → binarized at 120

Requirements

Python 3.8+
pandas
numpy
scikit-learn

Install dependencies with:
bashpip install pandas numpy scikit-learn

Usage
bashpython heart_disease_predictor.py
The program will:

Generate a 50,000-sample synthetic dataset and save it as heart_disease_dataset_50k.csv
Train the ensemble model (this may take a minute)
Print model accuracy and a classification report
Prompt you to enter patient data interactively
Output a diagnosis and risk probability score

Example Output
--- High-Precision Heart Disease Predictor (Ensemble v4.2) ---
1. Generating massive synthetic medical data with 50000 samples...
2. Preprocessing and splitting data...
3. Training Ensemble Model (Random Forest + Gradient Boosting)...
4. Evaluating Ensemble Performance...

   >>> Ensemble Model Accuracy: 92.34%

==================================================
   PATIENT DIAGNOSTIC RESULT:
   Model Strategy: Ensemble (Gradient Boosting + Random Forest)

   >>> POSITIVE (HIGH RISK of Heart Disease)
   >>> Probability of Disease (Risk Score): 81.47%
==================================================
   Interpretation: The model indicates a very strong likelihood of heart disease.

Output Interpretation
Risk ScoreInterpretation> 70%Very strong likelihood of heart disease50% – 70%Probable risk — further evaluation recommended30% – 50%Moderate, inconclusive risk — further testing advised< 30%Low risk based on the provided features

Project Structure
heart-disease-predictor/
│
├── heart_disease_predictor.py   # Main script
├── heart_disease_dataset_50k.csv  # Auto-generated training dataset (after first run)
└── README.md

How the Synthetic Data Works
The training data is generated using a weighted risk formula that incorporates known clinical risk factors:

Positive correlators: Age, male sex, high cholesterol, high BP, exercise-induced angina, elevated fasting blood sugar
Negative correlators: Higher max heart rate, certain chest pain type encodings
A small Gaussian noise term (σ = 0.8) is applied to simulate real-world variability
The classification threshold is set at the 60th percentile of risk scores, yielding a realistic class distribution


Limitations

Trained entirely on synthetic data — not validated on real clinical datasets (e.g., UCI Heart Disease Dataset)
Binary outcome only — does not model disease severity
Does not account for additional clinical markers (e.g., ST depression, family history, smoking status)
Should not be used for any real medical decision-making
