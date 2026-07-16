import joblib
import pandas as pd

# Load model
model = joblib.load("model/disease_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

print("✅ Model Loaded")

# Sample patient data
sample = {
    "Age": 45,
    "Gender": 1,              # Male=1, Female=0 (according to your encoder)
    "Blood_Pressure": 150,
    "Heart_Rate": 90,
    "Cholesterol_Level": 240,
    "BMI": 30.5
}

# Convert to DataFrame
input_df = pd.DataFrame([sample])

# Prediction
pred = model.predict(input_df)

# Decode prediction
disease = target_encoder.inverse_transform(pred)

print("\nPredicted Disease :", disease[0])

# Confidence
prob = model.predict_proba(input_df)

print("Confidence :", round(max(prob[0])*100,2),"%")