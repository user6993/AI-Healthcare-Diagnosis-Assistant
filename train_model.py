import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Dataset Path
# ==========================================
DATASET_PATH = "dataset/healthcare_dataset.csv"

# ==========================================
# Check Dataset
# ==========================================
if not os.path.exists(DATASET_PATH):
    print("❌ Dataset Not Found")
    exit()

print("✅ Dataset Found")

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv(DATASET_PATH)

print("✅ Dataset Loaded")
df.fillna(0, inplace=True)
df = pd.read_csv(DATASET_PATH)

df.fillna(0, inplace=True)
print(df.head())

# ==========================================
# Show Columns
# ==========================================
print("\nDataset Columns")
print(df.columns)

# ==========================================
# Remove Patient_ID
# ==========================================
if "Patient_ID" in df.columns:
    df.drop(columns=["Patient_ID"], inplace=True)

# ==========================================
# Encode Gender
# ==========================================
gender_encoder = LabelEncoder()

if "Gender" in df.columns:
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# ==========================================
# Encode Target
# ==========================================
target_encoder = LabelEncoder()

df["Diagnosis"] = target_encoder.fit_transform(df["Diagnosis"])

# ==========================================
# Features & Target
# ==========================================
X = df.drop(columns=[
    "Diagnosis",
    "Treatment_Plan",
    "Follow_Up_Date"
])
y = df["Diagnosis"]

print("✅ Features Prepared")

# ==========================================
# Create model folder
# ==========================================
os.makedirs("model", exist_ok=True)

# ==========================================
# Save Feature Columns
# ==========================================
joblib.dump(list(X.columns), "model/feature_columns.pkl")

# ==========================================
# Save Target Encoder
# ==========================================
joblib.dump(target_encoder, "model/target_encoder.pkl")

print("✅ feature_columns.pkl Saved")
print("✅ target_encoder.pkl Saved")

# ==========================================
# Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("✅ Train Test Split Done")

# ==========================================
# Random Forest Model
# ==========================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training Started...")

model.fit(X_train, y_train)

print("✅ Training Completed")

# ==========================================
# Accuracy
# ==========================================
accuracy = model.score(X_test, y_test)

print(f"\n🎯 Accuracy : {accuracy*100:.2f}%")

# ==========================================
# Save Model
# ==========================================
joblib.dump(model, "model/disease_model.pkl")

print("✅ disease_model.pkl Saved")
print("\n🎉 Everything Completed Successfully")

print(X.select_dtypes(include=["object"]).columns)