import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

df.columns = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
              "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
              "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
              "LUNG_CANCER"]

# Encode categorical values
df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# Convert symptom scores from (1/2) → (0/1)
symptoms = df.columns.difference(["GENDER", "AGE", "LUNG_CANCER"])
for col in symptoms:
    df[col] = df[col].map({1: 0, 2: 1})

# Downsample lung cancer cases to balance
pos = df[df['LUNG_CANCER'] == 1]
neg = df[df['LUNG_CANCER'] == 0]
pos_down = pos.sample(n=len(neg), random_state=42)
df_balanced = pd.concat([pos_down, neg]).sample(frac=1, random_state=42)

# Train model
X = df_balanced.drop("LUNG_CANCER", axis=1)
y = df_balanced["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model Accuracy: {acc:.2f}")

# Save model and columns
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/lung_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
