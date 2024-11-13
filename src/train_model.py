# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Ensure the models directory exists
os.makedirs("../models", exist_ok=True)

# Load dataset
data = pd.read_csv("../data/posture_data.csv")

# Split features and labels
X = data[["chin_y", "shoulder_avg_y"]]
y = data["label"].map({"straight": 0, "slouched": 1})  # Encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "../models/posture_model.pkl")
