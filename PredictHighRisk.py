#Using the data set, predicting patients who might be HIGH RISK and alerts the clinician.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_excel(r'C:\Users\kevin\OneDrive\Desktop\Data Projects\Health Data\cancerdata.xlsx')

# Encode the target variable (e.g., 'Level')
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Select features and target
X = data.drop(columns=['Patient Id', 'Level'])  # Features
y = data['Level']  # Target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = gb_model.predict(X_test)
y_proba = gb_model.predict_proba(X_test)  # Probabilities for all classes

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate the Multi-Class ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
print(f"Multi-Class ROC-AUC Score: {roc_auc:.2f}")

# ROC Curves for each class
fpr = {}
tpr = {}
thresholds = {}

for i in range(y_proba.shape[1]):
    fpr[i], tpr[i], thresholds[i] = roc_curve((y_test == i).astype(int), y_proba[:, i])

# Plot ROC Curves for each class
plt.figure(figsize=(10, 8))
for i in range(y_proba.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} vs Rest (AUC: {roc_auc_score((y_test == i).astype(int), y_proba[:, i]):.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class Classification")
plt.legend()
plt.show()

# Plot Feature Importance
feature_importance = pd.Series(gb_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance', figsize=(10, 6))
plt.ylabel("Importance Score")
plt.show()

# Columns to analyze
top_symptoms = ['Smoking', 'Fatigue', 'Alcohol use']

# Ensure symptoms are numeric or encoded
for symptom in top_symptoms:
    if data[symptom].dtype == 'object':
        data[symptom] = LabelEncoder().fit_transform(data[symptom])

# Ensure the symptoms are present in the dataset
missing_symptoms = [symptom for symptom in top_symptoms if symptom not in data.columns]
if missing_symptoms:
    print(f"Missing columns in the dataset: {missing_symptoms}")
else:
    # Plot average risk levels for each symptom
    for symptom in top_symptoms:
        # Group by symptom and calculate the average risk level
        avg_risk = data.groupby(symptom)['Level'].mean()

        # Plot the results
        plt.figure(figsize=(8, 6))
        avg_risk.plot(kind='bar', title=f"Average Risk Levels by Presence of {symptom}")
        plt.xlabel(f"{symptom} (0 = Absent, 1+ = Present)")
        plt.ylabel("Average Risk Level (Encoded)")
        plt.xticks(rotation=0)
        plt.show()