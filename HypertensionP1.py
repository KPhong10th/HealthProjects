#Using hypertension data to predict patients with high risk and the most common features to
#contribute.

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import time
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\kevin\OneDrive\Desktop\Data Projects\Health Data\Hypdata.csv')

# Step 1: Handle missing data
for column in ['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate']:
    data[column] = data[column].fillna(data[column].median())
if (data['glucose'].isnull().sum() / len(data)) * 100 > 30:
    data.drop(columns=['glucose'], inplace=True)
else:
    data['glucose'] = data['glucose'].fillna(data['glucose'].median())

# Step 2: Normalize continuous variables
continuous_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']
scaler = MinMaxScaler()
data[continuous_features] = scaler.fit_transform(data[continuous_features])

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Step 3: Prepare data for modeling
X = data.drop(columns=['Risk'])
y = data['Risk']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Step 4: Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate the model
y_val_proba = rf_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.4).astype(int)  # Adjust threshold
print("Classification Report (Threshold Adjusted):")
print(classification_report(y_val, y_val_pred))
print("ROC-AUC Score:", roc_auc_score(y_val, y_val_proba))

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Step 5: Analyze Feature Importance
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top predictors
print("Top Predictors of Hypertension Risk:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Step 6: Optimize Random Forest with RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Use RandomizedSearchCV for faster hyperparameter tuning
start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,  # Sample 10 random combinations
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print(f"Randomized search completed in {time.time() - start_time:.2f} seconds")

# Best model evaluation
best_rf_model = random_search.best_estimator_
y_val_proba = best_rf_model.predict_proba(X_val)[:, 1]
print("Classification Report (Best Random Forest):")
print(classification_report(y_val, (y_val_proba >= 0.4).astype(int)))
print("ROC-AUC Score (Best Random Forest):", roc_auc_score(y_val, y_val_proba))

# Save the best model
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')

# Step 7: Predict for new data
new_data = pd.DataFrame({
    'age': [50],
    'cigsPerDay': [10],
    'totChol': [240],
    'sysBP': [140],
    'diaBP': [90],
    'BMI': [27],
    'heartRate': [75],
    'male': [1],
    'currentSmoker': [1],
    'BPMeds': [0],
    'diabetes': [0],
    'glucose': [100]  # Include glucose if it was used during training
})

# Load scaler and align feature names
scaler = joblib.load('scaler.pkl')
feature_names = X_train.columns
new_data = new_data.reindex(columns=feature_names, fill_value=0)
new_data[continuous_features] = scaler.transform(new_data[continuous_features])

# Predict risk
risk_proba = best_rf_model.predict_proba(new_data)[:, 1]
risk_prediction = (risk_proba >= 0.4).astype(int)
print(f"Predicted Probability of Hypertension Risk: {risk_proba[0]:.2f}")
print(f"Predicted Risk (0 = Low, 1 = High): {risk_prediction[0]}")

