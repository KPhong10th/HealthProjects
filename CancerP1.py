#Finding the most common symptoms related to HIGH RISK LEVELS.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_excel(r'C:\Users\kevin\OneDrive\Desktop\Data Projects\Health Data\cancerdata.xlsx')

# Check for missing values
print(data.isnull().sum())

# Encode the target variable (e.g., 'Level')
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Select features and target
X = data.drop(columns=['Patient Id', 'Level'])
y = data['Level']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Feature importance from logistic regression coefficients
importance = pd.Series(logistic_model.coef_[0], index=data.drop(columns=['Patient Id', 'Level']).columns)
importance.sort_values(ascending=False, inplace=True)
print("Logistic Regression Feature Importance:\n", importance)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance from Random Forest
rf_importance = pd.Series(rf_model.feature_importances_, index=data.drop(columns=['Patient Id', 'Level']).columns)
rf_importance.sort_values(ascending=False, inplace=True)
print("Random Forest Feature Importance:\n", rf_importance)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize feature importance
plt.figure(figsize=(10, 6))
rf_importance.plot(kind='bar', title='Feature Importance (Random Forest)')
plt.show()


