import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
plt.style.use('seaborn-v0_8')
import warnings
warnings.filterwarnings('ignore')

# Set display options for DataFrame
pd.set_option('display.max_columns', None)

# Load the dataset
file_path = r'C:\Users\kevin\OneDrive\Desktop\Data Projects\Health Data\BMCancer.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Convert diagnosis column to numerical values
df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

# Visualization of correlations
plt.figure(figsize=(30, 20))
sns.heatmap(df.corr(), cmap='viridis', annot=True, mask=np.tril(np.ones_like(df.corr(), dtype=bool)))
plt.xticks(fontsize=25, rotation=90)
plt.yticks(fontsize=25, rotation=0)
plt.show()

# Data preprocessing
scaler = preprocessing.StandardScaler()
X = df.drop(columns='diagnosis').values
X = scaler.fit_transform(X)
Y = df['diagnosis'].values.reshape(-1, 1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models and evaluation metrics
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

models = [LogisticRegression(random_state=42), KNeighborsClassifier(),
          SVC(probability=True, random_state=42), GaussianNB(),
          DecisionTreeClassifier(random_state=42), RandomForestClassifier(random_state=42),
          xgb.XGBClassifier(), AdaBoostClassifier()]
model_names = ['LogisticRegression', 'KNN', 'SVM', 'NaiveBayes', 'DecisionTree', 'RandomForest', 'XGBoost', 'AdaBoostClassifier']

# Evaluate models
auc_scores = []
plt.figure(figsize=(10, 8))
for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    auc_scores.append(auc_score)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Compute and print F1-score, precision, and recall
f1_scores, recall, precision = [], [], []
for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    f1_scores.append(f1)
    recall.append(r)
    precision.append(p)
    print(f'{name}: F1-score = {f1:.3f}, Precision = {p:.3f}, Recall = {r:.3f}')

average_f1_score = np.mean(f1_scores)
print(f'Average F1-score: {average_f1_score:.3f}')
