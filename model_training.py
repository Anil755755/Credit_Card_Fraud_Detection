import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE

# Load Dataset
data = pd.read_csv('dataset/label_encoded_credit_card_fraud_dataset.csv')

# Drop Card_Number from features
X = data.drop(columns=['Class', 'Card_Number'])
y = data['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

# Train Models
log_model = LogisticRegression(max_iter=500)
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150)
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

log_model.fit(X_train_resampled, y_train_resampled)
rf_model.fit(X_train_resampled, y_train_resampled)
xgb_model.fit(X_train_resampled, y_train_resampled)
nn_model.fit(X_train_resampled, y_train_resampled)

# Create Stacking Classifier
estimators = [
    ('lr', log_model),
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('nn', nn_model)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X_train_resampled, y_train_resampled)

# Evaluate at different thresholds
y_probs = stacking_model.predict_proba(X_test_scaled)[:, 1]

print("\nEvaluation at different thresholds:")
best_f1 = 0
best_thresh = 0.5

for threshold in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print(f"Threshold: {threshold:.2f} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | AUC-ROC: {auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = threshold

# Save the final model and scaler
os.makedirs('models', exist_ok=True)
with open('models/final_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the best threshold
with open('models/threshold.txt', 'w') as f:
    f.write(str(best_thresh))

print(f"\n✅ Final Model Saved with best threshold = {best_thresh}")
