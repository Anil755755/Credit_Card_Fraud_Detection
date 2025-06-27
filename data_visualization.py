import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define model paths
model_paths = {
    "Logistic Regression": "models/logistic_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl",
    "Neural Network": "models/neural_net_model.pkl"
}

# Load models
models = {}
for name, path in model_paths.items():
    try:
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    except FileNotFoundError:
        print(f"[⚠️] {name} model not found at: {path}")

# Evaluate each model
for name, model in models.items():
    print(f"\n📊 Evaluating {name}...")

    # Dynamically detect feature count
    n_features = model.n_features_in_

    # Generate synthetic data with the required number of features
    X, y = make_classification(n_samples=1000, n_features=n_features,
                               n_informative=int(n_features * 0.7),
                               n_redundant=n_features - int(n_features * 0.7),
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Predict
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

    # Confusion Matrix + Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'plots/{name}_confusion_matrix.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}_roc_curve.png')
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} - Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{name}_precision_recall.png')
    plt.close()

print("\n✅ All visualizations are generated and saved in the 'plots/' folder.")
