import pickle
import os
from sklearn.ensemble import VotingClassifier

# Paths to individual models
model_dir = "models"
lr_path = os.path.join(model_dir, "logistic_model.pkl")
rf_path = os.path.join(model_dir, "random_forest_model.pkl")
xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
nn_path = os.path.join(model_dir, "neural_net_model.pkl")

# Load models
with open(lr_path, "rb") as f:
    lr = pickle.load(f)
with open(rf_path, "rb") as f:
    rf = pickle.load(f)
with open(xgb_path, "rb") as f:
    xgb = pickle.load(f)
with open(nn_path, "rb") as f:
    nn = pickle.load(f)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb),
        ('nn', nn)
    ],
    voting='hard'
)

# Save the voting model
voting_model_path = os.path.join(model_dir, "voting_model.pkl")
with open(voting_model_path, "wb") as f:
    pickle.dump(voting_clf, f)

print("✅ Voting model saved successfully!")
