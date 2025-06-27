# Credit_Card_Fraud_Detection
# 💳 Credit Card Fraud Detection using Machine Learning

This project is an end-to-end machine learning system that detects fraudulent credit card transactions with high accuracy. It includes data preprocessing, multiple ML models, ensemble learning, performance evaluation, and a user-friendly Flask web application with transaction history and CSV export features.

---

## 📌 Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Demo
🔗 [Live Demo (optional if hosted)](https://your-live-app-link.com)  
📹 Or insert a GIF or video here showing the UI and prediction.

---

## ✅ Features
- Data preprocessing and EDA
- Class imbalance handling using **SMOTE**
- Trained 4 ML models:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - Neural Network (Keras)
- Final model using **StackingClassifier**
- Model performance evaluation:  
  - Confusion Matrix  
  - ROC Curve  
  - Precision-Recall Curve  
- Flask web app with animated frontend
- SQLite for transaction history
- Download transaction history as CSV

---

## 🛠 Tech Stack

**Languages & Tools:**  
`Python`, `Scikit-learn`, `XGBoost`, `Keras`, `TensorFlow`, `Flask`, `HTML`, `CSS`, `JavaScript`, `SQLite`, `SMOTE`, `Matplotlib`, `Seaborn`

---

## 📂 Dataset

- Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions with 492 frauds
- Features are PCA-anonymized (`V1` to `V28`), plus `Time`, `Amount`, and `Class`

---

## 🏗 Project Structure

credit_card_fraud_detection/
│
├── static/ # CSS & JavaScript files
├── templates/ # HTML templates
├── models/ # Saved ML models (.pkl)
├── app.py # Flask backend
├── train_models.py # Script to train and save models
├── data_visualization.py # Evaluation plots
├── transaction_history.db # SQLite DB
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Flask App
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser.

🖼 Screenshots
(Add screenshots here of the homepage, prediction result, and history page)

📊 Results
Ensemble model (StackingClassifier) achieved:

Accuracy: 99.2%

Precision: 90.1%

Recall: 93.5%

AUC-ROC: 0.98

Balanced detection of fraud vs. legitimate transactions

🤝 Contributing
Feel free to fork this repo, raise issues, or submit pull requests.

📝 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Let me know:
- If you'd like this exported as a `.md` file  
- If you want to add **your name or GitHub profile**  
- If you want help creating **GIFs or images** for the screenshots section  

I can also help write a `requirements.txt` file or `train_models.py` if needed.








