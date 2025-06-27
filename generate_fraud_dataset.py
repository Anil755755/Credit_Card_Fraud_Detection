import numpy as np
import pandas as pd

# Function to generate a 12-digit card number
def generate_card_number():
    return str(np.random.randint(10**11, 10**12, dtype=np.int64))  # Avoid int32 overflow

# Function to generate amount > 100
def generate_amount():
    return round(np.random.uniform(100, 1000), 2)

n_samples = 1000  # Number of rows

# Dataset generation
data = {
    'Card_Number': [generate_card_number() for _ in range(n_samples)],
    'Time': np.random.randint(0, 24, n_samples),
    'Amount': [generate_amount() for _ in range(n_samples)],
    'Transaction_Type': np.random.choice(['POS', 'Transfer', 'ATM', 'Online'], n_samples),
    'Geo_Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Miami'], n_samples),
    'Merchant_Category': np.random.choice(['Groceries', 'Dining', 'Electronics', 'Fashion', 'Travel'], n_samples),
    'Is_International': np.random.choice([0, 1], n_samples),
    'Is_Recurring': np.random.choice([0, 1], n_samples),
    'Velocity': np.round(np.random.uniform(0, 1, n_samples), 2),
    'User_History_Score': np.round(np.random.uniform(0, 1, n_samples), 2),
    'Transaction_Hour': np.random.randint(0, 24, n_samples),
    'Location_Mismatch': np.random.choice([0, 1], n_samples),
    'Bot_Activity_Flag': np.random.choice([0, 1], n_samples),
    'Fraud_Risk_Score': np.round(np.random.uniform(0, 100, n_samples), 2),
    'Class': np.random.choice([0, 1], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("cleaned_creditcardFraud.csv", index=False)
print("✅ Dataset with 12-digit Card Numbers and Amount > 100 saved as 'cleaned_creditcardFraud.csv'")
