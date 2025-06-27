import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("E:\CREDIT_CARD_FRAUD_DETECTION\dataset\label_encoded_credit_card_fraud_dataset.csv")

# Columns to encode
cols_to_encode = ['Geo_Location', 'Transaction_Type', 'Merchant_Category']

# Apply Label Encoding
le_dict = {}
for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Save the new dataset
df.to_csv("label_encoded_credit_card_fraud_dataset.csv", index=False)

# Optional: Print mappings
for col in le_dict:
    print(f"{col} encoding:", le_dict[col])
