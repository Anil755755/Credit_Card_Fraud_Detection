import sqlite3

conn = sqlite3.connect('transactions.db')
cursor = conn.cursor()


# Create new table with correct schema (14 inputs + result + timestamp)
cursor.execute('''
    CREATE TABLE transactions (
        Card_Number TEXT,
        Time REAL,
        Amount REAL,
        Transaction_Type INTEGER,
        Geo_Location INTEGER,
        Merchant_Category INTEGER,
        Is_International INTEGER,
        Is_Recurring INTEGER,
        Velocity REAL,
        User_History_Score REAL,
        Transaction_Hour INTEGER,
        Location_Mismatch INTEGER,
        Bot_Activity_Flag INTEGER,
        Fraud_Risk_Score REAL,
        result TEXT,
        timestamp TEXT
    )
''')

conn.commit()
conn.close()

print("✅ Database setup complete.")
