from flask import Flask, render_template, request, jsonify, redirect
import sqlite3
import pickle
from datetime import datetime
import os
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('models/final_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Ensure DB and table exists
def init_db():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
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

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input in correct order
        input_data = [
            float(data['Time']),
            float(data['Amount']),
            int(data['Transaction_Type']),
            int(data['Geo_Location']),
            int(data['Merchant_Category']),
            int(data['Is_International']),
            int(data['Is_Recurring']),
            float(data['Velocity']),
            float(data['User_History_Score']),
            int(data['Transaction_Hour']),
            int(data['Location_Mismatch']),
            int(data['Bot_Activity_Flag']),
            float(data['Fraud_Risk_Score'])
        ]

        # Scale input and predict
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        result = "Fraud" if prediction == 1 else "Legit"

        # Save transaction to database
        conn = sqlite3.connect('transactions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (
                Card_Number, Time, Amount, Transaction_Type, Geo_Location,
                Merchant_Category, Is_International, Is_Recurring, Velocity,
                User_History_Score, Transaction_Hour, Location_Mismatch,
                Bot_Activity_Flag, Fraud_Risk_Score, result, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['Card_Number'], data['Time'], data['Amount'], data['Transaction_Type'],
            data['Geo_Location'], data['Merchant_Category'], data['Is_International'],
            data['Is_Recurring'], data['Velocity'], data['User_History_Score'],
            data['Transaction_Hour'], data['Location_Mismatch'],
            data['Bot_Activity_Flag'], data['Fraud_Risk_Score'],
            result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

        return jsonify({'result': result})
    
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)})

@app.route('/history')
def history():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT Card_Number, Time, Amount, Transaction_Type, Geo_Location,
               Merchant_Category, Is_International, Is_Recurring, Velocity,
               User_History_Score, Transaction_Hour, Location_Mismatch,
               Bot_Activity_Flag, Fraud_Risk_Score, result, timestamp
        FROM transactions
        ORDER BY timestamp DESC
        LIMIT 100
    ''')
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=rows)

@app.route('/clear', methods=['POST'])
def clear_history():
    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM transactions')
    conn.commit()
    conn.close()
    return jsonify({'message': 'Transaction history cleared successfully!'})

@app.route('/download', methods=['GET'])
def download_csv():
    import csv
    from flask import Response

    conn = sqlite3.connect('transactions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 100')
    rows = cursor.fetchall()
    conn.close()

    def generate():
        yield ','.join([
            "Card_Number", "Time", "Amount", "Transaction_Type", "Geo_Location",
            "Merchant_Category", "Is_International", "Is_Recurring", "Velocity",
            "User_History_Score", "Transaction_Hour", "Location_Mismatch",
            "Bot_Activity_Flag", "Fraud_Risk_Score", "result", "timestamp"
        ]) + '\n'
        for row in rows:
            yield ','.join(map(str, row)) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=transactions.csv"})

if __name__ == '__main__':
    app.run(debug=True)
