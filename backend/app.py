from flask import Flask, request, jsonify, render_template
import mysql.connector
from datetime import datetime

app = Flask(__name__)

# MySQL connection settings
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Megha@22',
    'database': 'new'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    from_date = data.get('fromDate')
    to_date = data.get('toDate')
    month = data.get('month')
    order_tdc = data.get('orderTdc')
    financial_year = data.get('financialYear')
    shift = data.get('shift')

    if not order_tdc:
        return jsonify({"error": "Order_Tdc is required"}), 400

    table_name = f"{financial_year}datacsv"

    query = f"SELECT * FROM {table_name} WHERE Order_TDC LIKE %s"
    params = [f"%{order_tdc}%"]

    if from_date:
        query += " AND Start_Date >= %s"
        params.append(from_date)
    if to_date:
        query += " AND End_Date <= %s"
        params.append(to_date)
    if month:
        query += " AND MONTH(Start_Date) = %s"
        params.append(month)
    if shift:
        query += " AND Shift = %s"
        params.append(shift)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

    # Format and rename fields for response
    formatted_results = []
    for row in results:
        formatted_results.append({
            'Row': row['ID'],
            'Start Date': row['Start_Date'].strftime('%Y-%m-%d') if isinstance(row['Start_Date'], datetime) else row['Start_Date'],
            'End Date': row['End_Date'].strftime('%Y-%m-%d') if isinstance(row['End_Date'], datetime) else row['End_Date'],
            'Order_Tdc': row['Order_TDC'],
            'O/P_Wt': row['O/P_Wt'],
            'Shift': row['Shift']
        })

    return jsonify(formatted_results)

if __name__ == '__main__':
    app.run(debug=True)
