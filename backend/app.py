from flask import Flask, render_template, jsonify
import mysql.connector

app = Flask(__name__)

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Megha@22",
    database="cgl225"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-data')
def get_data():
    cursor = db.cursor()
    cursor.execute("SELECT `Actual Product`, `Order Tdc`, `O/P Wt` FROM `25datacsv`")
    results = cursor.fetchall()
    
    data = {
        'products': [],
        'order_tdc': [],
        'output_wt': []
    }
    for row in results:
        data['products'].append(row[0])
        data['order_tdc'].append(row[1])
        data['output_wt'].append(row[2])
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
