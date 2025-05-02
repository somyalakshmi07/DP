from flask import Flask, render_template, jsonify
import mysql.connector

app = Flask(__name__)

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Megha@22",
    database="new"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-data')
def get_data():
    cursor = db.cursor()
    cursor.execute("SELECT `Order_Tdc`, `O/P_Wt` FROM `25datacsv`")
    results = cursor.fetchall()

    order_tdc = []
    output_wt = []

    for row in results:
        order_tdc.append(str(row[0]))
        try:
            output_wt.append(float(row[1]))
        except (ValueError, TypeError):
            output_wt.append(0)  # fallback if null or bad value

    data = {
        'order_tdc': order_tdc,
        'output_wt': output_wt
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
