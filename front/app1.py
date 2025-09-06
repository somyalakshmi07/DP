from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify, session, flash
import mysql.connector
from mysql.connector import pooling
import pandas as pd
from io import BytesIO  # This line was incomplete - make sure it has BytesIO
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import joblib
import traceback
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from flask_cors import CORS
import json
from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import re
import uuid
import sqlite3
from datetime import datetime
import pandas as pd
from io import BytesIO
from werkzeug.utils import secure_filename


# Add this configuration at the top of your app, after the imports
# UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new')
# ALLOWED_EXTENSIONS = {'csv'}

# Your upload folder path
UPLOAD_FOLDER = r"C:\ProgramData\MySQL\MySQL Server 8.0\Data\new"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions
# ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'








app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.secret_key = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

# Database configuration with environment variables
db_config =  {
    "host": "localhost",
    "user": "root",  # <-- change to your local MySQL user
    "password": "Megha@2207",  # <-- change to your local MySQL password
    "database": "new",
    "port": 3306
}

# Create connection pool
try:
    db_pool = pooling.MySQLConnectionPool(**db_config)
    print("✅ Database connection pool created successfully")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    db_pool = None

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe conversion functions
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, np.datetime64)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif obj is None:
        return None
    elif not isinstance(obj, (str, int, float, bool)):
        return str(obj)
    else:
        return obj

def init_db():
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (month_year TEXT PRIMARY KEY, available_time REAL)''')
    current_month = datetime.now().strftime("%b%Y")
    c.execute("SELECT COUNT(*) FROM settings WHERE month_year = ?", (current_month,))
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO settings (month_year, available_time) VALUES (?, ?)", 
                 (current_month, 14400))
    conn.commit()
    conn.close()

init_db()

def extract_tdc_value(tdc_string):
    if tdc_string is None or tdc_string == '':
        return 0
    tdc_str = str(tdc_string)
    numbers = re.findall(r'\d+', tdc_str)
    if numbers:
        return float(numbers[0])
    else:
        return float(hash(tdc_str) % 1000)

try:
    with open('productivity_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    target = model_data.get('target', 'Productivity(TPH)')
    if 'label_encoder' in model_data:
        le = model_data['label_encoder']
    elif 'label_encoders' in model_data:
        le = model_data['label_encoders'].get('Actual Product', None)
    else:
        le = None
    print(f"Model loaded with features: {features}")
    print(f"Model loaded with target: {target}")
    feature_mapping = {
        'product_type': 'Product_Type',
        'tdc_value': 'TDC_Value', 
        'thickness': 'CRFH thickness(mm)',
        'zinc': 'Zinc'
    }
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model_data = None
    model = None
    scaler = None
    le = None
    features = []
    target = ""
    feature_mapping = {}

def get_orders_table(month_year):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  unit TEXT,
                  product_type TEXT,
                  tdc TEXT,
                  thickness REAL,
                  zinc REAL,
                  quantity INTEGER,
                  productivity REAL,
                  required_time REAL,
                  booking_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    return table_name

def get_available_time(month_year):
    try:
        conn = sqlite3.connect('production_planning.db')
        c = conn.cursor()
        c.execute("SELECT available_time FROM settings WHERE month_year = ?", (month_year,))
        result = c.fetchone()
        conn.close()
        return safe_float(result[0] if result else None, 14400.0)
    except Exception as e:
        print(f"Error getting available time: {str(e)}")
        return 14400.0

def get_booked_time(month_year):
    table_name = f"orders_{month_year}"
    try:
        conn = sqlite3.connect('production_planning.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not c.fetchone():
            return 0.0
        c.execute(f"SELECT SUM(required_time) FROM {table_name}")
        result = c.fetchone()
        return safe_float(result[0] if result else None, 0.0)
    except Exception as e:
        print(f"Error getting booked time: {str(e)}")
        return 0.0
    finally:
        if 'conn' in locals():
            conn.close()

def update_available_time(month_year, available_time):
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (month_year, available_time) VALUES (?, ?)",
              (month_year, available_time))
    conn.commit()
    conn.close()

def get_all_orders(month_year):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = c.fetchone()
        if not table_exists:
            return []
        c.execute(f"SELECT * FROM {table_name} ORDER BY booking_date DESC")
        columns = [description[0] for description in c.description]
        orders = [dict(zip(columns, row)) for row in c.fetchall()]
        return orders
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return []
    finally:
        conn.close()

def get_order(month_year, order_id):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    try:
        c.execute(f"SELECT * FROM {table_name} WHERE id = ?", (order_id,))
        row = c.fetchone()
        if row:
            columns = [description[0] for description in c.description]
            order = dict(zip(columns, row))
        else:
            order = None
    except:
        order = None
    finally:
        conn.close()
    return order

def update_order(month_year, order_id, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute(f"""UPDATE {table_name} 
                 SET unit = ?, product_type = ?, tdc = ?, thickness = ?, zinc = ?, 
                     quantity = ?, productivity = ?, required_time = ?
                 WHERE id = ?""",
              (unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time, order_id))
    conn.commit()
    conn.close()

def delete_order(month_year, order_id):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute(f"DELETE FROM {table_name} WHERE id = ?", (order_id,))
    conn.commit()
    conn.close()

def add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    try:
        c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      unit TEXT,
                      product_type TEXT,
                      tdc TEXT,
                      thickness REAL,
                      zinc REAL,
                      quantity INTEGER,
                      productivity REAL,
                      required_time REAL,
                      booking_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute(f"""INSERT INTO {table_name} 
                     (unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time))
        conn.commit()
        print(f"Order added successfully to {table_name}")
        return True
    except Exception as e:
        print(f"Error adding order: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_month_data(month_year):
    try:
        available_time = get_available_time(month_year)
        booked_time = get_booked_time(month_year)
        remaining_time = available_time - booked_time
        return {
            'available_time': float(available_time),
            'booked_time': float(booked_time),
            'remaining_time': float(remaining_time)
        }
    except Exception as e:
        print(f"Error in get_month_data: {str(e)}")
        return {
            'available_time': 14400.0,
            'booked_time': 0.0,
            'remaining_time': 14400.0
        }
    
# import csv
# import chardet
# import traceback

# def import_csv_to_mysql(file_path, table_name, action="add"):
#     """
#     Import CSV into MySQL with minimal modifications:
#     - Detects file encoding
#     - Uses the header row as column names (escaped and made unique if necessary)
#     - If action == "update": drop the table first and reload
#     - Inserts all rows as TEXT exactly as read (no cleaning)
#     """
#     conn = None
#     try:
#         conn = db_pool.get_connection()
#         cursor = conn.cursor()

#         # 1) detect encoding
#         with open(file_path, "rb") as f:
#             raw = f.read(100000)  # sample
#             enc = chardet.detect(raw).get("encoding") or "latin1"
#         print(f"[import_csv_to_mysql] using encoding: {enc}")

#         # 2) open CSV with detected encoding
#         with open(file_path, newline='', encoding=enc, errors='replace') as csvfile:
#             reader = csv.reader(csvfile)
#             try:
#                 headers = next(reader)
#             except StopIteration:
#                 raise ValueError("CSV file is empty")

#             # 3) prepare SQL-safe column identifiers (quote with backticks)
#             #    - escape internal backticks by doubling them
#             #    - ensure non-empty
#             #    - ensure uniqueness by appending suffixes if duplicate
#             sanitized_cols = []
#             seen = {}
#             for i, raw_col in enumerate(headers):
#                 col = "" if raw_col is None else str(raw_col)
#                 if col == "":
#                     col = f"col_{i+1}"   # minimal unavoidable rename for empty header
#                 # escape any backticks inside name (MySQL uses backticks for identifiers)
#                 col_escaped = col.replace("`", "``")
#                 base = col_escaped
#                 count = seen.get(base, 0)
#                 if count:
#                     # make unique by suffixing
#                     col_escaped = f"{base}_{count+1}"
#                     seen[base] = count + 1
#                 else:
#                     seen[base] = 1
#                 sanitized_cols.append(col_escaped)

#             # 4) create/drop table depending on action
#             if action == "update":
#                 cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")

#             # Build CREATE TABLE statement (all columns as LONGTEXT to preserve everything)
#             cols_def = ", ".join([f"`{c}` LONGTEXT" for c in sanitized_cols])
#             create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({cols_def})"
#             cursor.execute(create_sql)

#             # 5) prepare insert statement
#             placeholders = ", ".join(["%s"] * len(sanitized_cols))
#             insert_sql = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in sanitized_cols])}) VALUES ({placeholders})"

#             # 6) read and insert rows exactly as-is; pad/truncate rows to match columns
#             batch = []
#             for row in reader:
#                 # row may be shorter/longer than header: pad with '' or truncate
#                 if len(row) < len(sanitized_cols):
#                     row = list(row) + [''] * (len(sanitized_cols) - len(row))
#                 elif len(row) > len(sanitized_cols):
#                     row = row[:len(sanitized_cols)]
#                 # ensure all values are strings (preserve as-is)
#                 row = [None if v is None else str(v) for v in row]
#                 batch.append(row)

#                 # to avoid huge memory, insert in chunks
#                 if len(batch) >= 1000:
#                     cursor.executemany(insert_sql, batch)
#                     batch = []

#             if batch:
#                 cursor.executemany(insert_sql, batch)

#         conn.commit()
#         print(f"[import_csv_to_mysql] Imported {file_path} -> `{table_name}` (rows inserted).")

#     except Exception as e:
#         # log and re-raise so upload route can catch and return error
#         logger.error(f"CSV import failed for {file_path}: {e}")
#         logger.error(traceback.format_exc())
#         raise
#     finally:
#         if conn and conn.is_connected():
#             cursor.close()
#             conn.close()


import csv
import chardet
import traceback

def import_csv_to_mysql(file_path, table_name, action="add"):
    """
    Import CSV into MySQL with minimal modifications:
    - Detects file encoding
    - Uses the header row as column names (backtick-escaped)
    - If action == "update": drop the table first and reload
    - Inserts all rows as-is (NULLs, spaces, duplicates accepted)
    """
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()

        # 1) detect encoding
        with open(file_path, "rb") as f:
            raw = f.read(100000)  # sample
            enc = chardet.detect(raw).get("encoding") or "latin1"
        print(f"[import_csv_to_mysql] using encoding: {enc}")

        # 2) open CSV with detected encoding
        with open(file_path, newline='', encoding=enc, errors='replace') as csvfile:
            reader = csv.reader(csvfile)
            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty")

            # 3) prepare SQL-safe column identifiers (only escape with backticks)
            sanitized_cols = []
            for i, raw_col in enumerate(headers):
                col = "" if raw_col is None else str(raw_col).strip()
                if col == "":
                    col = f"col_{i+1}"   # only for empty header
                col_escaped = col.replace("`", "``")  # escape backticks inside names
                sanitized_cols.append(col_escaped)

            # 4) create/drop table depending on action
            if action == "update":
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")

            # Build CREATE TABLE statement
            # 👉 keep LONGTEXT for all columns (accepts numbers, decimals, NULLs, blanks)
            cols_def = ", ".join([f"`{c}` LONGTEXT" for c in sanitized_cols])
            create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({cols_def})"
            cursor.execute(create_sql)

            # 5) prepare insert statement
            placeholders = ", ".join(["%s"] * len(sanitized_cols))
            insert_sql = f"INSERT INTO `{table_name}` ({', '.join([f'`{c}`' for c in sanitized_cols])}) VALUES ({placeholders})"

            # 6) read and insert rows as-is
            batch = []
            for row in reader:
                # row may be shorter/longer than header → pad/truncate
                if len(row) < len(sanitized_cols):
                    row = list(row) + [None] * (len(sanitized_cols) - len(row))
                elif len(row) > len(sanitized_cols):
                    row = row[:len(sanitized_cols)]
                # keep None as NULL, strings as-is
                batch.append([None if v == "" else v for v in row])

                if len(batch) >= 1000:
                    cursor.executemany(insert_sql, batch)
                    batch = []

            if batch:
                cursor.executemany(insert_sql, batch)

        conn.commit()
        print(f"[import_csv_to_mysql] Imported {file_path} -> `{table_name}` (rows inserted).")

    except Exception as e:
        logger.error(f"CSV import failed for {file_path}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()





@app.route('/order_punch')
def order_punch():
    current_month = datetime.now().strftime("%b%Y")
    available_time = get_available_time(current_month)
    booked_time = get_booked_time(current_month)
    remaining_time = available_time - booked_time
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute("SELECT month_year FROM settings ORDER BY month_year")
    months = [row[0] for row in c.fetchall()]
    conn.close()
    orders = get_all_orders(current_month)
    return render_template('index.html', 
                         available_time=available_time,
                         booked_time=booked_time,
                         remaining_time=remaining_time,
                         current_month=current_month,
                         months=months,
                         orders=orders)
@app.route('/update_available_time', methods=['POST'])
def update_available_time_route():
    month_year = request.form['month_year']
    available_time = float(request.form['available_time'])
    update_available_time(month_year, available_time)
    return jsonify({'success': True, 'message': 'Available time updated successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc = request.form.get('tdc', '')
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        month_year = request.form.get('month_year', datetime.now().strftime("%b%Y"))
        confirm_booking = request.form.get('confirm_booking', 'false').lower() == 'true'
        if not all([unit, product_type, tdc]):
            return jsonify({'error': 'Missing required fields'})
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        tdc_value = extract_tdc_value(tdc)
        if le is not None:
            try:
                product_type_encoded = le.transform([product_type])[0]
            except ValueError:
                product_type_encoded = 0
        else:
            product_type_encoded = hash(product_type) % 1000
        features_dict = {
            'product_type': product_type_encoded,
            'tdc_value': tdc_value,
            'zinc': zinc,
            'thickness': thickness
        }
        mapped_features = {feature_mapping.get(k, k): v for k, v in features_dict.items()}
        features_df = pd.DataFrame([mapped_features], columns=features)
        features_scaled = scaler.transform(features_df)
        productivity = model.predict(features_scaled)[0]
        required_time = (quantity / productivity) * 60 if productivity > 0 else 0
        month_data = get_month_data(month_year)
        can_book = bool(month_data['remaining_time'] >= required_time)
        if confirm_booking and can_book:
            success = add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time)
            if not success:
                return jsonify({'error': 'Failed to save order to database'})
            month_data = get_month_data(month_year)
        response_data = {
            'success': True,
            'unit': unit,
            'product_type': product_type,
            'tdc': tdc,
            'thickness': thickness,
            'zinc': zinc,
            'quantity': quantity,
            'productivity': round(productivity, 2),
            'required_time': round(required_time, 2),
            'available_time': month_data['available_time'],
            'booked_time': month_data['booked_time'],
            'remaining_time': month_data['remaining_time'],
            'can_book': can_book
        }
        return jsonify(response_data)
    except Exception as e:
        import traceback
        error_msg = f"Error in predict: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg})

@app.route('/book_order', methods=['POST'])
def book_order():
    try:
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc = request.form.get('tdc', '')
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        month_year = request.form.get('month_year', datetime.now().strftime("%b%Y"))
        if not all([unit, product_type, tdc]):
            return jsonify({'error': 'Missing required fields'})
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        tdc_value = extract_tdc_value(tdc)
        if le is not None:
            try:
                product_type_encoded = le.transform([product_type])[0]
            except ValueError:
                product_type_encoded = 0
        else:
            product_type_encoded = hash(product_type) % 1000
        features_dict = {
            'product_type': product_type_encoded,
            'tdc_value': tdc_value,
            'zinc': zinc,
            'thickness': thickness
        }
        mapped_features = {feature_mapping.get(k, k): v for k, v in features_dict.items()}
        features_df = pd.DataFrame([mapped_features], columns=features)
        features_scaled = scaler.transform(features_df)
        productivity = model.predict(features_scaled)[0]
        required_time = (quantity / productivity) * 60 if productivity > 0 else 0
        print(f"Booking order: {unit}, {product_type}, {quantity}, {month_year}")
        success = add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time)
        if not success:
            return jsonify({'error': 'Failed to add order to database'})
        month_data = get_month_data(month_year)
        return jsonify({
            'success': True,
            'message': 'Order booked successfully',
            'available_time': month_data['available_time'],
            'booked_time': month_data['booked_time'],
            'remaining_time': month_data['remaining_time']
        })
    except Exception as e:
        print(f"Error booking order: {e}")
        return jsonify({'error': str(e)})

@app.route('/get_orders/<month_year>')
def get_orders(month_year):
    try:
        orders = get_all_orders(month_year)
        available_time = float(get_available_time(month_year))
        booked_time = float(get_booked_time(month_year))
        remaining_time = float(available_time - booked_time)
        print(f"Found {len(orders)} orders for {month_year}")
        serializable_orders = [convert_to_serializable(order) for order in orders]
        return jsonify({
            'orders': serializable_orders,
            'available_time': available_time,
            'booked_time': booked_time,
            'remaining_time': remaining_time
        })
    except Exception as e:
        print(f"Error in get_orders route: {e}")
        return jsonify({
            'error': f'Failed to fetch orders: {str(e)}',
            'orders': [],
            'available_time': 0,
            'booked_time': 0,
            'remaining_time': 0
        }), 500

@app.route('/get_order/<month_year>/<int:order_id>')
def get_order_route(month_year, order_id):
    order = get_order(month_year, order_id)
    if order:
        serializable_order = convert_to_serializable(order)
        return jsonify(serializable_order)
    else:
        return jsonify({'error': 'Order not found'}), 404

@app.route('/update_order/<month_year>/<int:order_id>', methods=['POST'])
def update_order_route(month_year, order_id):
    try:
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc_input = request.form.get('tdc', '')
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        if not all([unit, product_type, tdc_input]):
            return jsonify({'error': 'Missing required fields'})
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        tdc_value = extract_tdc_value(tdc_input)
        if le is not None:
            try:
                product_type_encoded = le.transform([product_type])[0]
            except ValueError:
                product_type_encoded = 0
        else:
            product_type_encoded = hash(product_type) % 1000
        features_dict = {
            'product_type': product_type_encoded,
            'tdc_value': tdc_value,
            'zinc': zinc,
            'thickness': thickness
        }
        mapped_features = {feature_mapping.get(k, k): v for k, v in features_dict.items()}
        features_df = pd.DataFrame([mapped_features], columns=features)
        features_scaled = scaler.transform(features_df)
        productivity = model.predict(features_scaled)[0]
        required_time = (quantity / productivity) * 60 if productivity > 0 else 0
        update_order(month_year, order_id, unit, product_type, tdc_input, thickness, zinc, quantity, productivity, required_time)
        month_data = get_month_data(month_year)
        return jsonify({
            'success': True,
            'message': 'Order updated successfully',
            'productivity': productivity,
            'required_time': required_time,
            'available_time': month_data['available_time'],
            'booked_time': month_data['booked_time'],
            'remaining_time': month_data['remaining_time']
        })
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/delete_order/<month_year>/<int:order_id>', methods=['DELETE'])
def delete_order_route(month_year, order_id):
    delete_order(month_year, order_id)
    month_data = get_month_data(month_year)
    orders = get_all_orders(month_year)
    serializable_orders = [convert_to_serializable(order) for order in orders]
    return jsonify({
        'success': True,
        'available_time': month_data['available_time'],
        'booked_time': month_data['booked_time'],
        'remaining_time': month_data['remaining_time'],
        'orders': serializable_orders
    })

@app.route('/export_orders/<month_year>')
def export_orders(month_year):
    orders = get_all_orders(month_year)
    df = pd.DataFrame(orders)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f'Orders_{month_year}', index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'orders_{month_year}.xlsx'
    )

@app.route('/get_available_months')
def get_available_months():
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute("SELECT month_year FROM settings ORDER BY month_year")
    months = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify({'months': months})
# Helper functions
def encode_product_type(product_type):
    """Placeholder for encoding product_type. Implement your logic here."""
    # Example: return 1 if product_type == 'GI' else 0
    logger.warning(f"Using placeholder for encode_product_type for: {product_type}")
    return 0

def encode_tdc(tdc):
    """Placeholder for encoding tdc. Implement your logic here."""
    # Example: return a numeric representation of the tdc code
    logger.warning(f"Using placeholder for encode_tdc for: {tdc}")
    return 0

def calculate_available_time():
    """Calculate available time based on current month"""
    now = datetime.now()
    days_in_month = (datetime(now.year, now.month % 12 + 1, 1) - 
                    datetime(now.year, now.month, 1)).days
    total_hours = days_in_month * 24
    shutdown = 24  # Example fixed shutdown time
    setup = 12     # Example setup time
    utilization = 0.98  # 98%
    return round((total_hours - shutdown - setup) * utilization, 2)

def create_table_if_not_exists(cursor, table_name):
    """Create orders table if it doesn't exist"""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            product_type VARCHAR(50),
            tdc VARCHAR(50),
            thickness FLOAT,
            width FLOAT,
            zinc_coating VARCHAR(50),
            quantity INT,
            booked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            required_time FLOAT,
            productivity FLOAT
        )
    """)

def get_month_table(cursor):
    """Get or create table name for current month"""
    now = datetime.now()
    month_year = now.strftime('%B_%Y').lower()
    create_table_if_not_exists(cursor, month_year)
    return month_year

def get_time_stats(cursor, table):
    """Calculate time statistics"""
    cursor.execute(f"SELECT SUM(required_time) FROM `{table}`")
    booked_time = cursor.fetchone()['SUM(required_time)'] or 0
    available_time = calculate_available_time()
    return {
        'available_time': available_time,
        'booked_time': round(booked_time, 2),
        'left_time': round(available_time - booked_time, 2)
    }

# Health check endpoint
@app.route('/health')
def health():
    try:
        if db_pool:
            connection = db_pool.get_connection()
            connection.ping(reconnect=True)
            connection.close()
            return jsonify({"status": "healthy", "database": "connected"})
        return jsonify({"status": "healthy", "database": "not connected"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to home page
    if 'user_id' in session:
        return redirect(url_for('index'))  # or url_for('home') if you have that route

    if db_pool is None:
        flash('Database connection is not available. Please contact admin.', 'error')
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = None
        try:
            conn = db_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()

            if not user:
                flash('User does not exist. Please register first.', 'error')
                return render_template('login.html')

            if check_password_hash(user['password_hash'], password):
                # Set session variables
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Login successful!', 'success')
                
                # Redirect to index route which will serve home.html
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'error')
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
        finally:
            if conn and hasattr(conn, 'is_connected') and conn.is_connected():
                conn.close()

    return render_template('login.html')

@app.before_request
def require_login():
    allowed_routes = ['login', 'register', 'static', 'health']
    if request.endpoint not in allowed_routes and 'user_id' not in session:
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

def create_tables_if_not_exist(cursor):
    """Create all required tables if they don't exist"""
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create your existing orders table
    now = datetime.now()
    month_year = now.strftime('%B_%Y').lower()
    create_table_if_not_exists(cursor, month_year)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate inputs
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        conn = None
        try:
            conn = db_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Check if username exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash('Username already exists', 'danger')
                return render_template('register.html')
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash)
            )
            conn.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration', 'danger')
            return render_template('register.html')
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    return render_template('register.html')

# Main routes
@app.route('/')
@app.route('/index')
@app.route('/home')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session.get('username', 'User'))

@app.route('/production')
def production():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('production.html')

@app.route('/productivity')
def productivity():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("productivity.html")


@app.route('/export', methods=['POST'])
def export_data():
    conn = None
    cursor = None
    try:
        filters = request.get_json()
        logger.info(f"Export request with filters: {filters}")

        unit = filters.get('unit')
        financial_year = filters.get('financialYear')
        shift = filters.get('shift')
        order_tdc = filters.get('orderTdc')
        month = filters.get('month')
        from_date = filters.get('fromDate')
        to_date = filters.get('toDate')

        # Validate compulsory filters
        if not unit or not financial_year:
            return jsonify({"error": "Unit and Financial Year are required."}), 400

        # Determine table name
        fy_number = financial_year[2:]
        if unit.lower() == 'cgl-2':
            table_name = f"{fy_number}datacsv"
        elif unit.lower() == 'cgl-3':
            table_name = f"{fy_number}datacgl"
        else:
            return jsonify({"error": "Invalid unit selected."}), 400

        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if table exists
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        if not cursor.fetchone():
            return jsonify({"error": f"Table {table_name} not found"}), 404

        # Build query parts (same as search function)
        query_parts = []
        params = []

        # DATE RANGE FILTERING
        if from_date and not to_date:
            day = from_date[8:10]
            month_part = from_date[5:7]
            year = from_date[0:4]
            from_date_ddmmyyyy = f"{day}-{month_part}-{year}"
            query_parts.append("STR_TO_DATE(`Start Date`, '%d-%m-%Y') >= STR_TO_DATE(%s, '%d-%m-%Y')")
            params.append(from_date_ddmmyyyy)

        elif from_date and to_date:
            day_from = from_date[8:10]
            month_from = from_date[5:7]
            year_from = from_date[0:4]
            from_date_ddmmyyyy = f"{day_from}-{month_from}-{year_from}"
            
            day_to = to_date[8:10]
            month_to = to_date[5:7]
            year_to = to_date[0:4]
            to_date_ddmmyyyy = f"{day_to}-{month_to}-{year_to}"
            
            query_parts.append("STR_TO_DATE(`Start Date`, '%d-%m-%Y') BETWEEN STR_TO_DATE(%s, '%d-%m-%Y') AND STR_TO_DATE(%s, '%d-%m-%Y')")
            params.extend([from_date_ddmmyyyy, to_date_ddmmyyyy])

        # MONTH FILTER
        elif month and not from_date and not to_date:
            month_str = str(month).zfill(2)
            query_parts.append("SUBSTRING(`Start Date`, 4, 2) = %s")
            params.append(month_str)

        # ORDER TDC FILTER
        if order_tdc:
            query_parts.append("`Order Tdc` LIKE %s")
            params.append(f"%{order_tdc}%")

        # SHIFT FILTER
        if shift:
            query_parts.append("`Shift` = %s")
            params.append(shift)

        # Build the query
        base_query = f"SELECT * FROM `{table_name}`"
        if query_parts:
            base_query += " WHERE " + " AND ".join(query_parts)
        base_query += " ORDER BY STR_TO_DATE(`Start Date`, '%d-%m-%Y') DESC, `Start Time` DESC"

        cursor.execute(base_query, tuple(params))
        rows = cursor.fetchall()

        if not rows:
            return jsonify({"error": "No data found for the selected filters"}), 404

        # Create Excel file
        output = BytesIO()
        
        # Create a simple Excel file using pandas
        df = pd.DataFrame(rows)
        
        # Convert any datetime objects to strings
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create Excel writer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Create a sheet for filter info
            filter_info = pd.DataFrame({
                'Filter': ['Unit', 'Financial Year', 'Shift', 'Order TDC', 'Month', 'From Date', 'To Date', 'Export Date', 'Total Records'],
                'Value': [
                    unit,
                    financial_year,
                    shift if shift else 'All',
                    order_tdc if order_tdc else 'All', 
                    month if month else 'All',
                    from_date if from_date else 'Not specified',
                    to_date if to_date else 'Not specified',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(rows)
                ]
            })
            
            filter_info.to_excel(writer, sheet_name='Filter Info', index=False)
            
            # Write main data
            df.to_excel(writer, sheet_name='Production Data', index=False)
            
            # Add header for production data sheet
            worksheet = writer.sheets['Production Data']
            worksheet.cell(row=1, column=1, value='PRODUCTION DATA')
            
        output.seek(0)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"Production_Data_{unit}_{financial_year}_{timestamp}.xlsx"

        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        logger.error(f"Export Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Export failed: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
@app.route('/search', methods=['POST'])
def search_data():
    conn = None
    cursor = None
    try:
        filters = request.get_json()
        logger.info(f"Received search request: {filters}")

        unit = filters.get('unit')
        financial_year = filters.get('financialYear')
        shift = filters.get('shift')
        order_tdc = filters.get('orderTdc')
        month = filters.get('month')
        from_date = filters.get('fromDate')
        to_date = filters.get('toDate')

        # Validate compulsory filters
        if not unit:
            return jsonify({"error": "Unit is a required filter."}), 400
        if not financial_year:
            return jsonify({"error": "Financial Year is a required filter."}), 400

        # Determine table name
        fy_number = financial_year[2:]
        if unit.lower() == 'cgl-2':
            table_name = f"{fy_number}datacsv"
        elif unit.lower() == 'cgl-3':
            table_name = f"{fy_number}datacgl"
        else:
            return jsonify({"error": "Invalid unit selected."}), 400

        # Check if table exists
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True, buffered=True)
        
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        if not cursor.fetchone():
            return jsonify([])

        logger.info(f"Using table: {table_name}")

        # Get the original column order from the database
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        columns_info = cursor.fetchall()
        original_column_order = [col['Field'] for col in columns_info]
        
        # Build SELECT clause with original column order
        select_columns = ", ".join([f"`{col}`" for col in original_column_order])

        # Build query parts
        query_parts = []
        params = []

        # DATE RANGE FILTERING (fromDate only)
        if from_date and not to_date:
            # Convert yyyy-mm-dd format to dd-mm-yyyy for comparison
            day = from_date[8:10]
            month_part = from_date[5:7]
            year = from_date[0:4]
            from_date_ddmmyyyy = f"{day}-{month_part}-{year}"
            
            query_parts.append("STR_TO_DATE(`Start Date`, '%d-%m-%Y') >= STR_TO_DATE(%s, '%d-%m-%Y')")
            params.append(from_date_ddmmyyyy)

        # DATE RANGE FILTERING (both fromDate and toDate)
        elif from_date and to_date:
            day_from = from_date[8:10]
            month_from = from_date[5:7]
            year_from = from_date[0:4]
            from_date_ddmmyyyy = f"{day_from}-{month_from}-{year_from}"
            
            day_to = to_date[8:10]
            month_to = to_date[5:7]
            year_to = to_date[0:4]
            to_date_ddmmyyyy = f"{day_to}-{month_to}-{year_to}"
            
            query_parts.append("STR_TO_DATE(`Start Date`, '%d-%m-%Y') BETWEEN STR_TO_DATE(%s, '%d-%m-%Y') AND STR_TO_DATE(%s, '%d-%m-%Y')")
            params.extend([from_date_ddmmyyyy, to_date_ddmmyyyy])

        # MONTH FILTER (only if no date range is specified)
        elif month and not from_date and not to_date:
            month_str = str(month).zfill(2)
            query_parts.append("SUBSTRING(`Start Date`, 4, 2) = %s")
            params.append(month_str)

        # ORDER TDC FILTER
        if order_tdc:
            query_parts.append("`Order Tdc` LIKE %s")
            params.append(f"%{order_tdc}%")

        # SHIFT FILTER
        if shift:
            query_parts.append("`Shift` = %s")
            params.append(shift)

        # Build the query with original column order
        base_query = f"SELECT {select_columns} FROM `{table_name}`"
        if query_parts:
            base_query += " WHERE " + " AND ".join(query_parts)
        base_query += " ORDER BY STR_TO_DATE(`Start Date`, '%d-%m-%Y') DESC, `Start Time` DESC LIMIT 500"

        logger.info(f"Executing query: {base_query}")
        logger.info(f"With parameters: {params}")

        cursor.execute(base_query, tuple(params))
        rows = cursor.fetchall()

        # Convert datetime to string for JSON and preserve column order
        results = []
        for row in rows:
            ordered_row = {}
            # Ensure we iterate in the original column order
            for column_name in original_column_order:
                if column_name in row:  # Check if column exists in the result
                    value = row[column_name]
                    if isinstance(value, datetime):
                        ordered_row[column_name] = value.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(value, bytes):
                        ordered_row[column_name] = value.decode('latin1')
                    else:
                        ordered_row[column_name] = value
                else:
                    # Handle case where column might not be in result
                    ordered_row[column_name] = None
            results.append(ordered_row)

        logger.info(f"Found {len(results)} results from table {table_name}")
        return jsonify(results)

    except mysql.connector.Error as err:
        logger.error(f"MySQL Error: {err}")
        return jsonify([])
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return jsonify([])
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
@app.route('/summary', methods=['GET'])
def summary():
    conn = None
    try:
        fy = request.args.get('fy')  # e.g., FY25
        actual_product = request.args.get('actual_product')  # e.g., GI, or All

        if not fy or not fy.startswith("FY"):
            return "Invalid FY value", 400

        if not actual_product:
            return "Missing actual_product filter", 400

        fy_number = fy[2:]  # FY25 → 25
        table_name = f"{fy_number}datacsv"

        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)

        # Products to include when "All" is selected
        product_list = ['GI', 'GL', 'PPGL', 'ZM']

        # Build query
        if actual_product == "All":
            format_strings = ','.join(['%s'] * len(product_list))
            query = f"""
                SELECT 
                    `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` IN ({format_strings})
                GROUP BY `Actual Product`
                ORDER BY FIELD(`Actual Product`, {format_strings})
            """
            values = product_list * 2  # For both WHERE and ORDER BY FIELD
            cursor.execute(query, values)
            results = cursor.fetchall()

            # Grand total
            total_query = f"""
                SELECT 
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` IN ({format_strings})
            """
            cursor.execute(total_query, product_list)
            total = cursor.fetchone()
            total["Actual Product"] = "Grand Total"
            results.append(total)

        else:
            query = f"""
                SELECT 
                    `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` = %s
                GROUP BY `Actual Product`
            """
            cursor.execute(query, (actual_product,))
            results = cursor.fetchall()

            total_query = f"""
                SELECT 
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` = %s
            """
            cursor.execute(total_query, (actual_product,))
            total = cursor.fetchone()
            total["Actual Product"] = "Grand Total"
            results.append(total)

        return render_template("summary.html", data=results, fy=fy, actual_product=actual_product)

    except Exception as e:
        return f"Error occurred: {str(e)}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/export-summary', methods=['POST'])
def export_summary():
    conn = None
    try:
        fy = request.form.get('fy')
        actual_product = request.form.get('actual_product')

        if not fy or not actual_product:
            return "Missing filters", 400

        table_name = f"{fy[-2:]}datacsv"
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)

        product_list = ['GI', 'GL', 'PPGL', 'ZM']

        if actual_product == "All":
            format_strings = ','.join(['%s'] * len(product_list))
            query = f"""
                SELECT 
                    `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` IN ({format_strings})
                GROUP BY `Actual Product`
                ORDER BY FIELD(`Actual Product`, {format_strings})
            """
            values = product_list * 2
            cursor.execute(query, values)
            rows = cursor.fetchall()

            # Grand Total
            total_query = f"""
                SELECT 
                    'Grand Total' AS `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2),
                    ROUND(SUM(`O/P Wt`), 2),
                    ROUND(SUM(`Total Length`), 2),
                    ROUND(SUM(`Area`), 2),
                    ROUND(SUM(`Zinc`), 2),
                    ROUND(SUM(`Process Duration(in min)`), 2)
                FROM {table_name}
                WHERE `Actual Product` IN ({format_strings})
            """
            cursor.execute(total_query, product_list)
            total = cursor.fetchone()
            rows.append(total)

        else:
            query = f"""
                SELECT 
                    `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2) AS `Prop IP Wt`,
                    ROUND(SUM(`O/P Wt`), 2) AS `O/P Wt`,
                    ROUND(SUM(`Total Length`), 2) AS `Total Length`,
                    ROUND(SUM(`Area`), 2) AS `Area`,
                    ROUND(SUM(`Zinc`), 2) AS `Zinc`,
                    ROUND(SUM(`Process Duration(in min)`), 2) AS `Process Duration(in min)`
                FROM {table_name}
                WHERE `Actual Product` = %s
                GROUP BY `Actual Product`
            """
            cursor.execute(query, (actual_product,))
            rows = cursor.fetchall()

            total_query = f"""
                SELECT 
                    'Grand Total' AS `Actual Product`,
                    ROUND(SUM(`Prop IP Wt`), 2),
                    ROUND(SUM(`O/P Wt`), 2),
                    ROUND(SUM(`Total Length`), 2),
                    ROUND(SUM(`Area`), 2),
                    ROUND(SUM(`Zinc`), 2),
                    ROUND(SUM(`Process Duration(in min)`), 2)
                FROM {table_name}
                WHERE `Actual Product` = %s
            """
            cursor.execute(total_query, (actual_product,))
            total = cursor.fetchone()
            rows.append(total)

        df = pd.DataFrame(rows)
        
        # Filename auto-increment
        export_counts = defaultdict(int)
        export_counts[table_name] += 1
        filename = f"{table_name}({export_counts[table_name]})_{actual_product}_summary.xlsx"

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name="Summary")
        writer.close()
        output.seek(0)

        return send_file(output, download_name=filename, as_attachment=True)

    except Exception as e:
        print("Export Error:", e)
        return f"An error occurred during export: {str(e)}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
@app.route('/get-available-fy', methods=['POST'])
def get_available_fy():
    """Get available financial years by scanning the upload folder for specific unit"""
    try:
        data = request.get_json()
        unit = data.get('unit')
        
        if not unit:
            return jsonify({"error": "Unit is required"}), 400
        
        print(f"Scanning folder {UPLOAD_FOLDER} for {unit} files")
        
        # Check if folder exists
        if not os.path.exists(UPLOAD_FOLDER):
            print(f"Upload folder {UPLOAD_FOLDER} does not exist!")
            return jsonify({"fyList": []})
        
        # List all files in the folder for debugging
        all_files = os.listdir(UPLOAD_FOLDER)
        print(f"All files in folder: {all_files}")
        
        # Scan the upload folder for files matching the specific unit pattern
        fy_list = []
        for filename in all_files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            if unit == 'CGL-2' and filename.endswith('datcsv.csv'):
                print(f"Found CGL-2 file: {filename}")
                # Extract FY from filename (e.g., "24datcsv.csv" -> "FY24")
                if len(filename) >= 10:  # Ensure filename is long enough
                    fy_number = filename[:2]
                    # Validate it's a valid year number
                    if fy_number.isdigit() and 0 <= int(fy_number) <= 99:
                        fy_list.append(f"FY{fy_number}")
                        print(f"Added FY: FY{fy_number}")
            
            elif unit == 'CGL-3' and filename.endswith('datacgl.csv'):
                print(f"Found CGL-3 file: {filename}")
                # Extract FY from filename (e.g., "24datacgl.csv" -> "FY24")
                if len(filename) >= 12:  # Ensure filename is long enough
                    fy_number = filename[:2]
                    # Validate it's a valid year number
                    if fy_number.isdigit() and 0 <= int(fy_number) <= 99:
                        fy_list.append(f"FY{fy_number}")
                        print(f"Added FY: FY{fy_number}")
        
        # Remove duplicates and sort
        fy_list = sorted(list(set(fy_list)))
        
        print(f"Final FY list for {unit}: {fy_list}")
        return jsonify({"fyList": fy_list})
            
    except Exception as e:
        logger.error(f"Error getting available FY: {str(e)}")
        return jsonify({"error": "Failed to get available financial years"}), 500
@app.route('/list-uploaded-files')
def list_uploaded_files():
    """List all uploaded CSV files"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.csv'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file_size = os.path.getsize(file_path)
                files.append({
                    'name': filename,
                    'size': file_size,
                    'upload_date': datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500






    

@app.route('/debug/tables')
def debug_tables():
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SHOW TABLES LIKE '%data%'")
        tables = [list(table.values())[0] for table in cursor.fetchall()]
        return jsonify({"tables": tables})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/table-info/<table_name>')
def debug_table_info(table_name):
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        if not cursor.fetchone():
            return jsonify({"error": f"Table {table_name} does not exist"})
        
        # Get column info
        cursor.execute(f"DESCRIBE `{table_name}`")
        columns = cursor.fetchall()
        
        # Get sample data
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 5")
        sample_data = cursor.fetchall()
        
        return jsonify({
            "table_exists": True,
            "columns": columns,
            "sample_data": sample_data
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/data-sample/<table_name>')
def debug_data_sample(table_name):
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT `Actual Product`, `Next Unit`, COUNT(*) as count FROM `{table_name}` GROUP BY `Actual Product`, `Next Unit`")
        data_distribution = cursor.fetchall()
        return jsonify({"data_distribution": data_distribution})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/test-connection')
def test_connection():
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT NOW() as current_time")
        result = cursor.fetchone()
        return jsonify({"status": "connected", "database_time": result[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/debug/table-columns/<table_name>')
def debug_table_columns(table_name):
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        columns = cursor.fetchall()
        
        # Filter for columns that might be related to order TDC
        order_columns = [col for col in columns if 'order' in col['Field'].lower() or 'tdc' in col['Field'].lower()]
        
        return jsonify({
            "all_columns": columns,
            "order_related_columns": order_columns
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/table-sample/<table_name>')
def debug_table_sample(table_name):
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        if not cursor.fetchone():
            return jsonify({"error": f"Table {table_name} does not exist"})
        
        # Get sample data with date information
        cursor.execute(f"""
            SELECT 
                `Start Date`,
                `Order Tdc`,
                `Actual Product`,
                `Next Unit`,
                MONTH(STR_TO_DATE(`Start Date`, '%d-%m-%Y')) as extracted_month,
                LENGTH(`Start Date`) as date_length,
                SUBSTRING(`Start Date`, 4, 2) as month_part
            FROM `{table_name}` 
            LIMIT 20
        """)
        
        sample_data = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        columns = [col['Field'] for col in cursor.fetchall()]
        
        return jsonify({
            "table_exists": True,
            "columns": columns,
            "sample_data": sample_data,
            "date_analysis": "Examining date format and month extraction"
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/fy-tables')
def debug_fy_tables():
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check for all possible table patterns
        table_patterns = [
            '%datacsv%', '%datacgl%', '%data_csv%', '%data_cgl%',
            '%fy24%', '%fy25%', '%2024%', '%2025%'
        ]
        
        all_tables = []
        for pattern in table_patterns:
            cursor.execute("SHOW TABLES LIKE %s", (pattern,))
            tables = cursor.fetchall()
            # Use list(table.values())[0] for each table dict to get the table name
            all_tables.extend([list(table.values())[0] for table in tables])
        
        # Remove duplicates and sort
        unique_tables = sorted(list(set(all_tables)))
        
        return jsonify({
            "all_data_tables": unique_tables,
            "suggested_patterns": [
                "24datacsv", "25datacsv", "24datacgl", "25datacgl",
                "fy24datacsv", "fy25datacsv", "fy24datacgl", "fy25datacgl",
                "2024datacsv", "2025datacsv", "2024datacgl", "2025datacgl"
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/check-fy-table/<fy>/<unit>')
def debug_check_fy_table(fy, unit):
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        fy_number = fy[2:] if fy.startswith('FY') else fy
        fy_full = fy.lower()
        
        possible_tables = []
        if unit.lower() == 'cgl-2':
            possible_tables = [
                f"{fy_number}datacsv", f"{fy_full}datacsv", f"202{fy_number}datacsv",
                f"data_csv_{fy_number}", f"csv_data_{fy_number}"
            ]
        else:
            possible_tables = [
                f"{fy_number}datacgl", f"{fy_full}datacgl", f"202{fy_number}datacgl",
                f"data_cgl_{fy_number}", f"cgl_data_{fy_number}"
            ]
        
        existing_tables = []
        for table in possible_tables:
            cursor.execute("SHOW TABLES LIKE %s", (table,))
            if cursor.fetchone():
                existing_tables.append(table)
        
        # Check if tables have data
        table_info = {}
        for table in existing_tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM `{table}`")
            count = cursor.fetchone()['count']
            cursor.execute(f"SELECT MIN(`Start Date`) as min_date, MAX(`Start Date`) as max_date FROM `{table}`")
            date_range = cursor.fetchone()
            table_info[table] = {
                "record_count": count,
                "date_range": date_range
            }
        
        return jsonify({
            "financial_year": fy,
            "unit": unit,
            "possible_tables": possible_tables,
            "existing_tables": existing_tables,
            "table_info": table_info
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/debug/date-conversion-test')
def debug_date_conversion_test():
    """Test date conversion from yyyy-mm-dd to dd-mm-yyyy"""
    test_dates = [
        '2024-01-15',  # yyyy-mm-dd
        '2024-12-31',  # yyyy-mm-dd
        '2025-06-01',  # yyyy-mm-dd
    ]
    
    converted_dates = []
    for date_str in test_dates:
        # Convert yyyy-mm-dd to dd-mm-yyyy
        day = date_str[8:10]
        month_part = date_str[5:7]
        year = date_str[0:4]
        converted = f"{day}-{month_part}-{year}"
        converted_dates.append({
            "original": date_str,
            "converted": converted
        })
    
    return jsonify({
        "date_conversion_test": converted_dates,
        "explanation": "Converting from HTML date input format (yyyy-mm-dd) to database format (dd-mm-yyyy)"
    })

@app.route('/upload', methods=['POST'])
def upload_production_data():
    try:
        # Get the selected unit
        unit = request.form.get('unit')
        
        if not unit:
            flash('Please select a unit', 'error')
            return redirect(request.referrer or url_for('production'))
        
        # Handle file upload if you have file input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Ensure uploads directory exists
                os.makedirs('uploads', exist_ok=True)
                filename = secure_filename(file.filename)
                file.save(os.path.join('uploads', filename))
                flash('File uploaded successfully', 'success')
        
        # Process the form data and redirect to next step
        flash(f'Unit {unit} selected successfully', 'success')
        # You should define a 'next_step' route or change this redirect as needed
        return redirect(url_for('next_step'))
        
    except Exception as e:
        flash(f'Error processing form: {str(e)}', 'error')
        return redirect(request.referrer or url_for('production'))

@app.route('/next_step')
def next_step():
    unit = request.args.get('unit')
    if not unit:
        flash('No unit selected', 'error')
        return redirect(url_for('production'))
    
    return render_template('next_step.html', unit=unit)


@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Handle CSV file upload - save to directory AND import into MySQL as-is"""
    try:
        # Get form data
        unit = request.form.get('unit')
        financial_year = request.form.get('financialYear')
        action = request.form.get('action')  # expected 'add' or 'update'

        if not unit or not financial_year or not action:
            return jsonify({"success": False, "error": "Missing required parameters"}), 400

        if 'csvFile' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['csvFile']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "File must be a CSV"}), 400

        # Determine expected filename (keep your existing naming convention)
        fy_number = financial_year[2:]  # e.g., FY22 -> "22"
        if unit == 'CGL-2':
            expected_filename = f"{fy_number}datacsv.csv"
        elif unit == 'CGL-3':
            expected_filename = f"{fy_number}datacgl.csv"
        else:
            return jsonify({"success": False, "error": "Invalid unit"}), 400

        # Accept both exact filename or a filename that endswith expected (tolerant)
        # but to keep your original enforcement, leave strict equality check:
        if file.filename != expected_filename:
            return jsonify({"success": False, "error": f"File name must be {expected_filename} for {unit}"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, expected_filename)

        # add/update checks for the filesystem copy
        if action == 'add' and os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File {expected_filename} already exists"}), 400
        if action == 'update' and not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File {expected_filename} does not exist for update"}), 400

        # Save raw CSV into UPLOAD_FOLDER
        file.save(file_path)

        # Import CSV into MySQL as-is; on update we drop & reload, on add we create if not exists
        import_csv_to_mysql(file_path, expected_filename.replace(".csv", ""), action=action)

        file_size = os.path.getsize(file_path)
        return jsonify({
            "success": True,
            "message": f"File {expected_filename} saved and imported into MySQL",
            "file_path": file_path,
            "financial_year": financial_year,
            "unit": unit,
            "file_size": file_size
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500

@app.route('/debug-files')
def debug_files():
    """Debug endpoint to see all files in upload folder"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file_info = {
                'name': filename,
                'size': os.path.getsize(file_path),
                'is_file': os.path.isfile(file_path),
                'type': 'CGL-2' if filename.endswith('datacsv.csv') else 'CGL-3' if filename.endswith('datacgl.csv') else 'Other'
            }
            files.append(file_info)
        
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500   



@app.route('/debug-folder')
def debug_folder():
    """Debug endpoint to see exactly what's in the upload folder"""
    try:
        folder_info = {
            "folder_path": UPLOAD_FOLDER,
            "folder_exists": os.path.exists(UPLOAD_FOLDER),
            "folder_writable": os.access(UPLOAD_FOLDER, os.W_OK) if os.path.exists(UPLOAD_FOLDER) else False,
            "files": []
        }
        
        if folder_info["folder_exists"]:
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                folder_info["files"].append({
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'is_file': os.path.isfile(file_path),
                    'file_type': 'CGL-2' if filename.endswith('datcsv.csv') else 'CGL-3' if filename.endswith('datacgl.csv') else 'Other'
                })
        
        return jsonify(folder_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

    

if __name__ == "__main__":
    # Initialize database tables
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()
        create_tables_if_not_exist(cursor)
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)