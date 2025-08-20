from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify, session, flash
import mysql.connector
from mysql.connector import pooling
import pandas as pd
from io import BytesIO
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import joblib
import traceback
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from flask_cors import CORS  # Added for CORS support

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.secret_key = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

# Database configuration with environment variables
db_config = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "port": int(os.getenv('DB_PORT', '3306')),
    "user": os.getenv('DB_USER', 'appadmin'),
    "password": os.getenv('DB_PASSWORD', 'Megha@2207'),
    "database": os.getenv('DB_NAME', 'new'),
    "pool_name": "my_pool",
    "pool_size": 5
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

# Load ML model (if available)
try:
    model = joblib.load('models/prod_model.joblib')
    logger.info("ML model loaded successfully")
except FileNotFoundError:
    logger.warning("ML model file 'models/prod_model.joblib' not found. Using default calculations.")
    model = None

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
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = None
        try:
            conn = db_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                flash('Logged in successfully!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
                return redirect(url_for('login'))
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'danger')
            return redirect(url_for('login'))
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    return render_template('login.html')

@app.before_request
def require_login():
    allowed_routes = ['login', 'register', 'static', 'health', 'home']
    if request.endpoint not in allowed_routes and 'user_id' not in session:
        return redirect(url_for('login'))
    
@app.route('/logout')
def logout():
    session.pop('user_id', None)
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
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        conn = None
        try:
            conn = db_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Check if username exists
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
            
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
            return redirect(url_for('register'))
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
    
    return render_template('register.html')

# Main routes
@app.route('/')
def home():
    return "🚀 Flask App is Running on Azure!"

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('base.html')

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

# Order Punch System Routes
@app.route('/predict', methods=['POST'])
def predict():
    """Predict productivity and required time using all relevant features"""
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")

        # Validate all required fields
        required_fields = {
            'product_type': str,
            'tdc': str,
            'thickness': float,
            'width': float,
            'zinc_coating': float,
            'quantity': int
        }
        
        # Validate and convert inputs
        features = {}
        for field, field_type in required_fields.items():
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            try:
                features[field] = field_type(data[field])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field}")

        logger.info(f"Validated features: {features}")

        if model:
            # Prepare feature array in correct order
            product_type_encoded = encode_product_type(features['product_type'])
            tdc_encoded = encode_tdc(features['tdc'])
            
            # Correct array structure - single list of features
            model_input = np.array([
                features['thickness'],
                features['width'],
                features['zinc_coating'],
                product_type_encoded,
                tdc_encoded
            ]).reshape(1, -1)  # Reshape to 2D array for prediction
            
            logger.info(f"Model input features: {model_input}")

            # Get prediction
            productivity = model.predict(model_input)[0]
            required_time = features['quantity'] / productivity if productivity > 0 else 0
        else:
            # Fallback calculation
            logger.warning("Using fallback calculation (no model)")
            productivity = 1.0
            required_time = features['quantity']

        logger.info(f"Prediction results - Productivity: {productivity}, Required Time: {required_time}")

        return jsonify({
            'productivity': round(productivity, 2),
            'required_time': round(required_time, 2)
        })

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal prediction error'}), 500

@app.route('/order_punch', methods=['GET', 'POST'])
def order_punch():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        table = get_month_table(cursor)
        time_stats = get_time_stats(cursor, table)

        if request.method == 'POST':
            form = request.form
            logger.info(f"Form data received: {dict(form)}")

            try:
                # Validate and convert inputs
                prediction_data = {
                    'product_type': form['product_type'],
                    'tdc': form['tdc'],
                    'thickness': float(form['thickness']),
                    'width': float(form['width']),
                    'zinc_coating': float(form['zinc_coating']),
                    'quantity': int(form['quantity'])
                }

                # Get prediction
                response = requests.post(
                    url_for('predict', _external=True),
                    json=prediction_data,
                    timeout=5
                )

                if response.status_code != 200:
                    error_msg = response.json().get('error', 'Prediction failed')
                    raise ValueError(error_msg)

                prediction = response.json()
                
                # Verify available time
                if prediction['required_time'] > time_stats['left_time']:
                    raise ValueError("Not enough available time for this order")

                # Insert order
                cursor.execute(f"""
                    INSERT INTO `{table}` 
                    (product_type, tdc, thickness, width, zinc_coating, 
                     quantity, required_time, productivity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    prediction_data['product_type'],
                    prediction_data['tdc'],
                    prediction_data['thickness'],
                    prediction_data['width'],
                    prediction_data['zinc_coating'],
                    prediction_data['quantity'],
                    prediction['required_time'],
                    prediction['productivity']
                ))
                conn.commit()
                flash('Order booked successfully!', 'success')

            except ValueError as ve:
                flash(f'Error: {str(ve)}', 'danger')
            except Exception as e:
                logger.error(f"Order submission error: {str(e)}")
                flash('Error submitting order. Please try again.', 'danger')

            return redirect(url_for('order_punch'))

        # GET request - show existing orders
        cursor.execute(f"SELECT * FROM `{table}` ORDER BY id DESC")
        records = cursor.fetchall()

        return render_template("order_punch.html",
                           records=records,
                           available_time=time_stats['available_time'],
                           booked_time=time_stats['booked_time'],
                           left_time=time_stats['left_time'],
                           table=table)

    except Exception as e:
        logger.error(f"Order punch error: {str(e)}")
        flash('An error occurred', 'danger')
        return redirect(url_for('order_punch'))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/order/delete/<int:id>')
def delete_order(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()
        table = get_month_table(cursor)
        cursor.execute(f"DELETE FROM `{table}` WHERE id = %s", (id,))
        conn.commit()
        flash('Order deleted successfully', 'success')
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        flash('Error deleting order', 'danger')
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    return redirect(url_for('order_punch'))

@app.route('/order/edit/<int:id>', methods=['GET', 'POST'])
def edit_order(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        table = get_month_table(cursor)

        if request.method == 'POST':
            form = request.form
            
            # Get prediction
            try:
                response = requests.post(
                    url_for('predict', _external=True),
                    json={
                        'product_type': form['product_type'],
                        'tdc': form['tdc'],
                        'thickness': float(form['thickness']),
                        'width': float(form['width']),
                        'zinc_coating': float(form['zinc_coating']),
                        'quantity': int(form['quantity']),
                    }
                )
                prediction = response.json()
                if 'error' in prediction:
                    raise ValueError(prediction['error'])
                    
                productivity = prediction['productivity']
                required_time = prediction['required_time']
            except Exception as e:
                logger.warning(f"Using fallback calculation: {str(e)}")
                productivity = 1.0
                required_time = float(form['quantity'])

            cursor.execute(f"""
                UPDATE `{table}` SET
                product_type=%s, tdc=%s, thickness=%s, width=%s,
                zinc_coating=%s, quantity=%s, required_time=%s, productivity=%s
                WHERE id=%s
            """, (
                form['product_type'], form['tdc'], float(form['thickness']),
                float(form['width']), form['zinc_coating'], int(form['quantity']),
                required_time, productivity, id
            ))
            conn.commit()
            flash('Order updated successfully!', 'success')
            return redirect(url_for('order_punch'))

        # Get order details
        cursor.execute(f"SELECT * FROM `{table}` WHERE id = %s", (id,))
        order = cursor.fetchone()
        return render_template("edit.html", order=order)

    except Exception as e:
        logger.error(f"Edit error: {str(e)}")
        flash('Error updating order', 'danger')
        return redirect(url_for('order_punch'))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# Data routes
@app.route('/search', methods=['POST'])
def search_data():
    conn = None
    try:
        filters = request.get_json()
        financial_year = filters.get('financialYear')

        if not financial_year:
            return jsonify({"error": "Financial Year is a required filter."}), 400

        table_name = f"{financial_year[2:]}datacsv"

        query_parts = []
        params = []

        # Dynamically build the WHERE clause based on provided filters
        if filters.get('fromDate') and filters.get('toDate'):
            query_parts.append("`Start Date` BETWEEN %s AND %s")
            params.extend([filters['fromDate'], filters['toDate']])
        if filters.get('month'):
            query_parts.append("MONTH(`Start Date`) = %s")
            params.append(filters['month'])
        if filters.get('orderTdc'):
            query_parts.append("`Order_Tdc` LIKE %s")
            params.append(f"%{filters['orderTdc']}%")
        if filters.get('shift'):
            query_parts.append("`Shift` = %s")
            params.append(filters['shift'])
        if filters.get('unit'):
            query_parts.append("`Next Unit` = %s") # Assuming 'unit' maps to 'Next Unit' column
            params.append(filters['unit'])

        base_query = f"SELECT * FROM `{table_name}`"
        if query_parts:
            base_query += " WHERE " + " AND ".join(query_parts)
        base_query += " ORDER BY `Start Date` DESC, `Start Time` DESC LIMIT 500"

        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True, buffered=True)
        cursor.execute(base_query, tuple(params))
        results = [dict(row) for row in cursor.fetchall()]

        # Convert datetime objects to strings for JSON compatibility
        for row in results:
            for key, value in row.items():
                if isinstance(value, datetime):
                    row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, bytes): # Handle potential bytearray from DB
                    row[key] = value.decode('utf-8')

        return jsonify(results)

    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return jsonify({"error": f"Database error: {err.msg}"}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/get_filtered_data', methods=['POST'])
def get_filtered_data():
    data = request.json
    cgl = data.get('cgl')
    fy = data.get('fy')
    product = data.get('product')

    print(f"Received: CGL={cgl}, FY={fy}, Product={product}")  # Debug

    if cgl != "CGL-2" or not fy or not product:
        return jsonify({"error": "Missing or invalid inputs"}), 400

    table = f"{fy}datacsv"
    print(f"Querying table: {table}")  # Debug

    conn = None
    try:
        # Use **db_config to unpack the dictionary into keyword arguments
        conn = db_pool.get_connection()
        # Use a dictionary cursor from the start for both operations
        cursor = conn.cursor(dictionary=True)

        # Ensure Area column exists
        cursor.execute(f"SHOW COLUMNS FROM `{table}` LIKE 'Area';")
        if not cursor.fetchone():
            cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `Area` DOUBLE")

        cursor.execute(f"SHOW COLUMNS FROM `{table}` LIKE 'Zinc';")
        if not cursor.fetchone():
            cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `Zinc` DOUBLE")

        # Update Area and Zinc values
        cursor.execute(f"""
            UPDATE `{table}` 
            SET `Area` = ROUND((`Ip Width` * `Total Length`) / 1000, 4), `Zinc` = ROUND((`Ip Width` * `Total Length` * `Total Zn/AlZn Coating`) / 1000000000, 3)
        """)

        conn.commit()

        query = f"""
        SELECT 
            `Op Batch No`,
            `Actual Product`, 
            `Actual Tdc`,
            CASE 
                WHEN LEFT(`Actual Tdc`, 3) = 'ZAP' THEN 'Appliance'
                WHEN LEFT(`Actual Tdc`, 3) = 'ZST' THEN 'Retail'
                WHEN LEFT(`Actual Tdc`, 3) = 'ZGN' THEN 'Retail'
                WHEN LEFT(`Actual Tdc`, 3) = 'ZTU' THEN 'P&T'
                WHEN LEFT(`Actual Tdc`, 3) = 'ZPL' THEN 'Panel'
                WHEN LEFT(`Actual Tdc`, 3) = 'ZEX' THEN 'export'
                ELSE 'Other'
            END AS segment,
            `Prop Ip Wt`,
            `O/P Wt`,
            `Total Length`,
            `Area`,
            `Zinc`,
            ROUND(`Process Duration(in min)`, 0) AS `Process Duration(in min)`,
            ROUND((`Prop Ip Wt` * 1000) / (7.850 * (`Ip Width` * `Total Length`) / 1000), 3) AS `CRFH thickness`,
            ROUND((`Total Zn/AlZn Coating` / 
                CASE 
                    WHEN `Actual Product` = 'GI' THEN 7140
                    WHEN `Actual Product` IN ('GL', 'PPGL') THEN 3750
                    WHEN `Actual Product` = 'ZM' THEN 6850
                    ELSE NULL
                END
            ) + ROUND((`Prop Ip Wt` * 1000) / (7.850 * (`Ip Width` * `Total Length`) / 1000), 3), 4) AS `GP thickness`,
            `Total Zn/AlZn Coating`,
            `Op Width`,
            ROUND((`Total Length`/`Process Duration(in min)`),3) as speed,
            ROUND((`O/P Wt`/`Process Duration(in min)`)*60,3) as productivity

        FROM `{table}`
        WHERE `Actual Product` = %s
        """

        cursor.execute(query, (product,))
        rows = cursor.fetchall()
        return jsonify(rows)

    except Exception as e:
        print(f"Error: {e}")  # Debug
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure the connection is closed even if an error occurs
        if conn and conn.is_connected():
            cursor.close()
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