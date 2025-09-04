from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import re
import sqlite3
from datetime import datetime
import pandas as pd
from io import BytesIO

app = Flask(__name__)
# Safe conversion functions
def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if conversion fails"""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, returning default if conversion fails"""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == '':
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
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

# Database initialization
def init_db():
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    # Create settings table for monthly available time
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (month_year TEXT PRIMARY KEY, available_time REAL)''')
    
    # Check if any month exists, if not create current month with default available time
    current_month = datetime.now().strftime("%b%Y")
    c.execute("SELECT COUNT(*) FROM settings WHERE month_year = ?", (current_month,))
    if c.fetchone()[0] == 0:
        # Set default available time to 14400 minutes (10 days of 24 hours)
        c.execute("INSERT INTO settings (month_year, available_time) VALUES (?, ?)", 
                 (current_month, 14400))
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Function to extract numerical value from TDC
def extract_tdc_value(tdc_string):
    if tdc_string is None or tdc_string == '':
        return 0
    
    tdc_str = str(tdc_string)
    numbers = re.findall(r'\d+', tdc_str)
    
    if numbers:
        return float(numbers[0])
    else:
        return float(hash(tdc_str) % 1000)

# Load the model
try:
    with open('productivity_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Use 'target' instead of 'targets'
    target = model_data.get('target', 'Productivity(TPH)')
    
    # Handle the label encoder correctly
    if 'label_encoder' in model_data:
        le = model_data['label_encoder']
    elif 'label_encoders' in model_data:
        # Get the encoder for 'Actual Product'
        le = model_data['label_encoders'].get('Actual Product', None)
    else:
        le = None
    
    print(f"Model loaded with features: {features}")
    print(f"Model loaded with target: {target}")
    
    # Create a mapping from our feature names to the model's expected feature names
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

# Get or create orders table for a specific month
def get_orders_table(month_year):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    # Create table with proper schema including booking_date default
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

# Get available time for a month
# Get available time for a month
# Get available time for a month
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

# Get total booked time for a month
# Get total booked time for a month
# Get total booked time for a month
def get_booked_time(month_year):
    table_name = f"orders_{month_year}"
    
    try:
        conn = sqlite3.connect('production_planning.db')
        c = conn.cursor()
        
        # Check if table exists first
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

# Get all orders for a month
def get_all_orders(month_year):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    try:
        # First check if the table exists
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

# Get a specific order
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

# Update an order
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

# Delete an order
def delete_order(month_year, order_id):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    c.execute(f"DELETE FROM {table_name} WHERE id = ?", (order_id,))
    conn.commit()
    conn.close()

# Add a new order
# Add a new order
def add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time):
    table_name = f"orders_{month_year}"
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    try:
        # First make sure the table exists
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

# Get month data (available, booked, remaining time)
# Get month data (available, booked, remaining time)
def get_month_data(month_year):
    try:
        available_time = get_available_time(month_year)
        booked_time = get_booked_time(month_year)
        remaining_time = available_time - booked_time
        
        # Ensure all values are floats
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

@app.route('/')
def home():
    current_month = datetime.now().strftime("%b%Y")
    available_time = get_available_time(current_month)
    booked_time = get_booked_time(current_month)
    remaining_time = available_time - booked_time
    
    # Get all available months
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    c.execute("SELECT month_year FROM settings ORDER BY month_year")
    months = [row[0] for row in c.fetchall()]
    conn.close()
    
    # Get orders for current month
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
        # Get form data with safe defaults
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc = request.form.get('tdc', '')
        
        # Use safe conversion functions
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        
        month_year = request.form.get('month_year', datetime.now().strftime("%b%Y"))
        confirm_booking = request.form.get('confirm_booking', 'false').lower() == 'true'
        
        # Validate required fields
        if not all([unit, product_type, tdc]):
            return jsonify({'error': 'Missing required fields'})
        
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        
        # Extract TDC value
        tdc_value = extract_tdc_value(tdc)
        
        # Encode product type if label encoder is available
        if le is not None:
            try:
                product_type_encoded = le.transform([product_type])[0]
            except ValueError:
                # Handle unknown product types
                product_type_encoded = 0
        else:
            # Fallback: use simple encoding if no label encoder
            product_type_encoded = hash(product_type) % 1000
        
        # Prepare features in the correct order and with the correct names
        features_dict = {
            'product_type': product_type_encoded,
            'tdc_value': tdc_value,
            'zinc': zinc,
            'thickness': thickness
        }
        
        # Map our feature names to the model's expected feature names
        mapped_features = {feature_mapping.get(k, k): v for k, v in features_dict.items()}
        
        # Create DataFrame with features in the correct order
        features_df = pd.DataFrame([mapped_features], columns=features)
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        productivity = model.predict(features_scaled)[0]
        
        # Calculate required time
        required_time = (quantity / productivity) * 60 if productivity > 0 else 0
        
        # Get month data
        month_data = get_month_data(month_year)
        
        # Check if we can book this order
        can_book = bool(month_data['remaining_time'] >= required_time)
        
        # If this is a booking confirmation and we can book, save the order
        if confirm_booking and can_book:
            success = add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time)
            if not success:
                return jsonify({'error': 'Failed to save order to database'})
            
            # Get updated month data after booking
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
        # Get form data with safe conversions
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc = request.form.get('tdc', '')
        
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        
        month_year = request.form.get('month_year', datetime.now().strftime("%b%Y"))
        
        # Validate inputs
        if not all([unit, product_type, tdc]):
            return jsonify({'error': 'Missing required fields'})
        
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        
        # First predict productivity and required time
        # (You might want to refactor this to avoid code duplication)
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
        
        # Add the order to the database
        success = add_order(month_year, unit, product_type, tdc, thickness, zinc, quantity, productivity, required_time)
        
        if not success:
            return jsonify({'error': 'Failed to add order to database'})
        
        # Get updated month data after booking
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
        
        # Convert all orders to JSON-serializable format
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
        # Convert to JSON-serializable format
        serializable_order = convert_to_serializable(order)
        return jsonify(serializable_order)
    else:
        return jsonify({'error': 'Order not found'}), 404

@app.route('/update_order/<month_year>/<int:order_id>', methods=['POST'])
def update_order_route(month_year, order_id):
    try:
        # Get form data with safe conversions
        unit = request.form.get('unit', '')
        product_type = request.form.get('product_type', '')
        tdc_input = request.form.get('tdc', '')
        
        thickness = safe_float(request.form.get('thickness'), 0.5)
        zinc = safe_float(request.form.get('zinc'), 130.0)
        quantity = safe_int(request.form.get('quantity'), 1)
        
        # Validate inputs
        if not all([unit, product_type, tdc_input]):
            return jsonify({'error': 'Missing required fields'})
        
        if thickness <= 0 or zinc <= 0 or quantity <= 0:
            return jsonify({'error': 'Thickness, zinc coating, and quantity must be positive values'})
        
        # Extract numerical value from TDC
        tdc_value = extract_tdc_value(tdc_input)
        
        # Encode product type if label encoder is available
        if le is not None:
            try:
                product_type_encoded = le.transform([product_type])[0]
            except ValueError:
                product_type_encoded = 0
        else:
            product_type_encoded = hash(product_type) % 1000
        
        # Prepare features
        features_dict = {
            'product_type': product_type_encoded,
            'tdc_value': tdc_value,
            'zinc': zinc,
            'thickness': thickness
        }
        
        # Map our feature names to the model's expected feature names
        mapped_features = {feature_mapping.get(k, k): v for k, v in features_dict.items()}
        
        # Create DataFrame with features in the correct order
        features_df = pd.DataFrame([mapped_features], columns=features)
        
        # Scale the features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        productivity = model.predict(features_scaled)[0]
        
        # Calculate required time
        required_time = (quantity / productivity) * 60 if productivity > 0 else 0
        
        # Update the order
        update_order(month_year, order_id, unit, product_type, tdc_input, thickness, zinc, quantity, productivity, required_time)
        
        # Get updated month data
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
    
    # Get updated month data after deletion
    month_data = get_month_data(month_year)
    
    # Get all orders for the month
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
    
    # Create DataFrame
    df = pd.DataFrame(orders)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Create Excel file in memory
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
    
    # Get all months from settings table
    c.execute("SELECT month_year FROM settings ORDER BY month_year")
    months = [row[0] for row in c.fetchall()]
    
    conn.close()
    
    return jsonify({'months': months})

if __name__ == '__main__':
    app.run(debug=True)