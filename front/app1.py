from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import mysql.connector
import pandas as pd
import io 
from io import BytesIO
import os
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)

# Define the database configuration centrally
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Megha@22",
    "database": "new"
}

# Database configuration for the Order Punch system
db_config_orderdata = {
    "host": "localhost",
    "user": "root",
    "password": "Megha@22",
    "database": "order_data"
}

@app.route('/')
def index():
    # The default page is now the home/welcome page
    return render_template('home.html')

@app.route('/production')
def production():
    return render_template('production.html')

@app.route('/productivity')
def productivity():
    return render_template("productivity.html")

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

        conn = mysql.connector.connect(**db_config)
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
        conn = mysql.connector.connect(**db_config)
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

        conn = mysql.connector.connect(**db_config)
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

from collections import defaultdict

# Track export counts per table in memory (resets when server restarts)
export_counts = defaultdict(int)


@app.route('/export-summary', methods=['POST'])
def export_summary():
    conn = None
    try:
        fy = request.form.get('fy')
        actual_product = request.form.get('actual_product')

        if not fy or not actual_product:
            return "Missing filters", 400

        table_name = f"{fy[-2:]}datacsv"
        conn = mysql.connector.connect(**db_config)
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

# === START: Code merged from app2.py for Order Punch ===

# Helper functions for Order Punch
def create_table_if_not_exists(cursor, table_name):
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
            required_time FLOAT
        )
    """)

def get_month_table(cursor):
    now = datetime.now()
    month_year = now.strftime('%B_%Y').lower()
    create_table_if_not_exists(cursor, month_year)
    return month_year

@app.route('/order_punch', methods=['GET', 'POST'])
def order_punch():    
    # Constants
    total_hours = 744
    shutdown = 24
    setup = 12
    utilization = 98
    available_time = round((total_hours - shutdown - setup) * (utilization / 100), 2)

    conn = None
    try:
        conn = mysql.connector.connect(**db_config_orderdata)
        cursor = conn.cursor(dictionary=True)
        table = get_month_table(cursor)

        if request.method == 'POST':
            form = request.form

            # Extract data from form
            product_type = form['product_type']
            tdc = form['tdc']
            thickness = float(form['thickness'])
            width = float(form['width'])
            zinc_coating = form['zinc_coating']
            quantity = int(form['quantity'])

            productivity = 1.0  # default fixed
            required_time = round(quantity / productivity, 2)

            # Get already booked time
            cursor.execute(f"SELECT SUM(required_time) FROM `{table}`")
            booked_time = cursor.fetchone()['SUM(required_time)'] or 0
            remaining = round(available_time - booked_time, 2)

            if required_time <= remaining:
                cursor.execute(f"""
                    INSERT INTO `{table}` 
                    (product_type, tdc, thickness, width, zinc_coating, quantity, required_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    product_type, tdc, thickness, width, zinc_coating, quantity, required_time
                ))
                conn.commit()
            else:
                # Fetch existing records to display even when there's an error
                cursor.execute(f"SELECT * FROM `{table}` ORDER BY id DESC")
                records = cursor.fetchall()
                return render_template("order_punch.html",
                                       error="Operation time booked. Not enough available time.",
                                       show_form=True,
                                       records=records,
                                       available_time=available_time,
                                       booked_time=booked_time,
                                       left_time=remaining,
                                       table=table)

            return redirect(url_for('order_punch'))

        # Fetch records for GET request
        cursor.execute(f"SELECT * FROM `{table}` ORDER BY id DESC")
        records = cursor.fetchall()

        # Booked Time
        cursor.execute(f"SELECT SUM(required_time) FROM `{table}`")
        booked_time = cursor.fetchone()['SUM(required_time)'] or 0
        left_time = round(available_time - booked_time, 2)

        return render_template("order_punch.html",
                               records=records,
                               available_time=available_time,
                               booked_time=booked_time,
                               left_time=left_time,
                               table=table,
                               error=None,
                               show_form=False)
    except mysql.connector.Error as err:
        return f"Database Error: {err}", 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/order/delete/<int:id>')
def delete_order(id):
    conn = None
    try:
        conn = mysql.connector.connect(**db_config_orderdata)
        cursor = conn.cursor()
        table = get_month_table(cursor) # We need the cursor to determine the table
        cursor.execute(f"DELETE FROM `{table}` WHERE id = %s", (id,))
        conn.commit()
        return redirect(url_for('order_punch'))
    except mysql.connector.Error as err:
        return f"Database Error: {err}", 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/order/edit/<int:id>', methods=['GET', 'POST'])
def edit_order(id):
    conn = None
    try:
        conn = mysql.connector.connect(**db_config_orderdata)
        cursor = conn.cursor(dictionary=True)
        table = get_month_table(cursor)

        if request.method == 'POST':
            form = request.form
            # Recalculate required_time on edit
            quantity = int(form['quantity'])
            productivity = 1.0  # default fixed
            required_time = round(quantity / productivity, 2)

            cursor.execute(f"""
                UPDATE `{table}` SET
                product_type=%s, tdc=%s, thickness=%s, width=%s,
                zinc_coating=%s, quantity=%s, required_time=%s
                WHERE id=%s
            """, (
                form['product_type'], form['tdc'], form['thickness'],
                form['width'], form['zinc_coating'], form['quantity'], required_time, id
            ))
            conn.commit()
            return redirect(url_for('order_punch'))

        # For GET request, fetch the order to pre-fill the form
        cursor.execute(f"SELECT * FROM `{table}` WHERE id = %s", (id,))
        order = cursor.fetchone()
        return render_template("edit.html", order=order)
    except mysql.connector.Error as err:
        return f"Database Error: {err}", 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# === END: Code merged from app2.py ===

if __name__ == "__main__":
    import os
    # Use a standard web port like 5000, not the MySQL port 3306
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
