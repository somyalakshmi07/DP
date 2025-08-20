from flask import Flask, request, jsonify, render_template
import mysql.connector
from datetime import datetime


app = Flask(__name__)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Megha@2207',
    'database': 'new'
}

@app.route('/')
def index():
    return render_template('production.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    from_date = data.get('fromDate')
    to_date = data.get('toDate')
    month = data.get('month')
    order_tdc = data.get('orderTdc')
    financial_year = data.get('financialYear')
    shift = data.get('shift')
    
    financial_year = data.get('financialYear')
    if not financial_year or financial_year == 'FY25':
        table_name = '25datacsv'
    elif financial_year == '24':
        # Return FY24 data
        table_name = '24datacsv'
    elif financial_year == '23':
        # Return FY23 data
        table_name = '23datacsv'
    elif financial_year == '22':
        # Return FY22 data
        table_name = '22datacsv'
    # else:
    #     return jsonify({'message': 'Data not found'})
    else:
    #     table_name = f"{financial_year[2:]}datacsv"
          table_name = f"{financial_year}datacsv"
    query = f"""
    SELECT 
        `Row`, `Actual Product`, `Segment`, `DP FLAG`, `MotherBatchNo`, `Ip Width`, `Ip Thick`,
        `Mother Ip Wt`, `Order_Tdc`, `Op Batch No`, `Actual Tdc`, `Op Thk`, `Op Width`, `Prop Ip Wt`,
        `O/P_Wt`, `Total Length`, `Target coating weight`, `ZN/AlZn Coating Top`, `ZN/AlZn Coating Bot`,
        `Total Zn/AlZn Coating`, `Spangle Type`, `Tlv Usage`, `Tlv Elongation`, `SPM Usage`,
        `SPM Elongation`, `Entry Baby Wt`, `Entry End Cut`, `Exit Baby Wt`, `Exit End Cut`,
        `Trim Loss`, `Total Scrap`, `Surface Finish`, `Passivation_Type`, `Passivation Flag`,
        `Logo`, `Liner Marking`, `Ip Idm`, `Ip Odm`, `Cr grade`, `Zn theo weight`, `Sleeve`,
        `L2 Remarks`, `Next Unit`, `Status`, `Material Yield(%) with Zinc`, 
        `Material Yield(%) without Zinc`, `Start Date`, `Start Time`, `End Date`, `End Time`,
        `Shift`, `Process Duration(in min)`, `Pdo Time`, `Age(Days)`, `PlanThickness`,
        `PlanWidth`, `Target Thick`, `Target Width`, `Anneal Code`, `No Of Samples`,
        `Oil Usage`, `Oil type`, `Plan path`, `Actual Path`, `Customer`, `Order Desc`,
        `PLTCM/CCM Prod Date`, `QA Remarks`, `Qa Code`, `Ip Mat`, `Distribution Channel`,
        `destinationCity`, `Plan Order`, `Actual Order`, `Plan Edge Cond`, `Actual Edge`,
        `NCO Flag`, `Nco Reason`, `Unloaded Wt`, `Trimming`, `End use`, `Hr Batch No`,
        `Sleeve Used`, `PlnOrdIdDesc`, `Planned Product`, `PlanCustomer`, `L3 remarks`,
        `coil_type`, `Average Line Speed (mpm)`, `Committed Date`, `Delivery date`, `Idm`,
        `Odm`, `Heat No`, `Hr grade`, `Hold Reason Remark`, `User Id`, `c`, `mn`, `s`, `si`,
        `ph`, `al`, `cr`, `ca`, `cu`, `n`, `ni`, `mo`, `v`, `nb`, `ti`, `t1`, `b`, `sn`,
        `cq`, `ctAvg`, `ftAvg`, `ctAvg`, `hrThk`, `hrWdt`, `hrWt`, `hrCrown`, `hrWdg`,
        `slabNo`, `Surface Conditioning Mill Force`, `Holding Section Strip Actual Temperature`,
        `Surface Conditioning Mill Elongation`, `Tension Leveller Elongation`,
        `Furnace Entry Speed`, `Tube Treatment 6 Strip Actual Temperature`, `PDOTr`,
        `PDOPor`, `PDOSpeedSetupFurn`, `Schd Line No`, `NRI`, `RA_CODE`,
        `Surface_Roughness_Min`, `Surface_Roughness_Max`
    FROM {table_name} WHERE 1=1
"""

    params = []

    if order_tdc:
        query += " AND `Order Tdc` LIKE %s"
        params.append(f"%{order_tdc}%")
    if from_date:
        query += " AND `Start Date` >= %s"
        params.append(from_date)
    if to_date:
        query += " AND `End Date` <= %s"
        params.append(to_date)
    if month:
        query += " AND MONTH(`Start Date`) = %s"
        params.append(month)
    if shift:
        query += " AND Shift = %s"
        params.append(shift)

    query += " LIMIT 500"  # Limit the results to the first 500 rows

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

    formatted_results = []
    for row in results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                formatted_row[key] = value.strftime('%Y-%m-%d')
            else:
                formatted_row[key] = str(value) if value is not None else 'N/A'
        formatted_results.append(formatted_row)

    return jsonify(formatted_results)


@app.route('/calculation')
def calculation_page():
    return render_template('calculation.html')


@app.route('/calculate', methods=['POST'])
def calculate_sum():
    production_name = request.form.get('production_name')
    fy_year = request.form.get('fy_year')

    if not production_name or not fy_year:
        return jsonify({'message': 'Please provide both Production Name and FY Year'}), 400

    # Determine the correct table
    if fy_year == '25':
        table_name = '25datacsv'
    elif fy_year == '24':
        table_name = '24datacsv'
    elif fy_year == '23':
        table_name = '23datacsv'
    elif fy_year == '22':
        table_name = '22datacsv'
    else:
        return jsonify({'message': 'Invalid FY Year provided'}), 400

    query = f"""
        SELECT SUM(`O/P_Wt`) AS total_output
        FROM {table_name}
        WHERE `Actual Product` = %s
    """

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query, (production_name,))
        result = cursor.fetchone()
        total = result[0] if result[0] is not None else 0
        return jsonify({'message': f"Total O/P_Wt for {production_name} in FY{fy_year}: {total}"})
    except Exception as e:
        return jsonify({'message': f"Error: {str(e)}"}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == '__main__':
    app.run(debug=True)
