# fix_booking_date_migration.py
import sqlite3
from datetime import datetime

def fix_booking_date_tables():
    conn = sqlite3.connect('production_planning.db')
    c = conn.cursor()
    
    # Get all tables that match the pattern orders_*
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'orders_%'")
    tables = [row[0] for row in c.fetchall()]
    
    for table_name in tables:
        print(f"Checking table: {table_name}")
        
        # Check if table has booking_date column with proper default
        c.execute(f"PRAGMA table_info({table_name})")
        columns_info = c.fetchall()
        booking_date_column = None
        
        for column in columns_info:
            if column[1] == 'booking_date':
                booking_date_column = column
                break
        
        # Recreate table with proper default constraint
        print(f"Recreating table: {table_name}")
        
        # Get all data from current table
        c.execute(f"SELECT * FROM {table_name}")
        rows = c.fetchall()
        column_names = [description[0] for description in c.description]
        
        # Drop old table
        c.execute(f"DROP TABLE {table_name}")
        
        # Create new table with proper schema
        c.execute(f"""CREATE TABLE {table_name}
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      unit TEXT,
                      product_type TEXT,
                      tdc TEXT,
                      thickness REAL,
                      zinc REAL,
                      quantity INTEGER,
                      productivity REAL,
                      required_time REAL,
                      booking_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        
        # Reinsert data with current timestamp for null booking dates
        for row in rows:
            row_dict = dict(zip(column_names, row))
            booking_date = row_dict.get('booking_date')
            if booking_date is None:
                booking_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Build the insert statement
            columns = [col for col in column_names if col != 'id']
            placeholders = ', '.join(['?'] * len(columns))
            values = [row_dict[col] for col in columns]
            values[-1] = booking_date  # Replace booking_date
            
            c.execute(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})", values)
        
        print(f"Table {table_name} recreated successfully")
    
    conn.commit()
    conn.close()
    print("Booking date migration completed")

if __name__ == '__main__':
    fix_booking_date_tables()