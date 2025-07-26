import sqlite3

conn = sqlite3.connect('login.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    username TEXT,
    password TEXT,
    team TEXT,
    verified INTEGER DEFAULT 0
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS pending_requests (
    email TEXT,
    team TEXT,
    otp TEXT,
    otp_time TEXT
)
''')

conn.commit()
conn.close()
