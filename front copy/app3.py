from flask import Flask, render_template, request, redirect, session, flash, url_for
import sqlite3, random, smtplib
from datetime import datetime, timedelta
from config import *

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize database
def init_db():
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  team TEXT NOT NULL,
                  verified INTEGER DEFAULT 0)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS pending_requests
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  team TEXT NOT NULL,
                  otp TEXT NOT NULL,
                  otp_time TEXT NOT NULL)''')
    
    conn.commit()
    conn.close()

# ---- Utility functions ----
def send_email(receiver, subject, body):
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            msg = f"Subject: {subject}\n\n{body}"
            smtp.sendmail(EMAIL_ADDRESS, receiver, msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

def generate_otp():
    return str(random.randint(100000, 999999))

# ---- Routes ----

@app.route('/', methods=['GET', 'POST'])
def homelogin():
    if request.method == 'POST':
        if 'team' not in request.form or 'email' not in request.form:
            flash('Please fill all fields', 'error')
            return redirect(url_for('homelogin'))
            
        team = request.form['team']
        email = request.form['email']

        # Check if already registered
        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()

        if user:
            flash('You are already registered. Please log in.', 'info')
            return redirect(url_for('login'))

        # Notify admin and head
        subject = "Login Request Approval"
        approve_url = f"http://localhost:5000/approve?email={email}&team={team}"
        reject_url = f"http://localhost:5000/reject?email={email}"
        
        body = f"""A new user with email {email} is requesting access to the {team} team.

Approve: {approve_url}
Reject: {reject_url}"""
        
        send_email(ADMIN_EMAIL, subject, body)
        if team == 'production':
            send_email(PRODUCTION_HEAD_EMAIL, subject, body)
        else:
            send_email(MARKETING_HEAD_EMAIL, subject, body)

        flash('Request sent. Await admin approval.', 'success')
        return redirect(url_for('homelogin'))

    return render_template('homelogin.html')

@app.route('/approve')
def approve():
    email = request.args.get('email')
    team = request.args.get('team')
    
    if not email or not team:
        return "Missing parameters", 400
        
    otp = generate_otp()
    otp_time = datetime.now().isoformat()

    # Save to pending
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    c.execute("INSERT INTO pending_requests (email, team, otp, otp_time) VALUES (?, ?, ?, ?)",
              (email, team, otp, otp_time))
    conn.commit()
    conn.close()

    send_email(email, "OTP for Verification", f"Your OTP is: {otp}\nValid for 5 minutes.")
    return "Approved! OTP sent to user."

@app.route('/reject')
def reject():
    email = request.args.get('email')
    if not email:
        return "Missing email parameter", 400
        
    send_email(email, "Login Request Rejected", "Your request has been rejected by the admin.")
    return "Rejected."

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        if 'email' not in request.form or 'otp' not in request.form:
            flash('Please fill all fields', 'error')
            return redirect(url_for('verify'))
            
        email = request.form['email']
        otp = request.form['otp']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT otp, otp_time, team FROM pending_requests WHERE email=?", (email,))
        record = c.fetchone()
        conn.close()

        if not record:
            flash('No OTP found. Try again.', 'error')
            return redirect(url_for('verify'))

        saved_otp, otp_time_str, team = record
        otp_time = datetime.fromisoformat(otp_time_str)

        if datetime.now() - otp_time > timedelta(minutes=5):
            flash('OTP expired.', 'error')
            return redirect(url_for('verify'))

        if otp == saved_otp:
            session['email'] = email
            session['team'] = team
            return redirect(url_for('register'))
        else:
            flash('Invalid OTP.', 'error')
            return redirect(url_for('verify'))
            
    # Pre-fill email if coming from approval
    email_prefill = request.args.get('email', '')
    return render_template('otp_verify.html', email=email_prefill)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'email' not in session:
        flash('Please complete OTP verification first', 'error')
        return redirect(url_for('homelogin'))

    if request.method == 'POST':
        if 'username' not in request.form or 'password' not in request.form:
            flash('Please fill all fields', 'error')
            return redirect(url_for('register'))
            
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (email, username, password, team, verified) VALUES (?, ?, ?, ?, 1)",
                      (session['email'], username, password, session['team']))
            c.execute("DELETE FROM pending_requests WHERE email=?", (session['email'],))
            conn.commit()
            flash('Registered successfully! Please login.', 'success')
            session.clear()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if 'email' not in request.form or 'password' not in request.form:
            flash('Please fill all fields', 'error')
            return redirect(url_for('login'))
            
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['email'] = email
            session['username'] = user[2]
            session['team'] = user[4]
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/home')
def home():
    if 'logged_in' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('homelogin'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)