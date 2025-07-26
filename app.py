# from flask import Flask, render_template, request, redirect, session, flash
# import sqlite3, random, smtplib
# from datetime import datetime, timedelta
# from config import *

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # ---- Utility functions ----
# def send_email(receiver, subject, body):
#     with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
#         smtp.starttls()
#         smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
#         msg = f"Subject: {subject}\n\n{body}"
#         smtp.sendmail(EMAIL_ADDRESS, receiver, msg)

# def generate_otp():
#     return str(random.randint(100000, 999999))

# # ---- Routes ----

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         team = request.form['team']
#         email = request.form['email']

#         # Check if already registered
#         conn = sqlite3.connect('login.db')
#         c = conn.cursor()
#         c.execute("SELECT * FROM users WHERE email=?", (email,))
#         user = c.fetchone()

#         if user:
#             flash("You are already registered. Please log in.")
#             return redirect('/login')

#         # Notify admin and head
#         subject = "Login Request Approval"
#         body = f"A new user with email {email} is requesting access to the {team} team.\n\nApprove: http://localhost:5000/approve?email={email}&team={team}\nReject: http://localhost:5000/reject?email={email}"
        
#         send_email(ADMIN_EMAIL, subject, body)
#         if team == 'production':
#             send_email(PRODUCTION_HEAD_EMAIL, subject, body)
#         else:
#             send_email(MARKETING_HEAD_EMAIL, subject, body)

#         flash("Request sent. Await admin approval.")
#         return redirect('/')

#     return render_template('home.html')


# @app.route('/approve')
# def approve():
#     email = request.args.get('email')
#     team = request.args.get('team')
#     otp = generate_otp()
#     otp_time = datetime.now().isoformat()

#     # Save to pending
#     conn = sqlite3.connect('login.db')
#     c = conn.cursor()
#     c.execute("INSERT INTO pending_requests (email, team, otp, otp_time) VALUES (?, ?, ?, ?)",
#               (email, team, otp, otp_time))
#     conn.commit()
#     conn.close()

#     send_email(email, "OTP for Verification", f"Your OTP is: {otp}\nValid for 5 minutes.")
#     return "Approved! OTP sent to user."


# @app.route('/reject')
# def reject():
#     email = request.args.get('email')
#     send_email(email, "Login Request Rejected", "Your request has been rejected by the admin.")
#     return "Rejected."


# @app.route('/verify', methods=['GET', 'POST'])
# def verify():
#     if request.method == 'POST':
#         email = request.form['email']
#         otp = request.form['otp']

#         conn = sqlite3.connect('login.db')
#         c = conn.cursor()
#         c.execute("SELECT otp, otp_time, team FROM pending_requests WHERE email=?", (email,))
#         record = c.fetchone()
#         if not record:
#             return "No OTP found. Try again."

#         saved_otp, otp_time_str, team = record
#         otp_time = datetime.fromisoformat(otp_time_str)

#         if datetime.now() - otp_time > timedelta(minutes=5):
#             return "OTP expired."

#         if otp == saved_otp:
#             session['email'] = email
#             session['team'] = team
#             return redirect('/register')
#         else:
#             return "Invalid OTP."
#     return render_template('otp_verify.html')


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if 'email' not in session:
#         return redirect('/')

#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         conn = sqlite3.connect('login.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO users (email, username, password, team, verified) VALUES (?, ?, ?, ?, 1)",
#                   (session['email'], username, password, session['team']))
#         c.execute("DELETE FROM pending_requests WHERE email=?", (session['email'],))
#         conn.commit()
#         conn.close()

#         flash("Registered successfully!")
#         session.clear()
#         return redirect('/login')

#     return render_template('register.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         conn = sqlite3.connect('login.db')
#         c = conn.cursor()
#         c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
#         user = c.fetchone()
#         conn.close()

#         if user:
#             return f"Welcome back, {user[2]} ({user[4]})"
#         else:
#             return "Invalid credentials."

#     return render_template('login.html')

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, redirect, session, flash
import sqlite3, random, smtplib
from datetime import datetime, timedelta
from config import *

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ---- Utility functions ----
def send_email(receiver, subject, body):
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        msg = f"Subject: {subject}\n\n{body}"
        smtp.sendmail(EMAIL_ADDRESS, receiver, msg)

def generate_otp():
    return str(random.randint(100000, 999999))

# ---- Routes ----

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        team = request.form['team']
        email = request.form['email']

        # Check if already registered
        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()

        if user:
            flash("You are already registered. Please log in.")
            return redirect('/login')

        # Notify admin and head
        subject = "Login Request Approval"
        body = f"A new user with email {email} is requesting access to the {team} team.\n\nApprove: http://localhost:5000/approve?email={email}&team={team}\nReject: http://localhost:5000/reject?email={email}"
        
        send_email(ADMIN_EMAIL, subject, body)
        if team == 'production':
            send_email(PRODUCTION_HEAD_EMAIL, subject, body)
        else:
            send_email(MARKETING_HEAD_EMAIL, subject, body)

        flash("Request sent. Await admin approval.")
        return redirect(f'/verify?email={email}')

    return render_template('home.html')


@app.route('/approve')
def approve():
    email = request.args.get('email')
    team = request.args.get('team')
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
    send_email(email, "Login Request Rejected", "Your request has been rejected by the admin.")
    return "Rejected."


@app.route('/verify', methods=['GET', 'POST'])
def verify():
    email_prefill = request.args.get('email', '')
    if request.method == 'POST':
        email = request.form['email']
        otp = request.form['otp']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT otp, otp_time, team FROM pending_requests WHERE email=?", (email,))
        record = c.fetchone()
        if not record:
            return "No OTP found. Try again."

        saved_otp, otp_time_str, team = record
        otp_time = datetime.fromisoformat(otp_time_str)

        if datetime.now() - otp_time > timedelta(minutes=5):
            return "OTP expired."

        if otp == saved_otp:
            session['email'] = email
            session['team'] = team
            return redirect('/register')
        else:
            return "Invalid OTP."
    return render_template('otp_verify.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'email' not in session:
        return redirect('/')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (email, username, password, team, verified) VALUES (?, ?, ?, ?, 1)",
                  (session['email'], username, password, session['team']))
        c.execute("DELETE FROM pending_requests WHERE email=?", (session['email'],))
        conn.commit()
        conn.close()

        flash("Registered successfully!")
        session.clear()
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('login.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            return f"Welcome back, {user[2]} ({user[4]})"
        else:
            return "Invalid credentials."

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
