from flask import Flask, request, render_template, redirect
from mail_utils import send_email
import random, json, os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

DATA_FILE = "otp_store.json"
ADMIN_EMAIL = "sukku1681@gmail.com"

def save_request(data):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            all_data = json.load(f)
    else:
        all_data = []
    all_data.append(data)
    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/marketing')
def marketing_page():
    return render_template("marketing.html")

@app.route('/production')
def production_page():
    return render_template("production.html")

@app.route('/send_otp', methods=['POST'])
def send_otp():
    email = request.form['email']
    username = request.form['username']
    department = request.form['department']
    otp = str(random.randint(100000, 999999))

    send_email(email, "Your OTP", f"Hello {username}, your OTP is: {otp}")
    save_request({
        "email": email,
        "username": username,
        "department": department,
        "otp": otp,
        "approved": False
    })

    notify = f"User  '{username}' is trying to log in to {department}.\nEmail: {email}\n\n"
    notify += f"<a href='http://localhost:5000/approve?email={email}'>Accept</a> | "
    notify += f"<a href='http://localhost:5000/decline?email={email}'>Decline</a>"

    send_email(ADMIN_EMAIL, f"{department.title()} Login Attempt", notify)

    return render_template("success.html")

@app.route('/approve')
def approve():
    email = request.args.get('email')
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    for user in data:
        if user["email"] == email:
            user["approved"] = True
            break
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Send email with password setup link
    setup_link = f"http://localhost:5000/setup_password?email={email}"
    send_email(email, "Request Approved", f"Your login request has been approved. Please set your password: {setup_link}")
    
    return "User  Approved"

@app.route('/setup_password', methods=['GET', 'POST'])
def setup_password():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Save the password (in a real app, hash it)
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        for user in data:
            if user["email"] == email:
                user["password"] = password  # Store the password
                break
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        return "Password has been set successfully!"

    email = request.args.get('email')
    if email is None:
        return "Email parameter is missing.", 400  # Return an error if email is not provided
    return render_template("setup_password.html", email=email)

@app.route('/decline')
def decline():
    email = request.args.get('email')
    send_email(email, "Request Declined", "Your login request has been declined.")
    return "User  Declined"

if __name__ == '__main__':
    app.run(debug=True)

