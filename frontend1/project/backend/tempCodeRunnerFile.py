from flask import Flask, request, render_template, redirect, url_for
from mail_utils import send_email
import random, json, os

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_FILE = "otp_store.json"
ADMIN_EMAIL = "admin@gmail.com"  # Admin’s email address

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/<dept>')
def dept_page(dept):
    return render_template(f"{dept}.html")

@app.route('/send_otp', methods=['POST'])
def send_otp():
    email = request.form['email']
    username = request.form['username']
    department = request.form['department']
    otp = str(random.randint(100000, 999999))

    users = load_data()
    users.append({
        "email": email,
        "username": username,
        "department": department,
        "otp": otp,
        "approved": False,
        "password": None
    })
    save_data(users)

    send_email(email, "Your OTP", f"Your OTP is: {otp}")

    # Admin review link
    review_link = f"http://localhost:5000/review?email={email}"
    admin_msg = f"""
    A login request was received from:

    Name: {username}
    Email: {email}
    Department: {department}

    Click to review and accept/decline: {review_link}
    """
    send_email(ADMIN_EMAIL, "Review Login Request", admin_msg)
    return render_template("success.html")

@app.route('/review')
def review():
    email = request.args.get("email")
    users = load_data()
    user = next((u for u in users if u["email"] == email), None)
    if not user:
        return "User not found", 404
    return render_template("review.html", user=user)

@app.route('/process_review', methods=['POST'])
def process_review():
    email = request.form["email"]
    action = request.form["action"]
    users = load_data()
    for user in users:
        if user["email"] == email:
            if action == "accept":
                user["approved"] = True
                # Send set password email
                link = f"http://localhost:5000/set_password?email={email}"
                send_email(email, "Request Approved", f"Your request has been approved. Click here to set your password: {link}")
            else:
                send_email(email, "Request Declined", "Sorry, your login request has been declined.")
            break
    save_data(users)
    return "Action processed."

@app.route('/set_password', methods=['GET', 'POST'])
def set_password():
    email = request.args.get("email") or request.form.get("email")
    if request.method == 'POST':
        password = request.form["password"]
        users = load_data()
        for user in users:
            if user["email"] == email:
                user["password"] = password
                break
        save_data(users)
        return "Password set successfully. You can now login."
    return render_template("set_password.html", email=email)

if __name__ == '__main__':
    app.run(debug=True)