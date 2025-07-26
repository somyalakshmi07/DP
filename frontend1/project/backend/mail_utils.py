# import smtplib, ssl
# from email.message import EmailMessage

# SENDER_EMAIL = "suraksha.doijodes@gmail.com"
# SENDER_PASSWORD = "eyqkfzuezrkmlske"  # App Password (not Gmail password)

# def send_email(to, subject, body):
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg['Subject'] = subject
#     msg['From'] = SENDER_EMAIL
#     msg['To'] = to

#     context = ssl.create_default_context()
#     with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
#         server.login(SENDER_EMAIL, SENDER_PASSWORD)
#         server.send_message(msg)

# import smtplib
# from email.message import EmailMessage

# SENDER_EMAIL = "youremail@gmail.com"
# SENDER_PASSWORD = "your_app_password"

# def send_email(to, subject, content):
#     msg = EmailMessage()
#     msg.set_content(content)
#     msg["Subject"] = subject
#     msg["From"] = SENDER_EMAIL
#     msg["To"] = to

#     with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
#         smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
#         smtp.send_message(msg)

import smtplib, ssl
from email.message import EmailMessage

SENDER_EMAIL = "sukku1681@gmail.com"  # New Gmail address
SENDER_PASSWORD = "tfftrwdbbmjvegtg"  # New App Password

def send_email(to, subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

