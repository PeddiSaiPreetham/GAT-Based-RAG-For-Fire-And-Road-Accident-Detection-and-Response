# # modules/alerting.py
# """
# Alerting utilities: Twilio SMS, Email via SMTP, Webhook (requests).

# SMTP CONFIG FORMAT (in config.yaml)

# alerts:
#   email:
#     smtp_server: "smtp.gmail.com"
#     smtp_port: 587
#     smtp_user: "your_email@gmail.com"
#     smtp_pass: "your_app_password_here"
#     from: "your_email@gmail.com"   # OPTIONAL (defaults to smtp_user)

# """

# import logging
# log = logging.getLogger("alerting")

# # --------------------
# # Twilio (optional)
# # --------------------
# try:
#     from twilio.rest import Client as TwilioClient
#     TWILIO_AVAILABLE = True
# except Exception:
#     TWILIO_AVAILABLE = False
#     TwilioClient = None

# # --------------------
# # Email + Webhook
# # --------------------
# import smtplib
# from email.message import EmailMessage
# import requests
# import time


# class Alerting:
#     def __init__(self, cfg: dict):
#         """
#         cfg:
#           sid: Twilio SID
#           token: Twilio Token
#           from: Twilio FROM number

#           smtp:
#             server: smtp.gmail.com
#             port: 587
#             user: your_email
#             pass: your_password
#             from: optional_from_email
#         """

#         # ----------------------
#         # TWILIO SETUP
#         # ----------------------
#         self.tw_sid = cfg.get("sid")
#         self.tw_token = cfg.get("token")
#         self.tw_from = cfg.get("from")
#         self.tw_client = None

#         if self.tw_sid and TWILIO_AVAILABLE:
#             try:
#                 self.tw_client = TwilioClient(self.tw_sid, self.tw_token)
#             except Exception as e:
#                 log.warning("Twilio init failed: %s", e)

#         # ----------------------
#         # SMTP SETUP
#         # ----------------------
#         self.smtp = cfg.get("smtp")
#         if self.smtp:
#             required = ["server", "port", "user", "pass"]
#             missing = [k for k in required if k not in self.smtp]
#             if missing:
#                 log.warning("SMTP config missing: %s", missing)

#     # =======================================================
#     # SMS
#     # =======================================================
#     def send_sms(self, to, body):
#         if not self.tw_client:
#             log.info("Twilio not configured → SMS skipped.")
#             return
#         try:
#             self.tw_client.messages.create(
#                 body=body,
#                 from_=self.tw_from,
#                 to=to
#             )
#             log.info("SMS sent to %s", to)
#         except Exception as e:
#             log.warning("SMS failed: %s", e)

#     # =======================================================
#     # EMAIL (robust version with UTF-8 + retry + HTML fallback)
#     # =======================================================
#     def send_email(self, to, subject, body):
#         if not self.smtp:
#             log.info("SMTP not configured → email skipped.")
#             return

#         msg = EmailMessage()
#         msg["Subject"] = subject
#         msg["From"] = self.smtp.get("from") or self.smtp.get("user")
#         msg["To"] = to

#         # Plain text + HTML alternative
#         msg.set_content(body, charset="utf-8")
#         msg.add_alternative(f"<pre>{body}</pre>", subtype="html")

#         server_addr = self.smtp["server"]
#         port = int(self.smtp.get("port", 587))
#         user = self.smtp["user"]    # LOGIN EMAIL
#         pwd = self.smtp["pass"]     # APP PASSWORD (if Gmail)

#         # Retry SMTP up to 3 times
#         for attempt in range(3):
#             try:
#                 with smtplib.SMTP(server_addr, port, timeout=10) as s:
#                     s.starttls()
#                     s.login(user, pwd)
#                     s.send_message(msg)

#                 log.info("Email sent to %s", to)
#                 return

#             except Exception as e:
#                 log.warning(f"Email attempt {attempt+1}/3 failed: {e}")
#                 time.sleep(1)

#         log.error("Email failed after 3 attempts.")

#     # =======================================================
#     # WEBHOOK
#     # =======================================================
#     def send_webhook(self, url, payload, timeout=5):
#         try:
#             requests.post(url, json=payload, timeout=timeout)
#             log.info("Webhook posted to %s", url)
#         except Exception as e:
#             log.warning("Webhook failed: %s", e)


# modules/alerting.py (DEBUG VERSION)
import logging
log = logging.getLogger("alerting")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TwilioClient = None
    TWILIO_AVAILABLE = False

import smtplib
from email.message import EmailMessage
import requests


class Alerting:
    def __init__(self, cfg: dict):
        # Twilio -------------------------
        self.tw_sid = cfg.get("sid")
        self.tw_token = cfg.get("token")
        self.tw_from = cfg.get("from")

        print("Twilio SID:", self.tw_sid)
        print("Twilio FROM:", self.tw_from)

        self.tw_client = None
        if self.tw_sid and TWILIO_AVAILABLE:
            try:
                self.tw_client = TwilioClient(self.tw_sid, self.tw_token)
                print("Twilio client initialized successfully.")
            except Exception as e:
                print("Twilio initialization failed:", e)
                self.tw_client = None

        # SMTP ---------------------------
        self.smtp = cfg.get("email") or cfg.get("smtp")

        if not self.smtp:
            print("SMTP IS MISSING!!")


    # ----------------------------------------------------------------------
    def send_sms(self, to, body):
        if not self.tw_client:
            print("Twilio NOT configured. SMS skipped.")
            return

        try:
            self.tw_client.messages.create(body=body, from_=self.tw_from, to=to)
            print(f"SMS sent to {to}")
        except Exception as e:
            print("SMS FAILED:", e)


    # ----------------------------------------------------------------------
    def send_email(self, to, subject, body):
        if not self.smtp:
            print("SMTP configuration missing, EMAIL SKIPPED")
            return

        server = self.smtp.get("smtp_server")
        port   = int(self.smtp.get("smtp_port", 587))
        user   = self.smtp.get("smtp_user")
        pwd    = self.smtp.get("smtp_pass")
        sender = self.smtp.get("from") or user

        # Validate
        if not server or not user or not pwd:
            print("ERROR: One or more SMTP fields were empty, email skipped ❌")
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = to
            msg.set_content(body)

            print("Connecting to SMTP server...")
            with smtplib.SMTP(server, port, timeout=15) as s:
                s.starttls()
                #print("SMTP TLS handshake OK")
                s.login(user, pwd)
                #print("SMTP LOGIN SUCCESS")
                s.send_message(msg)
                #print("EMAIL SENT SUCCESSFULLY")

        except Exception as e:
            print("EMAIL FAILED →", e)



    # ----------------------------------------------------------------------
    def send_webhook(self, url, payload, timeout=5):
        try:
            requests.post(url, json=payload, timeout=timeout)
            print("Webhook sent to", url)
        except Exception as e:
            print("Webhook FAILED:", e)
