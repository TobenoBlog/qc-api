import os

JWT_SECRET = os.getenv("JWT_SIGNING_KEY", "qc_secret_2025")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ALLOWED_ISS = os.getenv("ALLOWED_ISS", "https://tobenicelife.com")
