import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import jwt  # PyJWT

JWT_SECRET = os.getenv("JWT_SIGNING_KEY", "change-me")
JWT_ALG = "HS256"
ALLOWED_ISS = os.getenv("ALLOWED_ISS", "")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://your-wp-site.com")

app = FastAPI(title="QC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.query_params.get("jwt")

def verify_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if ALLOWED_ISS and payload.get("iss") != ALLOWED_ISS:
            raise jwt.InvalidIssuerError("bad iss")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid token: {e}")

async def require_user(request: Request) -> Dict[str, Any]:
    token = extract_token(request)
    if not token:
        raise HTTPException(401, "Missing token")
    return verify_jwt(token)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/practice")
def practice(user=Depends(require_user)):
    return {"ok": True, "user": {"id": user.get("sub"), "name": user.get("name")}}
