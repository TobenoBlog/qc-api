from fastapi import Request, HTTPException
from typing import Dict, Any
import jwt
from jwt import InvalidTokenError, ExpiredSignatureError
from .settings import JWT_SECRET, JWT_ALG, ALLOWED_ISS

def http_401(detail: str = "Unauthorized") -> HTTPException:
    return HTTPException(status_code=401, detail=detail)

def extract_token(request: Request) -> str:
    """優先順位:
    1) Authorization: Bearer <jwt>
    2) Cookie: qc_jwt=<jwt>
    3) Query: ?jwt=<jwt>
    """
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:].strip()

    cookie_jwt = request.cookies.get("qc_jwt")
    if cookie_jwt:
        return cookie_jwt

    q = request.query_params.get("jwt")
    if q:
        return q

    raise http_401("Missing token")

def verify_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            options={"require": ["exp", "iss"]},
        )
    except ExpiredSignatureError:
        raise http_401("Token expired")
    except InvalidTokenError:
        raise http_401("Invalid token")

    iss = payload.get("iss")
    if iss != ALLOWED_ISS:
        raise http_401("Invalid issuer")

    return payload
