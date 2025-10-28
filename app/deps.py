from fastapi import Depends, Request
from typing import Dict, Any
from .security import extract_token, verify_jwt

def get_token(request: Request) -> str:
    return extract_token(request)

def get_current_claims(token: str = Depends(get_token)) -> Dict[str, Any]:
    return verify_jwt(token)

def get_current_user_id(claims: Dict[str, Any] = Depends(get_current_claims)) -> str:
    sub = claims.get("sub")
    return str(sub or "guest")
