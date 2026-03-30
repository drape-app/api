import logging
import httpx
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer()
_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        logger.info("Fetched Supabase JWKS (%d keys)", len(_jwks_cache.get("keys", [])))
    return _jwks_cache


def _decode_supabase_jwt(token: str) -> dict:
    """Verify Supabase JWT (HS256 or ES256) and return payload."""
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")

        if alg in ("RS256", "ES256"):
            jwks = _get_jwks()
            kid = header.get("kid")
            key = next((k for k in jwks["keys"] if k.get("kid") == kid), None) \
                  or jwks["keys"][0]
            payload = jwt.decode(token, key, algorithms=[alg], options={"verify_aud": False})
        else:
            payload = jwt.decode(
                token, settings.supabase_jwt_secret,
                algorithms=["HS256"], options={"verify_aud": False},
            )

        return payload
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """FastAPI dependency — returns {'user_id': str, 'email': str | None}."""
    payload = _decode_supabase_jwt(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing 'sub' claim")
    return {"user_id": user_id, "email": payload.get("email")}
