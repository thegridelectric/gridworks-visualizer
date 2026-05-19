from datetime import datetime, timedelta, timezone
import os
import threading
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from gw_data.db.models import UserSql

ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v2/sessions")

vis_sessionmaker = None
vis_sessionmaker_lock = threading.Lock()

def get_sessionmaker():
    global vis_sessionmaker
    if vis_sessionmaker is None:
        with vis_sessionmaker_lock:
            if vis_sessionmaker is None:
                url = os.getenv('VIS_DB2_URL')
                if not url:
                    raise ValueError('VIS_DB2_URL env variable is undefined')

                engine = create_engine(url, echo=True)
                vis_sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return vis_sessionmaker

def get_db():
    sessionmaker = get_sessionmaker()
    db = sessionmaker()
    try:
        yield db
    finally:
        db.close()

ACCESS_TOKEN_EXPIRE_MINUTES = 7 * 24 * 60

def encode_token(username) -> str:
    secret_key = os.getenv('VIS_SECRET_KEY')
    if not secret_key:
        raise ValueError('No secret key available')

    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": expires}, key = secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    secret_key = os.getenv('VIS_SECRET_KEY')
    if not secret_key:
        raise ValueError('VIS_SECRET_KEY env variable is undefined')
    
    return jwt.decode(token, key = secret_key, algorithms=[ALGORITHM])

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserSql:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.execute(select(UserSql).where(UserSql.username == username)).unique().scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    return user