from typing import Any
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from sqlalchemy import select
from sqlalchemy.orm import Session

from gw_data.db.models import UserSql

from ..dependencies import encode_token, get_current_user, get_db

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12, bcrypt__ident="2b")


class SessionToken(BaseModel):
    access_token: str
    token_type: str


class InstallationRole(BaseModel):
    role: str
    g_node_alias: str
    display_name: str
    address: dict[Any, Any]
    alert_status: dict[Any, Any]
    commit: str

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True, 
    )


class CurrentUser(BaseModel):
    id: uuid.UUID
    username: str
    installation_roles: list[InstallationRole]

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True, 
    )

@router.post("/api/v2/sessions", response_model=SessionToken)
def create_session(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.execute(
        select(UserSql).where(UserSql.username == form_data.username)
    ).unique().scalar_one_or_none()

    if user is None or not user.is_active or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = encode_token(user.username)
    return SessionToken(access_token=token, token_type="bearer")


@router.get("/api/v2/sessions/me", response_model=CurrentUser)
def get_session(current_user: UserSql = Depends(get_current_user)):
    roles = [
        InstallationRole(
            role=r.role, 
            g_node_alias=i.g_node.alias, 
            display_name=i.display_name,
            commit=i.scada_git_commit,
            address=i.address,
            alert_status = i.alert_status
        ) 
        for r in current_user.installation_roles
        for i in r.installations
    ]
    return CurrentUser(id=current_user.id, username=current_user.username, installation_roles=roles)
