from datetime import datetime
from typing import Annotated, Self

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from sema_module.sema.codec import SemaCodec

from ..dependencies import get_db

from gw_data.db.models import (
    MessageSql,
)

router = APIRouter()


class MessagesQueryParams(BaseModel):
    start: datetime
    end: datetime
    message_types: str = Field("")

    @model_validator(mode="after")
    def check_start_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("end_time must be after start_time")
        return self

ALLOWED_MESSAGE_TYPES = {
    'weather.forecast',
    'glitch',
    'gridworks.event.problem'
}

@router.get("/api/v2/installations/{installation_id}/messages")
def get_messages(
    installation_id,
    query: Annotated[MessagesQueryParams, Query()],
    db: Session = Depends(get_db),
):
    db_message_types = ALLOWED_MESSAGE_TYPES.intersection(query.message_types.split(','))
    db_query = (
        select(MessageSql.payload)
        .order_by(MessageSql.timestamp)
        .filter(
            MessageSql.timestamp >= query.start,
            MessageSql.timestamp <= query.end,
            MessageSql.message_type_name.in_(db_message_types),
            MessageSql.from_alias.like(f'%{installation_id}%')
        )
        .order_by(MessageSql.timestamp)
        .limit(1000)
    )

    db_results = db.execute(db_query).all()

    codec = SemaCodec()
    sema_results = [codec.from_dict(x[0]) for x in db_results]
    return sema_results
