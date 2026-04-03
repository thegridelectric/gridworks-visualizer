# from fastapi import APIRouter, Response, status
# from pydantic import BaseModel

# router = APIRouter()

# @router.get('/api/v2/session')
# def get_session(response: Response):
#     response.status_code = status.HTTP_204_NO_CONTENT
#     return None

# class SessionPostRequest(BaseModel):
#     email: str
#     password: str

# @router.post('/api/v2/session', status_code=201)
# def create_session(request: SessionPostRequest):
#     user = db.execute(users.select().where(users.c.username == request.email)).first()
#     if not user or not verify_password(request.password, user.hashed_password):
#         raise HTTPException(
#             status_code=401,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     access_token_expires = timedelta(minutes=gbo_access_token_expire_minutes)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
    
#     # Update last login with timezone-aware datetime
#     db.execute(
#         users.update().where(users.c.username == user.username).values(
#             last_login=datetime.now(timezone.utc)
#         )
#     )
#     db.commit()
    
#     return {"access_token": access_token, "token_type": "bearer"}



# print(f'user={request.email},pw={request.password}')