from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime

router = APIRouter(tags=["Authentication"])

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    is_active: bool
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse

# Временное хранилище пользователей (в памяти)
users_db = {}

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    user_id = str(uuid.uuid4())
    users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "password": user_data.password,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    return TokenResponse(
        access_token=f"token_{user_id}",
        refresh_token=f"refresh_{user_id}",
        token_type="bearer",
        expires_in=3600,
        user=UserResponse(
            id=user_id,
            email=user_data.email,
            name=user_data.name,
            is_active=True,
            created_at=datetime.utcnow()
        )
    )

@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    if user_data.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = users_db[user_data.email]
    if user["password"] != user_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return TokenResponse(
        access_token=f"token_{user['id']}",
        refresh_token=f"refresh_{user['id']}",
        token_type="bearer",
        expires_in=3600,
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            is_active=True,
            created_at=user["created_at"]
        )
    )

@router.get("/me")
async def get_me():
    # Временная заглушка
    return {"message": "Get current user"}
