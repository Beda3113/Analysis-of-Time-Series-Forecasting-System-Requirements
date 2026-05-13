from fastapi import APIRouter, FastAPI
from src.api.endpoints import auth

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router, prefix="/auth")

def register_routers(app: FastAPI) -> None:
    app.include_router(api_router)
