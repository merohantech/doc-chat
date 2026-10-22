from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import SettingsDep

router = APIRouter(prefix="/models", tags=["models"])


class ModelsOut(BaseModel):
    default: str
    allowed: list[str]


@router.get("", response_model=ModelsOut)
async def list_models(settings: SettingsDep) -> ModelsOut:
    """Models a new thread may use, so clients don't hard-code the list."""
    return ModelsOut(default=settings.default_chat_model, allowed=settings.allowed_chat_models)
