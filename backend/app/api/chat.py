from fastapi import APIRouter

from app.agent.service import AgentService
from app.models import ChatRequest, ChatResponse


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    service = AgentService()
    return service.answer(request)

