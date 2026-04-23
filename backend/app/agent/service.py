from app.agent.intent import Intent, IntentRouter
from app.langchain_pipeline.direct_chat import DirectChatChain
from app.langchain_pipeline.rag_chain import LangChainRAGChain
from app.memory.short_term import ShortTermMemory
from app.models import ChatRequest, ChatResponse


class AgentService:
    def __init__(self) -> None:
        self.memory = ShortTermMemory()
        self.intent_router = IntentRouter()
        self.direct_chat_chain = DirectChatChain()
        self.rag_chain = LangChainRAGChain()

    def answer(self, request: ChatRequest) -> ChatResponse:
        history = self.memory.load(request.user_id, request.session_id)
        intent_result = self.intent_router.classify(request.question)
        if intent_result.intent == Intent.DIRECT_CHAT:
            answer = self.direct_chat_chain.invoke(request.question, history)
            citations = []
        else:
            answer, citations = self.rag_chain.invoke(request.question, history)
        self.memory.save_turn(request.user_id, request.session_id, request.question, answer)
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            citations=citations,
            intent=intent_result.intent.value,
            intent_reason=intent_result.reason,
        )
