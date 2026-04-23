from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_settings


DIRECT_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是产品资料智能客服助手。当前问题被识别为普通闲聊或通用问题，"
            "请直接、友好、简洁地回答。不要声称已经检索产品资料，也不要编造产品参数。",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


class DirectChatChain:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = ChatTongyi(
            model_name=self.settings.dashscope_chat_model,
            temperature=0.6,
        )
        self.chain = DIRECT_CHAT_PROMPT | self.llm | StrOutputParser()

    def invoke(self, question: str, history: list[dict[str, str]]) -> str:
        if not self.settings.dashscope_api_key:
            return (
                "当前未配置 DASHSCOPE_API_KEY，无法调用通义千问进行直接闲聊回答。\n\n"
                f"你的问题是：{question}"
            )
        return self.chain.invoke(
            {
                "question": question,
                "history": self._to_langchain_messages(history),
            }
        ).strip()

    def _to_langchain_messages(self, history: list[dict[str, str]]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages
