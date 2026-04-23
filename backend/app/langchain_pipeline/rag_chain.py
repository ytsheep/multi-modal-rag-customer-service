from operator import itemgetter

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import get_settings
from app.langchain_pipeline.prompts import RAG_PROMPT
from app.langchain_pipeline.retriever import HybridLangChainRetriever, document_to_citation
from app.models import Citation


class LangChainRAGChain:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = HybridLangChainRetriever()
        self.llm = ChatTongyi(
            model_name=self.settings.dashscope_chat_model,
            temperature=0.2,
        )
        answer_chain = (
            {
                "context": lambda inputs: self._format_documents(inputs["documents"]),
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        self.chain = RunnablePassthrough.assign(
            documents=lambda inputs: self.retriever.invoke(inputs["question"])
        ).assign(answer=answer_chain)

    def invoke(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[Citation]]:
        langchain_history = self._to_langchain_messages(history)
        if not self.settings.dashscope_api_key:
            return self._missing_key_answer(question), []

        result = self.chain.invoke({"question": question, "history": langchain_history})
        documents = result.get("documents", [])
        citations = [document_to_citation(document) for document in documents]
        answer = str(result.get("answer", "")).strip()
        if not answer:
            answer = "无法从已上传资料中确认"
        return answer, citations

    def _format_documents(self, documents: list[Document]) -> str:
        if not documents:
            return "暂无相关产品资料片段。"
        blocks = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            file_name = metadata.get("file_name") or "未知文件"
            title_path = metadata.get("title_path") or "未识别章节"
            page_start = metadata.get("page_start") or ""
            page_end = metadata.get("page_end") or ""
            score = float(metadata.get("score") or 0.0)
            page_text = f"{page_start}-{page_end}".strip("-") or "无页码"
            blocks.append(
                "\n".join(
                    [
                        f"[{index}] 来源：{file_name}",
                        f"章节：{title_path}",
                        f"页码：{page_text}",
                        f"相关度：{score:.2f}",
                        f"原文：{document.page_content}",
                    ]
                )
            )
        return "\n\n".join(blocks)

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

    def _missing_key_answer(self, question: str) -> str:
        return (
            "当前未配置 DASHSCOPE_API_KEY，系统已完成产品资料 RAG 链路中的检索与 Prompt 组装，"
            "但未调用通义千问模型。\n\n"
            f"你的问题是：{question}\n\n"
            "请在 backend/.env 中配置 DASHSCOPE_API_KEY 后重试。"
        )
