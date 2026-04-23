from openai import OpenAI

from app.core.config import get_settings


class DashScopeLLM:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def ready(self) -> bool:
        return bool(self.settings.dashscope_api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.ready:
            return self._fallback_answer(messages)

        client = OpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        completion = client.chat.completions.create(
            model=self.settings.dashscope_chat_model,
            messages=messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""

    def _fallback_answer(self, messages: list[dict[str, str]]) -> str:
        user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "当前未配置 DASHSCOPE_API_KEY，系统已完成检索与上下文组装，但未调用千问模型。\n\n"
            f"你的问题是：{user_message}\n\n"
            "请在 backend/.env 或系统环境变量中配置 DASHSCOPE_API_KEY 后重试。"
        )

