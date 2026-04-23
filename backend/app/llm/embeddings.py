from collections.abc import Sequence
from openai import OpenAI

from app.core.config import get_settings


class DashScopeEmbeddingFunction:
    def __init__(self) -> None:
        self.settings = get_settings()

    def name(self) -> str:
        return f"dashscope-{self.settings.dashscope_embedding_model}-{self.settings.embedding_dimensions}"

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self.embed(list(input))

    def embed_query(self, input: str | Sequence[str]) -> list[list[float]]:
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        return self.embed(texts)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        return self.embed(list(input))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法生成 DashScope 文本向量。")

        client = OpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        vectors: list[list[float]] = []
        batch_size = 10
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(
                model=self.settings.dashscope_embedding_model,
                input=batch,
                dimensions=self.settings.embedding_dimensions,
                encoding_format="float",
            )
            sorted_data = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in sorted_data])
        return vectors
