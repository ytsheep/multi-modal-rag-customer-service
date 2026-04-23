import re
from dataclasses import dataclass
from enum import StrEnum


class Intent(StrEnum):
    DIRECT_CHAT = "direct_chat"
    RAG_QA = "rag_qa"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    reason: str


class IntentRouter:
    """Rule-based router for product manual assistant scenarios."""

    PRODUCT_KEYWORDS = {
        "产品",
        "型号",
        "订货号",
        "手册",
        "说明书",
        "资料",
        "规格",
        "参数",
        "技术参数",
        "额定",
        "输入",
        "输出",
        "电压",
        "电流",
        "功率",
        "频率",
        "温度",
        "湿度",
        "防护等级",
        "ip20",
        "ip65",
        "安装",
        "导轨",
        "接线",
        "端子",
        "力矩",
        "调试",
        "运行",
        "操作",
        "报警",
        "故障",
        "代码",
        "维护",
        "安全",
        "注意事项",
        "通信",
        "协议",
        "profinet",
        "modbus",
        "plc",
        "cpu",
        "电源",
        "对比",
        "区别",
        "差异",
        "相同",
        "引用",
        "原文",
        "依据",
        "第",
        "页",
        "pdf",
        "文档",
    }

    SMALL_TALK_PATTERNS = [
        r"^(你好|您好|哈喽|hello|hi|嗨)[呀啊嘛!！。.\s]*$",
        r"^(谢谢|感谢|多谢|辛苦了)[呀啊嘛!！。.\s]*$",
        r"^(再见|拜拜|bye)[呀啊嘛!！。.\s]*$",
        r"^(你是谁|你能做什么|介绍一下你自己)[?？。.\s]*$",
        r"^(讲个笑话|随便聊聊|陪我聊聊)[吧呀啊嘛!！。.\s]*$",
    ]

    GENERAL_PREFIXES = (
        "什么是",
        "为什么",
        "怎么理解",
        "解释一下",
        "简单介绍",
    )

    def classify(self, question: str) -> IntentResult:
        text = question.strip().lower()
        if not text:
            return IntentResult(Intent.DIRECT_CHAT, 1.0, "empty question")

        if self._looks_like_product_question(text):
            return IntentResult(Intent.RAG_QA, 0.9, "contains product manual keywords")

        if any(re.match(pattern, text, re.IGNORECASE) for pattern in self.SMALL_TALK_PATTERNS):
            return IntentResult(Intent.DIRECT_CHAT, 0.95, "matched small-talk pattern")

        if text.startswith(self.GENERAL_PREFIXES):
            return IntentResult(Intent.DIRECT_CHAT, 0.76, "general explanation question")

        if len(text) <= 16:
            return IntentResult(Intent.DIRECT_CHAT, 0.72, "short non-product utterance")

        return IntentResult(Intent.RAG_QA, 0.62, "default to product manual QA for safer grounding")

    def _looks_like_product_question(self, text: str) -> bool:
        if any(keyword in text for keyword in self.PRODUCT_KEYWORDS):
            return True
        if re.search(r"(a产品|b产品|产品a|产品b|西门子|simatic|sitop|s7-\d+)", text):
            return True
        if re.search(r"(第\s*\d+\s*页|\d+\s*(v|a|w|hz|mm|n·m|nm|℃|°c))", text, re.IGNORECASE):
            return True
        return False
