import json

from app.core.config import get_settings


class AliyunOCRClient:
    def recognize_png(self, image_bytes: bytes) -> str:
        settings = get_settings()
        if not settings.aliyun_access_key_id or not settings.aliyun_access_key_secret:
            raise RuntimeError("阿里云 OCR 未配置：请设置 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET。")

        try:
            from alibabacloud_ocr_api20210707.client import Client as OCRClient
            from alibabacloud_ocr_api20210707 import models as ocr_models
            from alibabacloud_tea_openapi import models as open_api_models
        except ImportError as exc:
            raise RuntimeError(
                "阿里云 OCR SDK 未安装。请按 README 安装可选 OCR 依赖后再处理扫描件 PDF。"
            ) from exc

        config = open_api_models.Config(
            access_key_id=settings.aliyun_access_key_id,
            access_key_secret=settings.aliyun_access_key_secret,
        )
        config.endpoint = settings.aliyun_ocr_endpoint
        client = OCRClient(config)
        request = ocr_models.RecognizeGeneralRequest(body=image_bytes)
        response = client.recognize_general(request)
        data = getattr(getattr(response, "body", None), "data", "")
        return self._extract_text(data)

    def _extract_text(self, data: object) -> str:
        if not data:
            return ""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                return data
        else:
            parsed = data
        if isinstance(parsed, dict):
            words = []
            for item in parsed.get("prism_wordsInfo", []) or parsed.get("wordsInfo", []):
                word = item.get("word") or item.get("text")
                if word:
                    words.append(word)
            if words:
                return "\n".join(words)
            return json.dumps(parsed, ensure_ascii=False)
        return str(parsed)
