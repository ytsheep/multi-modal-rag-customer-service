from dataclasses import dataclass, field
import re
import unicodedata

from app.ingestion.parser import ParsedBlock


TITLE_PATTERNS = [
    re.compile(r"^(第[一二三四五六七八九十百千万\d]+[章节篇部分])\s*(.+)?$"),
    re.compile(r"^([一二三四五六七八九十]+)[、.]\s*(.+)$"),
    re.compile(r"^([A-Z])\s+[\u4e00-\u9fffA-Za-z].{1,60}$"),
    re.compile(r"^(\d+(?:\.\d+){0,4})\s+[\u4e00-\u9fffA-Za-z].{1,80}$"),
    re.compile(r"^#{1,6}\s+(.+)$"),
]


LOW_VALUE_KEYWORDS = {
    "法律资讯",
    "商标",
    "责任免除",
    "Copyright",
    "保留所有权利",
    "Siemens Aktiengesellschaft",
    "Postfach",
}


@dataclass
class CleaningReport:
    input_blocks: int = 0
    output_blocks: int = 0
    removed_empty: int = 0
    removed_noise: int = 0
    normalized_symbols: int = 0
    merged_lines: int = 0
    title_count: int = 0
    notes: list[str] = field(default_factory=list)


class CleaningPipeline:
    def clean(self, blocks: list[ParsedBlock]) -> tuple[list[ParsedBlock], dict]:
        report = CleaningReport(input_blocks=len(blocks))
        cleaned: list[ParsedBlock] = []
        inferred_title_stack: list[str] = []

        for block in blocks:
            text = self._normalize(block.text, report)
            text = self._remove_noise_lines(text, report)
            if not text.strip():
                report.removed_empty += 1
                continue
            if self._is_low_value_block(text, block.title_path):
                report.removed_noise += 1
                continue

            text = self._repair_lines(text, report)
            title_path = block.title_path.strip()
            title_level = self._title_level(text)
            if title_level and not title_path:
                report.title_count += 1
                inferred_title_stack = inferred_title_stack[: title_level - 1]
                inferred_title_stack.append(self._compact_title(text))
                title_path = " > ".join(inferred_title_stack)
            elif not title_path and inferred_title_stack:
                title_path = " > ".join(inferred_title_stack)

            cleaned.append(
                ParsedBlock(
                    text=text,
                    page=block.page,
                    block_type=block.block_type,
                    title_path=title_path,
                )
            )

        report.output_blocks = len(cleaned)
        return cleaned, report.__dict__

    def _normalize(self, text: str, report: CleaningReport) -> str:
        before = text
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if before != text:
            report.normalized_symbols += 1
        return text.strip()

    def _remove_noise_lines(self, text: str, report: CleaningReport) -> str:
        lines = []
        for line in text.split("\n"):
            raw = line.strip()
            if not raw:
                continue
            if self._is_noise_line(raw):
                report.removed_noise += 1
                continue
            raw = re.sub(r"^[★▶•▪▸→]+\s*", "", raw)
            raw = re.sub(r"^[●•–—]\s*", "- ", raw)
            raw = raw.replace("…", "...")
            lines.append(raw)
        return "\n".join(lines).strip()

    def _repair_lines(self, text: str, report: CleaningReport) -> str:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) <= 1:
            return text
        repaired: list[str] = []
        for line in lines:
            if not repaired:
                repaired.append(line)
                continue
            prev = repaired[-1]
            if self._should_merge(prev, line):
                repaired[-1] = prev + " " + line
                report.merged_lines += 1
            else:
                repaired.append(line)
        return "\n".join(repaired)

    def _should_merge(self, prev: str, current: str) -> bool:
        if self._title_level(prev) or self._title_level(current):
            return False
        if re.match(r"^(\d+[.)、]|\d+\.\d+|[-*+])\s+", current):
            return False
        if prev.endswith(("。", "！", "？", ":", "：", ".", "!", "?", ";", "；")):
            return False
        if "|" in prev or "|" in current:
            return False
        if len(prev) < 12:
            return False
        return True

    def _title_level(self, text: str) -> int | None:
        stripped = self._compact_title(text)
        if stripped.startswith("#"):
            return min(len(stripped) - len(stripped.lstrip("#")), 6)
        for pattern in TITLE_PATTERNS:
            if pattern.match(stripped):
                if re.match(r"^\d+\.\d+\.\d+", stripped):
                    return 3
                if re.match(r"^\d+\.\d+", stripped):
                    return 2
                return 1
        return None

    def _compact_title(self, text: str) -> str:
        first_line = text.strip().split("\n", 1)[0]
        return re.sub(r"\s+", " ", first_line).strip()

    def _is_noise_line(self, line: str) -> bool:
        if len(line) <= 3 and re.fullmatch(r"\d+|[ivxlcdmIVXLCDM]+", line):
            return True
        if re.fullmatch(r"[-_=*~.]{3,}", line):
            return True
        if re.search(r"\.{8,}", line):
            return True
        if line in {"目录", "前言", "LOGO!", "概述"}:
            return True
        if re.match(r"^系统手册,\s*\d{2}/\d{4},\s*A5E", line):
            return True
        if re.match(r"^LOGO!\s+系统手册", line):
            return True
        if re.match(r"^SCALANCE .* C79000", line):
            return True
        if re.match(r"^C79000-G\d+", line):
            return True
        if any(keyword in line for keyword in LOW_VALUE_KEYWORDS):
            return True
        return False

    def _is_low_value_block(self, text: str, title_path: str = "") -> bool:
        stripped = text.strip()
        title = title_path.strip()
        if title and any(part in title for part in {"目录", "法律资讯"}):
            return True
        if len(stripped) < 5 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", stripped):
            return True
        dot_line_count = len(re.findall(r"\.{8,}", stripped))
        if dot_line_count >= 3:
            return True
        keyword_hits = sum(1 for keyword in LOW_VALUE_KEYWORDS if keyword in stripped)
        if keyword_hits >= 2 and len(stripped) < 1500:
            return True
        return False
