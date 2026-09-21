import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


STANDALONE_PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$",
    re.IGNORECASE,
)
STANDALONE_ROMAN_NUMBER = re.compile(r"^[ivxlcdm]{2,8}$", re.IGNORECASE)
STANDALONE_URL = re.compile(
    r"^\d{0,4}\s*(?:https?://|www\.)\S+$",
    re.IGNORECASE,
)
DECORATIVE_LINE = re.compile(r"^[\W_]{3,}$")
ATTACHED_ROMAN_HEADING = re.compile(r"^[ivxlcdm]{1,8}([A-Z][A-Z\s]{3,})$")


def parse_pdf(path: Path) -> list[ParsedPage]:
    try:
        reader = PdfReader(path)
        page_lines = [
            _clean_lines(page.extract_text() or "")
            for page in reader.pages
        ]
        repeated_margins = _find_repeated_margin_lines(page_lines)
        pages = []
        for index, lines in enumerate(page_lines, start=1):
            cleaned_lines = _remove_repeated_margins(lines, repeated_margins)
            text = " ".join(cleaned_lines).strip()
            pages.append(ParsedPage(page_number=index, text=text))
    except Exception as exc:
        raise ValueError("Unable to parse PDF") from exc

    usable_pages = [page for page in pages if page.text]
    if not usable_pages:
        raise ValueError("PDF contains no extractable text")
    return usable_pages


def clean_extracted_text(text: str) -> str:
    """Conservatively normalize one extracted PDF page."""
    return " ".join(_clean_lines(text)).strip()


def _clean_lines(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", normalized)

    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line or _is_obvious_artifact(line):
            continue
        attached_heading = ATTACHED_ROMAN_HEADING.fullmatch(line)
        if attached_heading:
            line = attached_heading.group(1).strip()
        lines.append(line)
    return lines


def _is_obvious_artifact(line: str) -> bool:
    return bool(
        STANDALONE_PAGE_NUMBER.fullmatch(line)
        or STANDALONE_ROMAN_NUMBER.fullmatch(line)
        or STANDALONE_URL.fullmatch(line)
        or DECORATIVE_LINE.fullmatch(line)
    )


def _find_repeated_margin_lines(page_lines: list[list[str]]) -> set[str]:
    if len(page_lines) < 3:
        return set()

    margin_keys: list[str] = []
    for lines in page_lines:
        if not lines:
            continue
        for line in {lines[0], lines[-1]}:
            if len(line) <= 120:
                margin_keys.append(_line_key(line))

    required_repetitions = max(3, math.ceil(len(page_lines) * 0.3))
    counts = Counter(margin_keys)
    return {
        line_key
        for line_key, count in counts.items()
        if line_key and count >= required_repetitions
    }


def _remove_repeated_margins(lines: list[str], repeated: set[str]) -> list[str]:
    if not lines or not repeated:
        return lines

    start = 1 if _line_key(lines[0]) in repeated else 0
    end = len(lines) - 1 if _line_key(lines[-1]) in repeated else len(lines)
    return lines[start:end]


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()
