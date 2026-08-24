"""Splits an earnings-call transcript into speaker turns and tags each with a role.

Motley Fool transcripts are newline-delimited: each line is either a section
marker (``"Prepared Remarks:"``, ``"Questions and Answers:"``), a speaker
header (``"Name -- Title"``, or a three-part ``"Name -- Firm -- Analyst"``
for outside analysts, or a bare ``"Operator"``/``"Interviewer"`` line with no
title), or a paragraph of that speaker's turn. Header lines are identified
heuristically (short, not ending in sentence punctuation) rather than by a
crammed regex over the whole document, since transcript bodies themselves
frequently contain the substring `` -- `` as an em-dash stand-in — see the
false-positive rate found by direct inspection of the raw data (~25% of all
`` -- ``-containing lines are body text, not headers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_HEADER_MAX_LEN = 100
_SENTENCE_ENDINGS = (".", "?", "!", ",", ":", ";")
_BARE_SPEAKER_NAMES = {"operator", "interviewer"}
_SECTION_MARKER_RE = re.compile(r"^[A-Za-z ]{3,40}:$")


class Role(Enum):
    CEO = "ceo"
    CFO = "cfo"
    OTHER_EXEC = "other_exec"
    IR = "ir"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OTHER = "other"


@dataclass(frozen=True)
class Turn:
    speaker: str
    title: str
    role: Role
    section: str | None
    text: str


def classify_role(speaker: str, title: str) -> Role:
    if speaker.strip().lower() in _BARE_SPEAKER_NAMES:
        return Role.OPERATOR
    title_lower = title.lower()
    if "analyst" in title_lower or "interviewer" in title_lower:
        return Role.ANALYST
    if "chief executive" in title_lower or re.search(r"\bceo\b", title_lower):
        return Role.CEO
    if "chief financial" in title_lower or re.search(r"\bcfo\b", title_lower):
        return Role.CFO
    if "investor relations" in title_lower:
        return Role.IR
    if "chief" in title_lower or "president" in title_lower or "officer" in title_lower:
        return Role.OTHER_EXEC
    return Role.OTHER


def _is_header_line(line: str) -> tuple[str, str] | None:
    """Returns (speaker, title) if `line` looks like a speaker header, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.lower() in _BARE_SPEAKER_NAMES:
        return stripped, ""
    if " -- " not in stripped:
        return None
    if len(stripped) > _HEADER_MAX_LEN or stripped.endswith(_SENTENCE_ENDINGS):
        return None
    if not stripped[0].isalpha() or not stripped[0].isupper():
        return None
    speaker, _, title = stripped.partition(" -- ")
    return speaker.strip(), title.strip()


def parse_turns(text: str) -> list[Turn]:
    """Splits a transcript into speaker turns. Consecutive paragraphs under the
    same header are merged into one turn; section markers (e.g. "Prepared
    Remarks:") update `section` for subsequent turns without becoming a turn
    themselves."""
    turns: list[Turn] = []
    section: str | None = None
    current_speaker: str | None = None
    current_title: str | None = None
    current_role: Role | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker is not None and buffer:
            turns.append(
                Turn(
                    speaker=current_speaker,
                    title=current_title or "",
                    role=current_role or Role.OTHER,
                    section=section,
                    text=" ".join(buffer).strip(),
                )
            )
        buffer.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _SECTION_MARKER_RE.match(line):
            flush()
            section = line.rstrip(":")
            current_speaker = None
            continue
        header = _is_header_line(line)
        if header is not None:
            flush()
            current_speaker, current_title = header
            current_role = classify_role(current_speaker, current_title)
            continue
        if current_speaker is None:
            continue
        buffer.append(line)

    flush()
    return turns
