from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UnifiedSample:
    id: str
    image: str | list[str]
    conversations: list[dict[str, str]]
    width: int | None = None
    height: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "image": self.image,
            "conversations": self.conversations,
        }
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        if self.meta:
            out["meta"] = self.meta
        return out


def count_image_tags(conversations: list[dict[str, str]]) -> int:
    return sum(turn.get("value", "").count("<image>") for turn in conversations)


def validate_sample(sample: UnifiedSample) -> str | None:
    if isinstance(sample.image, str):
        images = [sample.image]
    else:
        images = list(sample.image)

    if not images:
        return "missing_image"

    for path in images:
        if not Path(path).is_file():
            return "missing_image"

    if sample.conversations:
        n_tags = count_image_tags(sample.conversations)
        if n_tags != len(images):
            return "image_tag_mismatch"

        for turn in sample.conversations:
            if turn.get("from") not in ("human", "gpt"):
                return "invalid_role"
            if not turn.get("value", "").strip():
                return "empty_turn"

    return None


def caption_conversation(caption: str, *, human_prompt: str = "<image>") -> list[dict[str, str]]:
    return [
        {"from": "human", "value": human_prompt},
        {"from": "gpt", "value": caption.strip()},
    ]


def qa_conversation(question: str, answer: str) -> list[dict[str, str]]:
    q = question.strip()
    if not q.startswith("<image>"):
        q = f"<image>\n{q}"
    return [
        {"from": "human", "value": q},
        {"from": "gpt", "value": answer.strip()},
    ]
