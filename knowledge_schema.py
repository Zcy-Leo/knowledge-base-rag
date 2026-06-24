"""
knowledge_schema.py
============================
Defines the structured knowledge entry JSON Schema for the enterprise knowledge base.
Supports knowledge types: SOP steps, FAQ, manual sections, general paragraphs.
"""

import uuid
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional
from datetime import datetime


# Knowledge type enum
KnowledgeType = Literal["sop_step", "faq", "manual_section", "general"]


@dataclass
class KnowledgeEntry:
    """
    Structured knowledge entry schema.

    Fields:
        id           : Unique identifier (auto-generated UUID)
        type         : Knowledge type: sop_step | faq | manual_section | general
        title        : Entry title or question
        content      : Body content (supports Markdown with tables)
        source_file  : Source file name
        source_page  : Source page number (1-based)
        keywords     : Keyword list for retrieval augmentation
        created_at   : Creation timestamp (ISO format)
        metadata     : Extension metadata dict
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: KnowledgeType = "general"
    title: str = ""
    content: str = ""
    source_file: str = ""
    source_page: int = 0
    keywords: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to standard dict (JSON-serializable)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_chroma_text(self) -> str:
        """
        Generate text for vector storage (title + content combined).
        Includes both title and body semantics for RAG retrieval.
        """
        parts = []
        if self.title:
            parts.append(f"[Title] {self.title}")
        if self.content:
            parts.append(f"[Content] {self.content}")
        if self.keywords:
            parts.append(f"[Keywords] {', '.join(self.keywords)}")
        return "\n".join(parts)

    def to_chroma_metadata(self) -> dict:
        """Generate Chroma metadata dict (only simple serializable types, no nested dicts)."""
        base = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "keywords": ", ".join(self.keywords) if self.keywords else "",
            "created_at": self.created_at,
        }
        if self.metadata and isinstance(self.metadata, dict):
            for key, value in self.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    base[key] = value
                elif isinstance(value, list):
                    base[key] = ", ".join(str(v) for v in value)
                else:
                    base[key] = str(value)
        else:
            base["company"] = ""
            base["topic"] = ""
        return base


# JSON Schema definition (for documentation and validation)
KNOWLEDGE_ENTRY_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KnowledgeEntry",
    "description": "Schema for a single structured knowledge entry in the enterprise knowledge base",
    "type": "object",
    "required": ["id", "type", "title", "content", "source_file"],
    "properties": {
        "id": {
            "type": "string",
            "format": "uuid",
            "description": "Unique identifier for this entry"
        },
        "type": {
            "type": "string",
            "enum": ["sop_step", "faq", "manual_section", "general"],
            "description": "Knowledge type: sop_step=standard operating procedure, faq=frequently asked question, manual_section=manual chapter, general=generic paragraph"
        },
        "title": {
            "type": "string",
            "description": "Title or question of the knowledge entry"
        },
        "content": {
            "type": "string",
            "description": "Body content, supports Markdown format (including tables)"
        },
        "source_file": {
            "type": "string",
            "description": "Source file name"
        },
        "source_page": {
            "type": "integer",
            "minimum": 0,
            "description": "Source page number (1-based, 0 means unknown)"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keyword list for search augmentation"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp"
        },
        "metadata": {
            "type": "object",
            "description": "Extensible additional metadata"
        }
    }
}


# Rule-based knowledge type classifier
def classify_knowledge_type(title: str, content: str) -> KnowledgeType:
    """
    Rule-based automatic knowledge type classifier.
    Determines type based on textual features of title and content.
    """
    # Clean Markdown formatting: strip ** and *, trim whitespace
    title_lower = re.sub(r'\*+', '', title).lower().strip()
    content_lower = content.lower()

    # === SOP step detection (highest priority) ===

    # 1. Title starts with "To " (e.g. "To cable your router:") with numbered steps
    if title_lower.startswith("to "):
        if re.search(r'(?:^|\n)[\s*•\-]*\d+\.', content_lower):
            return "sop_step"
        if re.search(r'(?:^|\n)\s*-\s*\d+\.', content_lower):
            return "sop_step"

    # 2. Title contains explicit step/procedure keywords
    sop_keywords = ["step", "procedure", "process", "instruction"]
    for kw in sop_keywords:
        if kw in title_lower:
            return "sop_step"

    # 3. Content has >= 3 numbered list items -> document-style SOP
    step_pattern = re.compile(r'(?:^|\n)[\s*•\-]*\d+\.')
    step_count = len(step_pattern.findall(content_lower))
    if step_count >= 3:
        return "sop_step"

    # 4. Title contains "how to" with step list in content
    if "how to" in title_lower:
        if step_pattern.search(content_lower):
            return "sop_step"
        return "faq"

    # === FAQ detection ===
    faq_indicators = ["?", "what is", "what are", "what does", "why ", "when ",
                      "q:", "q&a", "faq", "frequently asked"]
    for kw in faq_indicators:
        if kw in title_lower:
            return "faq"

    # === Manual section detection ===
    manual_keywords = ["chapter", "section", "overview", "introduction", "specification"]
    for kw in manual_keywords:
        if kw in title_lower:
            return "manual_section"

    return "general"


class KnowledgeBase:
    """Container for multiple KnowledgeEntry objects."""

    def __init__(self, source_file: str = ""):
        self.source_file = source_file
        self.entries: list[KnowledgeEntry] = []

    def add(self, entry: KnowledgeEntry):
        self.entries.append(entry)

    def to_dict_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    def save_json(self, output_path: str):
        """Export the entire knowledge base to a JSON file."""
        data = {
            "schema_version": "1.0",
            "source_file": self.source_file,
            "total_entries": len(self.entries),
            "entries": self.to_dict_list()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Knowledge base exported to: {output_path} ({len(self.entries)} entries)")

    @classmethod
    def load_json(cls, json_path: str) -> "KnowledgeBase":
        """Load knowledge base from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kb = cls(source_file=data.get("source_file", ""))
        for item in data.get("entries", []):
            entry = KnowledgeEntry(**item)
            kb.add(entry)
        print(f"[OK] Loaded {len(kb.entries)} entries from {json_path}")
        return kb


if __name__ == "__main__":
    print("Knowledge Entry Schema Test\n")

    # Create a sample SOP entry
    entry1 = KnowledgeEntry(
        type="sop_step",
        title="How to Reset the Device",
        content="1. Power off the device.\n2. Hold the reset button for 10 seconds.\n3. Wait for LED to flash.",
        source_file="sample_manual.pdf",
        source_page=12,
        keywords=["reset", "factory reset", "power off"]
    )

    # Create a sample FAQ entry
    entry2 = KnowledgeEntry(
        type="faq",
        title="What does the red LED indicate?",
        content="A solid red LED means the device has encountered a hardware fault. Please contact support.",
        source_file="sample_manual.pdf",
        source_page=8,
        keywords=["LED", "red", "fault", "error"]
    )

    print("SOP entry JSON:")
    print(entry1.to_json())
    print()
    print("FAQ entry JSON:")
    print(entry2.to_json())
    print()

    kb = KnowledgeBase(source_file="sample_manual.pdf")
    kb.add(entry1)
    kb.add(entry2)
    kb.save_json("knowledge_output_demo.json")
    print("\nSchema module test complete.")
