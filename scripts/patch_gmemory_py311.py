"""Make the G-Memory checkout importable on Python 3.11 (syntax only, no behaviour change).

Upstream G-Memory uses PEP 701 f-strings (same quote type and backslashes inside
replacement fields), which require Python 3.12. This rewrites those four
expressions into equivalent 3.11-compatible forms. Run once after cloning:

    python scripts/patch_gmemory_py311.py external/GMemory
"""

from __future__ import annotations

import sys
from pathlib import Path

BS = "\\"
FIXES = {
    "mas/memory/mas_memory/GMemory.py": [
        (
            "f'> {state.graph['action']}" + BS + "n{state.graph['observation']}" + BS + "n'",
            "f'> {state.graph[\"action\"]}" + BS + "n{state.graph[\"observation\"]}" + BS + "n'",
        ),
        (
            "f\"Origin rules: " + BS + "n{'" + BS + "n'.join(related_rules)}\"",
            "\"Origin rules: " + BS + "n\" + '" + BS + "n'.join(related_rules)",
        ),
        (
            "f\"Merged rules: " + BS + "n{'" + BS + "n'.join(merged_rules)}\"",
            "\"Merged rules: " + BS + "n\" + '" + BS + "n'.join(merged_rules)",
        ),
    ],
    "mas/memory/mas_memory/memorybank.py": [
        (
            "get_extra_field('observation')}",
            "get_extra_field(\"observation\")}",
        ),
    ],
}


def main(root: str) -> int:
    for name, pairs in FIXES.items():
        path = Path(root) / name
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if new in text and old not in text:
                continue  # already patched
            if text.count(old) != 1:
                raise SystemExit(f"{name}: expected one occurrence of {old!r}, found {text.count(old)}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8", newline="")
    print("G-Memory patched for Python 3.11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "external/GMemory"))
