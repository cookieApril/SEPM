"""Fix a syntax error in the released MARBLE checkout (commit 8d60fa1).

``Evaluator.parse_research_ratings`` places ``except json.JSONDecodeError`` inside an
``if`` block, so ``marble/evaluator/evaluator.py`` cannot be imported. This restores the
evident intended structure (try/except around ``json.loads``, else-branch when no JSON is
found) without changing what the function returns. Run once after cloning:

    python scripts/patch_marble.py external/MARBLE
"""

from __future__ import annotations

import sys
from pathlib import Path

BROKEN = '''        try:
            content = assistant_answer.strip()

            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                ratings = json.loads(json_str)
                # Ensure ratings are integers
                ratings_dict: Dict[str, int] = {k: int(v) for k, v in ratings.items()}
                return ratings_dict
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from assistant's answer.")
                return {}
        else:
            self.logger.error("No JSON found in assistant's answer.")
            return {}
'''

FIXED = '''        content = assistant_answer.strip()

        json_start = content.find('{')
        json_end = content.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                ratings = json.loads(json_str)
                # Ensure ratings are integers
                ratings_dict: Dict[str, int] = {k: int(v) for k, v in ratings.items()}
                return ratings_dict
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from assistant's answer.")
                return {}
        else:
            self.logger.error("No JSON found in assistant's answer.")
            return {}
'''


def main(root: str) -> int:
    path = Path(root) / "marble" / "evaluator" / "evaluator.py"
    text = path.read_text(encoding="utf-8")
    if FIXED in text:
        print("MARBLE already patched")
        return 0
    if text.count(BROKEN) != 1:
        raise SystemExit(f"{path}: broken block not found; upstream may have changed")
    path.write_text(text.replace(BROKEN, FIXED), encoding="utf-8", newline="")
    print("MARBLE evaluator patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "external/MARBLE"))
