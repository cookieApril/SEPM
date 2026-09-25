#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from team_memory.hotpotqa_local import run_local_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one resumable local HotpotQA preflight")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("external/GMemory/data/hotpotqa/hotpot_dev_distractor_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/hotpotqa-local/preflight-result.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("benchmark-results/hotpotqa-local/episode-checkpoint.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8113/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--case-index", type=int, default=0)
    args = parser.parse_args()
    result = run_local_preflight(
        data_path=args.data,
        output=args.output,
        checkpoint=args.checkpoint,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        case_index=args.case_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
