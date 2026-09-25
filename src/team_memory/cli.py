"""Team Memory 纯 Python 包附带的只读命令行检查工具。

``retrieve`` 用自然语言查询已发布 SOP；``candidates`` 检查候选队列。CLI
刻意不暴露写操作，正式写入应通过 ``TeamMemoryService`` Python API 完成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import TeamMemoryService


def main() -> None:
    """解析子命令，读取指定 SQLite 数据库并输出 UTF-8 JSON。"""
    parser = argparse.ArgumentParser(description="Inspect a Team Memory database")
    parser.add_argument("--db", type=Path, default=Path("team_memory.db"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int, default=5)
    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    service = TeamMemoryService(args.db)
    # argparse 已将 command 限制为两个子命令，因此这里的 else 必然是 candidates。
    if args.command == "retrieve":
        payload = [item.model_dump(mode="json") for item in service.retrieve_sops(args.query, args.limit)]
    else:
        items, total = service.store.list_candidates(None, args.limit, 0)
        payload = {"total": total, "items": [item.model_dump(mode="json") for item in items]}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
