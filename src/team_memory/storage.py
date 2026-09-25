"""SQLite 持久化、队列租约、版本 CAS 与审计日志。

领域对象以完整 JSON 保存，常用过滤字段另建关系列和索引；这样既保留 Pydantic schema
的演进弹性，又避免队列/分页查询扫描 JSON。所有写操作使用 ``BEGIN IMMEDIATE``，
每次操作使用短连接，适合同一主机上的多个无状态 validator worker。

``sop_versions`` 保存不可变历史，``sop_heads`` 只指向当前版本并维护 active 标记。
更新通过 expected_version 比较实现 optimistic concurrency，绝不覆盖旧版本。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from .models import (
    AgentProfile,
    BlackboardEntry,
    CandidateStatus,
    ReproductionTrial,
    SOPCandidate,
    SOPVersion,
    Workspace,
)


# SQLite 结构分为：私有记忆、共享黑板、候选/试验、SOP 版本和审计五组表。
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    workspace_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS blackboard (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT UNIQUE NOT NULL,
    workspace_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    state_key TEXT,
    entry_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
);
CREATE INDEX IF NOT EXISTS idx_blackboard_workspace_seq ON blackboard(workspace_id, seq);
CREATE INDEX IF NOT EXISTS idx_blackboard_state ON blackboard(workspace_id, state_key);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    target_sop_id TEXT,
    status TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    leased_by TEXT,
    lease_until TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status, created_at);
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    task_family TEXT NOT NULL,
    trial_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_trials_candidate ON trials(candidate_id);
CREATE TABLE IF NOT EXISTS sop_versions (
    sop_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    sop_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(sop_id, version)
);
CREATE TABLE IF NOT EXISTS sop_heads (
    sop_id TEXT PRIMARY KEY,
    current_version INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(sop_id, current_version) REFERENCES sop_versions(sop_id, version)
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class ConcurrencyError(RuntimeError):
    """expected_version 与数据库 head 不一致，调用方应重新读取后再决策。"""
    pass


class NotFoundError(LookupError):
    """按业务 id 查询不到领域对象。"""
    pass


class SQLiteStore:
    """短连接事务存储；不缓存领域对象，跨 worker 以数据库状态为准。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建启用外键、busy timeout 和 Row 映射的新连接。"""
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        """进程内串行执行幂等 DDL；WAL 允许读者和单个写者并行。"""
        with self._schema_lock:
            connection = self._connect()
            try:
                connection.executescript(SCHEMA)
            finally:
                connection.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """只读/自动提交连接上下文，退出时确保关闭。"""
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """立即取得写锁；成功提交，任意异常回滚并原样抛出。"""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _json(model: Any) -> str:
        return model.model_dump_json()

    def upsert_agent(self, profile: AgentProfile) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO agents(agent_id, profile_json) VALUES (?, ?) "
                "ON CONFLICT(agent_id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=CURRENT_TIMESTAMP",
                (profile.agent_id, self._json(profile)),
            )
            self._audit(connection, "agent_upserted", profile.agent_id, profile.model_dump(mode="json"))

    def get_agent(self, agent_id: str) -> AgentProfile:
        with self.connection() as connection:
            row = connection.execute("SELECT profile_json FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"agent {agent_id!r} not found")
        return AgentProfile.model_validate_json(row["profile_json"])

    def put_workspace(self, workspace: Workspace, *, overwrite: bool = True) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO workspaces(workspace_id, workspace_json) VALUES (?, ?) "
                + ("ON CONFLICT(workspace_id) DO UPDATE SET workspace_json=excluded.workspace_json, updated_at=CURRENT_TIMESTAMP"
                   if overwrite else "ON CONFLICT(workspace_id) DO NOTHING"),
                (workspace.workspace_id, self._json(workspace)),
            )
            self._audit(connection, "workspace_saved", workspace.workspace_id, workspace.model_dump(mode="json"))

    def get_workspace(self, workspace_id: str) -> Workspace:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT workspace_json FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"workspace {workspace_id!r} not found")
        return Workspace.model_validate_json(row["workspace_json"])

    def append_entry(self, entry: BlackboardEntry, max_entries: int = 10_000) -> None:
        """追加黑板事件，并用硬容量限制防止单工作区无限增长。"""
        with self.transaction() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS n FROM blackboard WHERE workspace_id=?", (entry.workspace_id,)
            ).fetchone()["n"]
            if count >= max_entries:
                raise RuntimeError("blackboard capacity reached; archive the workspace before appending")
            connection.execute(
                "INSERT INTO blackboard(entry_id, workspace_id, agent_id, kind, state_key, entry_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.entry_id,
                    entry.workspace_id,
                    entry.agent_id,
                    entry.kind.value,
                    entry.state_key,
                    self._json(entry),
                    entry.created_at.isoformat(),
                ),
            )

    def list_entries(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
        agent_id: str | None = None,
    ) -> tuple[list[BlackboardEntry], int]:
        """分页读取黑板；消融时可用 ``agent_id`` 构造只看本地轨迹的视图。"""
        agent_clause = " AND agent_id=?" if agent_id is not None else ""
        params: list[Any] = [workspace_id]
        if agent_id is not None:
            params.append(agent_id)
        with self.connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS n FROM blackboard WHERE workspace_id=?{agent_clause}",
                params,
            ).fetchone()["n"]
            rows = connection.execute(
                f"SELECT entry_json FROM blackboard WHERE workspace_id=?{agent_clause} "
                "ORDER BY seq DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [BlackboardEntry.model_validate_json(row["entry_json"]) for row in rows], total

    def list_state_entries(
        self,
        workspace_id: str,
        state_key: str | None = None,
        agent_id: str | None = None,
    ) -> list[BlackboardEntry]:
        sql = "SELECT entry_json FROM blackboard WHERE workspace_id=? AND state_key IS NOT NULL"
        params: list[Any] = [workspace_id]
        if state_key is not None:
            sql += " AND state_key=?"
            params.append(state_key)
        if agent_id is not None:
            sql += " AND agent_id=?"
            params.append(agent_id)
        sql += " ORDER BY seq DESC"
        with self.connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [BlackboardEntry.model_validate_json(row["entry_json"]) for row in rows]

    def put_candidate(self, candidate: SOPCandidate) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO candidates(candidate_id, target_sop_id, status, candidate_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    candidate.candidate_id,
                    candidate.target_sop_id,
                    candidate.status.value,
                    self._json(candidate),
                    candidate.created_at.isoformat(),
                ),
            )
            self._audit(connection, "candidate_created", candidate.candidate_id, candidate.model_dump(mode="json"))

    def get_candidate(self, candidate_id: str) -> SOPCandidate:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT candidate_json FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"candidate {candidate_id!r} not found")
        return SOPCandidate.model_validate_json(row["candidate_json"])

    def update_candidate_status(self, candidate_id: str, status: CandidateStatus) -> SOPCandidate:
        candidate = self.get_candidate(candidate_id).model_copy(update={"status": status})
        with self.transaction() as connection:
            connection.execute(
                "UPDATE candidates SET status=?, candidate_json=?, updated_at=CURRENT_TIMESTAMP WHERE candidate_id=?",
                (status.value, self._json(candidate), candidate_id),
            )
            self._audit(connection, "candidate_status", candidate_id, {"status": status.value})
        return candidate

    def list_candidates(self, status: CandidateStatus | None, limit: int, offset: int) -> tuple[list[SOPCandidate], int]:
        where = " WHERE status=?" if status else ""
        params: list[Any] = [status.value] if status else []
        with self.connection() as connection:
            total = connection.execute(f"SELECT COUNT(*) AS n FROM candidates{where}", params).fetchone()["n"]
            rows = connection.execute(
                f"SELECT candidate_json FROM candidates{where} ORDER BY created_at LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [SOPCandidate.model_validate_json(row["candidate_json"]) for row in rows], total

    def claim_candidate(self, worker_id: str, lease_seconds: int = 60) -> SOPCandidate | None:
        """原子租用最早 pending 候选；过期租约可由其他 worker 自动接管。"""
        from .models import utc_now

        now = utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT candidate_id, candidate_json FROM candidates "
                "WHERE status=? AND (lease_until IS NULL OR lease_until < ?) "
                "ORDER BY created_at LIMIT 1",
                (CandidateStatus.PENDING.value, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            # UPDATE 再次带过期条件，防止 SELECT 与 UPDATE 之间的竞争造成双领。
            cursor = connection.execute(
                "UPDATE candidates SET leased_by=?, lease_until=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE candidate_id=? AND (lease_until IS NULL OR lease_until < ?)",
                (worker_id, lease_until.isoformat(), row["candidate_id"], now.isoformat()),
            )
            if cursor.rowcount != 1:
                return None
            self._audit(
                connection,
                "candidate_claimed",
                row["candidate_id"],
                {"worker_id": worker_id, "lease_until": lease_until.isoformat()},
            )
            return SOPCandidate.model_validate_json(row["candidate_json"])

    def release_candidate(self, candidate_id: str, worker_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE candidates SET leased_by=NULL, lease_until=NULL, updated_at=CURRENT_TIMESTAMP "
                "WHERE candidate_id=? AND leased_by=?",
                (candidate_id, worker_id),
            )
            if cursor.rowcount:
                self._audit(connection, "candidate_released", candidate_id, {"worker_id": worker_id})
            return cursor.rowcount == 1

    def add_trial(self, trial: ReproductionTrial) -> None:
        """持久化验证 trial；trial_id 主键也承担重复提交保护。"""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO trials(trial_id, candidate_id, workspace_id, task_family, trial_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trial.trial_id,
                    trial.candidate_id,
                    trial.workspace_id,
                    trial.task_family,
                    self._json(trial),
                    trial.created_at.isoformat(),
                ),
            )
            self._audit(connection, "trial_recorded", trial.trial_id, trial.model_dump(mode="json"))

    def list_trials(self, candidate_id: str) -> list[ReproductionTrial]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT trial_json FROM trials WHERE candidate_id=? ORDER BY created_at", (candidate_id,)
            ).fetchall()
        return [ReproductionTrial.model_validate_json(row["trial_json"]) for row in rows]

    def get_sop(self, sop_id: str, version: int | None = None) -> SOPVersion:
        """默认读取 head 指向版本；指定 version 时可审计任意历史版本。"""
        with self.connection() as connection:
            if version is None:
                row = connection.execute(
                    "SELECT v.sop_json, h.active FROM sop_heads h JOIN sop_versions v ON v.sop_id=h.sop_id "
                    "AND v.version=h.current_version WHERE h.sop_id=?",
                    (sop_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT sop_json FROM sop_versions WHERE sop_id=? AND version=?", (sop_id, version)
                ).fetchone()
        if row is None:
            raise NotFoundError(f"SOP {sop_id!r} version {version!r} not found")
        sop = SOPVersion.model_validate_json(row["sop_json"])
        return sop.model_copy(update={"active": bool(row["active"])}) if version is None else sop

    def list_active_sops(self, limit: int = 100, offset: int = 0) -> tuple[list[SOPVersion], int]:
        with self.connection() as connection:
            total = connection.execute("SELECT COUNT(*) AS n FROM sop_heads WHERE active=1").fetchone()["n"]
            rows = connection.execute(
                "SELECT v.sop_json FROM sop_heads h JOIN sop_versions v ON v.sop_id=h.sop_id "
                "AND v.version=h.current_version WHERE h.active=1 ORDER BY h.sop_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [SOPVersion.model_validate_json(row["sop_json"]) for row in rows], total

    def commit_sop(self, sop: SOPVersion, expected_version: int | None, candidate_id: str | None = None) -> None:
        """以 CAS 原子追加版本并推进 head；create 的 expected_version 必须为 None。"""
        with self.transaction() as connection:
            if candidate_id and self._candidate_promoted(connection, candidate_id):
                return
            head = connection.execute(
                "SELECT current_version FROM sop_heads WHERE sop_id=?", (sop.sop_id,)
            ).fetchone()
            actual = head["current_version"] if head else None
            # 比较发生在持有写锁的同一事务内，避免检查后被另一个 writer 抢先提交。
            if actual != expected_version:
                raise ConcurrencyError(
                    f"SOP {sop.sop_id!r} changed concurrently: expected {expected_version}, actual {actual}"
                )
            connection.execute(
                "INSERT INTO sop_versions(sop_id, version, sop_json, created_at) VALUES (?, ?, ?, ?)",
                (sop.sop_id, sop.version, self._json(sop), sop.created_at.isoformat()),
            )
            connection.execute(
                "INSERT INTO sop_heads(sop_id, current_version, active) VALUES (?, ?, ?) "
                "ON CONFLICT(sop_id) DO UPDATE SET current_version=excluded.current_version, active=excluded.active",
                (sop.sop_id, sop.version, int(sop.active)),
            )
            self._audit(connection, "sop_committed", sop.sop_id, {"version": sop.version})
            if candidate_id:
                self._mark_promoted(connection, candidate_id)

    def deactivate_sop(self, sop_id: str, expected_version: int, candidate_id: str | None = None) -> None:
        """逻辑停用当前 head，不删除任何版本历史。"""
        with self.transaction() as connection:
            if candidate_id and self._candidate_promoted(connection, candidate_id):
                return
            cursor = connection.execute(
                "UPDATE sop_heads SET active=0 WHERE sop_id=? AND current_version=?",
                (sop_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise ConcurrencyError("SOP delete lost an optimistic-concurrency race")
            self._audit(connection, "sop_deactivated", sop_id, {"version": expected_version})
            if candidate_id:
                self._mark_promoted(connection, candidate_id)

    @staticmethod
    def _candidate_promoted(connection: sqlite3.Connection, candidate_id: str) -> bool:
        row = connection.execute("SELECT status FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"candidate {candidate_id!r} not found")
        return row["status"] == CandidateStatus.PROMOTED.value

    def _mark_promoted(self, connection: sqlite3.Connection, candidate_id: str) -> None:
        row = connection.execute("SELECT candidate_json FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        candidate = SOPCandidate.model_validate_json(row["candidate_json"]).model_copy(
            update={"status": CandidateStatus.PROMOTED}
        )
        connection.execute(
            "UPDATE candidates SET status=?, candidate_json=? WHERE candidate_id=?",
            (CandidateStatus.PROMOTED.value, self._json(candidate), candidate_id),
        )
        self._audit(connection, "candidate_status", candidate_id, {"status": "promoted"})

    def record_sop_outcome(self, sop_id: str, success: bool) -> SOPVersion:
        """Increment current-head counters under the same write lock as the read."""
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT v.sop_json, h.active FROM sop_heads h JOIN sop_versions v "
                "ON h.sop_id=v.sop_id AND h.current_version=v.version WHERE h.sop_id=?",
                (sop_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"SOP {sop_id!r} not found")
            sop = SOPVersion.model_validate_json(row["sop_json"])
            updated = sop.model_copy(update={
                "retrieval_count": sop.retrieval_count + 1,
                "success_count": sop.success_count + int(success),
                "failure_count": sop.failure_count + int(not success),
            })
            connection.execute("UPDATE sop_versions SET sop_json=? WHERE sop_id=? AND version=?",
                               (self._json(updated), sop_id, sop.version))
            return updated.model_copy(update={"active": bool(row["active"])})

    def update_sop_metrics(self, sop: SOPVersion, expected_version: int) -> None:
        """只原位更新运行指标；调用方必须保持 procedure/metadata 内容不变。"""
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sop_versions SET sop_json=? WHERE sop_id=? AND version=?",
                (self._json(sop), sop.sop_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise ConcurrencyError("SOP metrics update failed")

    @staticmethod
    def _audit(connection: sqlite3.Connection, event: str, entity_id: str, payload: dict[str, Any]) -> None:
        """在业务写入的同一事务追加审计事件，保证二者共同提交或回滚。"""
        connection.execute(
            "INSERT INTO audit_log(event_type, entity_id, payload_json) VALUES (?, ?, ?)",
            (event, entity_id, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)),
        )
