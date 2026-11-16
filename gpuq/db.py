"""The queue: one SQLite file shared by the daemon, the CLI and the web page (WAL, so readers never block)."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

ACTIVE = ("pending", "held", "starting", "running")
FINAL = ("done", "failed", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL DEFAULT '',
  cmd TEXT NOT NULL,
  cwd TEXT NOT NULL DEFAULT '',          -- '' = the job's own directory
  gpus INTEGER NOT NULL DEFAULT 1,       -- per node
  nodes INTEGER NOT NULL DEFAULT 1,
  mem_gb REAL,                           -- NULL = whole GPUs; else may share a GPU with this much free
  hosts TEXT NOT NULL DEFAULT '',        -- comma-separated host / tag patterns
  priority INTEGER NOT NULL DEFAULT 0,
  after TEXT NOT NULL DEFAULT '',        -- comma-separated job ids
  after_any INTEGER NOT NULL DEFAULT 0,  -- 1: start once they ended, whatever the outcome
  wait_files TEXT NOT NULL DEFAULT '',   -- comma-separated host:path
  env TEXT NOT NULL DEFAULT '{}',
  cvd INTEGER NOT NULL DEFAULT 1,        -- export CUDA_VISIBLE_DEVICES
  retries INTEGER NOT NULL DEFAULT 0,
  attempt INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'pending',
  reason TEXT NOT NULL DEFAULT '',
  placement TEXT NOT NULL DEFAULT '[]',  -- [{host, rank, gpus, pid, dir}]
  cancel_req REAL,
  killed_at REAL,
  submitted REAL,
  started REAL,
  ended REAL,
  exit_code INTEGER,
  tail TEXT NOT NULL DEFAULT '',
  submit_info TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT NOT NULL);
-- one row per host per minute: how its GPUs were used (for the 24 h view and idle statistics)
CREATE TABLE IF NOT EXISTS samples(
  t INTEGER NOT NULL, host TEXT NOT NULL, gpus INTEGER NOT NULL,
  idle INTEGER NOT NULL, ours INTEGER NOT NULL, foreign_ INTEGER NOT NULL, down INTEGER NOT NULL,
  util REAL NOT NULL, PRIMARY KEY(t, host));
"""

# columns added after the first release: (name, declaration)
MIGRATIONS = [
    ("kind", "TEXT NOT NULL DEFAULT 'run'"),  # run | transfer
    ("time_limit", "REAL"),  # seconds; the job is stopped (failed) after this long
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15, isolation_level=None, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=15000")
    con.executescript(SCHEMA)
    have = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
    for name, decl in MIGRATIONS:
        if name not in have:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
    return con


def job_dict(row: sqlite3.Row) -> dict:
    j = dict(row)
    j["placement"] = json.loads(j["placement"])
    j["env"] = json.loads(j["env"])
    j["hosts"] = [p for p in j["hosts"].split(",") if p]
    j["after"] = [int(x) for x in j["after"].split(",") if x]
    j["wait_files"] = [p for p in j["wait_files"].split(",") if p]
    return j


def get(con, job_id: int) -> dict | None:
    row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return job_dict(row) if row else None


def jobs(con, states=None, limit: int | None = None, since: float | None = None) -> list[dict]:
    q, args = "SELECT * FROM jobs WHERE 1", []
    if states:
        q += f" AND state IN ({','.join('?' * len(states))})"
        args += list(states)
    if since is not None:
        q += " AND (ended IS NULL OR ended >= ?)"
        args.append(since)
    q += " ORDER BY id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    return [job_dict(r) for r in con.execute(q, args)]


def insert(con, **f) -> int:
    for k in ("placement", "env"):
        if k in f and not isinstance(f[k], str):
            f[k] = json.dumps(f[k])
    for k in ("hosts", "after", "wait_files"):
        if k in f and not isinstance(f[k], str):
            f[k] = ",".join(str(x) for x in f[k])
    f.setdefault("submitted", time.time())
    cols = ",".join(f)
    cur = con.execute(f"INSERT INTO jobs({cols}) VALUES({','.join('?' * len(f))})", list(f.values()))
    return cur.lastrowid


def update(con, job_id: int, where_state: str | tuple | None = None, **f) -> bool:
    """Set fields; with where_state, only if the job is (still) in that state. True if a row changed."""
    for k in ("placement", "env"):
        if k in f and not isinstance(f[k], str):
            f[k] = json.dumps(f[k])
    q = f"UPDATE jobs SET {','.join(k + '=?' for k in f)} WHERE id=?"
    args = [*f.values(), job_id]
    if where_state:
        ws = (where_state,) if isinstance(where_state, str) else tuple(where_state)
        q += f" AND state IN ({','.join('?' * len(ws))})"
        args += ws
    return con.execute(q, args).rowcount > 0


def kv_set(con, k: str, v) -> None:
    con.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))


def kv_get(con, k: str, default=None):
    row = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row["v"]) if row else default
