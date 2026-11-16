"""The web page: GPUs of every host, the queue (cancel / hold / release / priority / rerun / logs), work outside the
queue, 24 h usage, who reaches whom, and a submit form for jobs and transfers. Reading is open; submitting needs the
token of `gq token` (the page is reachable from the LAN and the tailnet)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs
from urllib.parse import urlparse

from . import db
from . import ops
from . import remote
from .config import Config

PAGE = Path(__file__).with_name("index.html")


def make_handler(cfg: Config):
    from .daemon import poke

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            con = db.connect(cfg.db_path)
            try:
                if u.path == "/":
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    active = db.jobs(con, db.ACTIVE)
                    recent = db.jobs(con, db.FINAL, limit=int(q.get("recent", 60)))
                    return self._json(
                        {
                            "now": time.time(),
                            "snap": db.kv_get(con, "hosts", {}),
                            "jobs": active + recent,
                            "stats": db.kv_get(con, "jobstats", {}),
                            "hosts": [h.name for h in cfg.hosts],
                        }
                    )
                if u.path == "/api/history":
                    return self._json(history(con, float(q.get("hours", 24))))
                if u.path == "/api/routes":
                    r = db.kv_get(con, "routes", {})
                    hosts = [h.name for h in cfg.hosts]
                    return self._json({"t": r.get("t"), "routes": r.get("routes", {}), "hosts": hosts})
                if u.path == "/api/log":
                    j = db.get(con, int(q["id"]))
                    if not j or not j["placement"]:
                        return self._send(
                            200, ((j or {}).get("tail") or "(还没有日志)").encode(), "text/plain; charset=utf-8"
                        )
                    rank = int(q.get("rank", 0))
                    p = j["placement"][min(rank, len(j["placement"]) - 1)]
                    text = remote.tail(cfg.host(p["host"]), p["dir"], int(q.get("n", 300)))
                    return self._send(200, text.encode(), "text/plain; charset=utf-8")
                self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                self._json({"error": repr(e)}, 500)
            finally:
                con.close()

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/api/submit":
                return self.submit(body)
            if path != "/api/op":
                return self._send(404, b"not found", "text/plain")
            con = db.connect(cfg.db_path)
            try:
                jid, op = int(body["id"]), body["op"]
                if op == "cancel":
                    msg = ops.cancel(con, jid)
                elif op == "hold":
                    msg = ops.hold(con, jid)
                elif op == "release":
                    msg = ops.release(con, jid)
                elif op == "prio":
                    msg = ops.prio(con, jid, int(body["value"]))
                elif op == "resubmit":
                    msg = ops.resubmit(con, jid)
                else:
                    return self._json({"error": f"unknown op {op}"}, 400)
                poke(cfg)
                self._json({"msg": msg})
            except Exception as e:  # noqa: BLE001
                self._json({"error": repr(e)}, 500)
            finally:
                con.close()

        def submit(self, body: dict) -> None:
            want = (cfg.state_dir / "token").read_text().strip()
            if not hmac.compare_digest(str(body.get("token", "")), want):
                return self._json({"error": "口令不对（在调度机上 gq token 查看）"}, 403)
            con = db.connect(cfg.db_path)
            who = f"web {self.client_address[0]}"
            try:
                if body.get("kind") == "transfer":
                    jid, how = ops.submit_transfer(cfg, con, body.get("src", ""), body.get("dst", ""),
                                                   {**body, "submit_info": who})  # fmt: skip
                    msg = f"#{jid} 已提交（{ {'same': '本机复制', 'push': '源机推送', 'pull': '目标机拉取', 'relay': '经调度机中转'}[how] }）"
                else:
                    spec = {**body, "submit_info": who}
                    spec["hosts"] = [x for x in str(body.get("hosts", "")).split(",") if x.strip()]
                    spec["after"] = [x for x in str(body.get("after", "")).replace(" ", ",").split(",") if x.strip()]
                    jid = ops.submit(cfg, con, spec)
                    msg = f"#{jid} 已提交"
                poke(cfg)
                self._json({"msg": msg, "id": jid})
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                self._json({"error": repr(e)}, 500)
            finally:
                con.close()

    return H


def history(con, hours: float) -> dict:
    """Per host: buckets of [t, idle, ours, foreign, down (fractions of its GPUs), mean util %] and GPU-hours."""
    since = time.time() - hours * 3600
    bucket = max(60, int(hours * 3600 / 144) // 60 * 60)  # ~144 columns
    rows = con.execute(
        "SELECT host, (t / ?) * ? AS b, AVG(CAST(idle AS REAL) / MAX(gpus, 1)), AVG(CAST(ours AS REAL) / MAX(gpus, 1)),"
        " AVG(CAST(foreign_ AS REAL) / MAX(gpus, 1)), AVG(down), AVG(util) FROM samples WHERE t >= ?"
        " GROUP BY host, b ORDER BY b",
        (bucket, bucket, since),
    ).fetchall()
    out: dict = {}
    for r in rows:
        out.setdefault(r[0], {"buckets": []})["buckets"].append([r[1], *(round(x, 3) for x in r[2:])])
    for r in con.execute(
        "SELECT host, SUM(idle), SUM(ours), SUM(foreign_), SUM(down * gpus), MAX(gpus), COUNT(*) FROM samples"
        " WHERE t >= ? GROUP BY host",
        (since,),
    ):
        out.setdefault(r[0], {"buckets": []}).update(
            {"idle_h": r[1] / 60, "ours_h": r[2] / 60, "foreign_h": r[3] / 60, "gpus": r[5], "minutes": r[6]}
        )
    return {"since": since, "bucket": bucket, "hosts": out}


def serve(cfg: Config) -> None:
    ThreadingHTTPServer((cfg.web_host, cfg.web_port), make_handler(cfg)).serve_forever()


def serve_in_thread(cfg: Config) -> threading.Thread:
    t = threading.Thread(target=serve, args=(cfg,), daemon=True, name="gq-web")
    t.start()
    return t
