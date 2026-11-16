"""The scheduler loop (one per state dir, on the scheduler host): probe hosts -> follow running jobs -> start what fits -> publish,
plus usage history, alerts (job ends, idle / hung / offline) and the transfer routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import logging
import os
import secrets
import shlex
import subprocess
import threading
import time

from . import db
from . import remote
from . import sched
from .config import Config
from .config import Host

KILL_AFTER_S = 30
ROUTES_EVERY_S = 6 * 3600
HUNG_UTIL = 5  # % : a GPU job below this on all its GPUs counts as not computing
log = logging.getLogger("gq")


def run_script(job: dict, host: Host, rank: int, gpus: list[int], ranks, port: int) -> str:
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if job["cvd"]:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env |= {
        "GQ_JOB_ID": str(job["id"]),
        "GQ_JOB_NAME": job["name"],
        "GQ_ATTEMPT": str(job["attempt"] + 1),
        "GQ_HOST": host.name,
        "GQ_GPUS": ",".join(map(str, gpus)),
        "GQ_NUM_GPUS": str(len(gpus)),
    }
    if job["nodes"] > 1:
        master = ranks[0][0]
        multi = {
            "NNODES": str(job["nodes"]),
            "NODE_RANK": str(rank),
            "MASTER_ADDR": master.addr or master.name,
            "MASTER_PORT": str(port),
        }
        env |= host.env | multi | {"GQ_" + k: v for k, v in multi.items()}
        env["GQ_NODES"] = ",".join(h.name for h, _ in ranks)
    env |= job["env"]
    cwd = shlex.quote(job["cwd"]) if job["cwd"] else '"$D"'
    lines = [
        "#!/bin/bash",
        f"# gq job {job['id']} rank {rank}",
        'D=$(cd "$(dirname "$0")" && pwd)',
        'echo $$ > "$D/pid"',
        *(f"export {k}={shlex.quote(str(v))}" for k, v in env.items()),
        f'echo "[gq] job {job["id"]} {shlex.quote(job["name"])} rank {rank}/{job["nodes"]} on {host.name} '
        f'GPU ${{GQ_GPUS:-none}} start $(TZ=Asia/Shanghai date "+%F %T")"',
        f'cd {cwd} || {{ echo 126 > "$D/exit_code"; exit 126; }}',
        'bash "$D/cmd.sh"',
        "rc=$?",
        'echo "[gq] exit $rc $(TZ=Asia/Shanghai date "+%F %T")"',
        'echo $rc > "$D/exit_code.tmp" && mv "$D/exit_code.tmp" "$D/exit_code"',
        "",
    ]
    return "\n".join(lines)


def short(args: str) -> str:
    """A readable name for a process: the module of -m, else the first *.py / *.sh, else the program."""
    t = args.split()
    if "-m" in t[:-1]:
        return t[t.index("-m") + 1]
    for x in t:
        if x.endswith((".py", ".sh")):
            return x.rsplit("/", 1)[-1]
    return t[0].rsplit("/", 1)[-1] if t else "?"


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.con = db.connect(cfg.db_path)
        self.snaps: dict = {}
        self.pool = ThreadPoolExecutor(max_workers=max(4, len(cfg.hosts)))
        self.last_sample_min = 0
        self.busy_since: dict[str, float] = {}  # host -> last time it was not wholly idle
        self.down_since: dict[str, float] = {}
        self.active_at: dict[int, float] = {}  # job id -> last time its GPUs computed
        self.alerted: set = set()
        self.routes_thread: threading.Thread | None = None

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def recover(self) -> None:
        for j in db.jobs(self.con, ("starting",)):
            if j["placement"]:
                db.update(self.con, j["id"], state="running")
            else:
                db.update(self.con, j["id"], state="pending", reason="守护进程重启，重新排队")

    def tick(self) -> None:
        running = db.jobs(self.con, ("running",))
        entries: dict[str, list] = {h.name: [] for h in self.cfg.hosts}
        for j in running:
            for p in j["placement"]:
                entries.setdefault(p["host"], []).append((p["dir"], p["pid"]))
        futs = {h.name: self.pool.submit(remote.probe, h, entries.get(h.name, [])) for h in self.cfg.hosts}
        snaps = {}
        for name, f in futs.items():
            try:
                snaps[name] = f.result()
            except Exception as e:  # noqa: BLE001
                snaps[name] = {"up": False, "err": repr(e)[:300], "t": time.time()}
        self.snaps = snaps
        for j in running:
            try:
                self.follow(j, snaps)
            except Exception:
                log.exception("follow job %s", j["id"])
        try:
            self.schedule(snaps)
        except Exception:
            log.exception("schedule")
        self.publish(snaps)
        for step in (self.sample, self.watch, self.maybe_routes):
            try:
                step(snaps)
            except Exception:
                log.exception(step.__name__)

    # ---- running jobs ------------------------------------------------------------------------------------------
    def follow(self, job: dict, snaps: dict) -> None:
        pl = job["placement"]
        if any(not snaps.get(p["host"], {}).get("up") for p in pl):
            return  # a host is unreachable: keep the job as it is until it answers again
        states = [snaps[p["host"]]["jobs"].get(p["dir"], ("gone",)) for p in pl]
        alive = [p for p, s in zip(pl, states) if s[0] == "run"]
        bad = [(p, s) for p, s in zip(pl, states) if s[0] == "gone" or (s[0] == "exit" and s[1] != 0)]
        now = time.time()
        if alive and job["time_limit"] and not job["cancel_req"] and now - job["started"] > job["time_limit"]:
            why = f"超过时限 {job['time_limit'] / 3600:g} h，终止中"
            db.update(self.con, job["id"], cancel_req=now, reason=why)
            job = {**job, "cancel_req": now, "reason": why}
        if alive:
            if job["cancel_req"] or bad:
                if not job["killed_at"]:
                    for p in alive:
                        remote.kill(self.cfg.host(p["host"]), p["pid"], "TERM")
                    if job["reason"].startswith("超过时限"):
                        why = job["reason"]
                    elif job["cancel_req"]:
                        why = "取消中"
                    else:
                        why = f"rank {bad[0][0]['rank']} 失败，正在终止其余 rank"
                    db.update(self.con, job["id"], killed_at=now, reason=why)
                elif now - job["killed_at"] > KILL_AFTER_S:
                    for p in alive:
                        remote.kill(self.cfg.host(p["host"]), p["pid"], "KILL")
            return
        if job["cancel_req"]:
            if job["reason"].startswith("超过时限"):
                return self.finish(job, "failed", None, job["reason"].replace("，终止中", "，已终止"))
            return self.finish(job, "cancelled", None, "已取消")
        if not bad:
            return self.finish(job, "done", 0, "")
        p, s = bad[0]
        if s[0] == "gone":
            return self.finish(
                job, "failed", -1, f"rank {p['rank']}（{p['host']}）进程消失，没有退出码（被杀或机器重启）"
            )
        return self.finish(job, "failed", s[1], f"rank {p['rank']}（{p['host']}）exit {s[1]}")

    def finish(self, job: dict, state: str, rc, reason: str) -> None:
        tail = ""
        if job["placement"]:
            p = job["placement"][0]
            try:
                tail = remote.tail(self.cfg.host(p["host"]), p["dir"], 60)
            except Exception:  # noqa: BLE001
                pass
        if state == "failed" and job["attempt"] <= job["retries"]:
            db.update(
                self.con,
                job["id"],
                state="pending",
                placement=[],
                killed_at=None,
                exit_code=rc,
                tail=tail,
                reason=f"第 {job['attempt']} 次运行失败（{reason}），重试",
            )
            log.info("job %s failed (%s), retry", job["id"], reason)
            return
        db.update(self.con, job["id"], state=state, exit_code=rc, ended=time.time(), reason=reason, tail=tail)
        log.info("job %s %s %s", job["id"], state, reason)
        self.notify(job, state, rc, reason)

    def notify(self, job, state, rc, reason) -> None:
        """A job ended (or can never start)."""
        if state == "cancelled" and reason in ("已取消", "排队时取消"):
            return  # the user did it
        took = f"，用时 {_dur(time.time() - job['started'])}" if job.get("started") else ""
        where = ",".join(p["host"] for p in job["placement"])
        icon = {"done": "✅", "failed": "❌"}.get(state, "⚪")
        kind = "传输" if job.get("kind") == "transfer" else "任务"
        text = f"{icon} gq {kind} #{job['id']} {job['name']} {STATE_ZH.get(state, state)}{took}"
        text += (f"（{where}）" if where else "") + (f"\n{reason}" if reason else "")
        self.alert("job_" + state, text, job=job, rc=rc, reason=reason)

    def alert(self, event: str, text: str, job: dict | None = None, rc=None, reason: str = "") -> None:
        log.info("alert %s: %s", event, text.replace("\n", " | "))
        if not self.cfg.notify_cmd:
            return
        if self.cfg.web_url:
            text += f"\n{self.cfg.web_url}"
        env = os.environ | {"GQ_EVENT": event, "GQ_TEXT": text, "GQ_REASON": reason}
        if job:
            env |= {
                "GQ_JOB_ID": str(job["id"]),
                "GQ_NAME": job["name"],
                "GQ_STATE": event.removeprefix("job_"),
                "GQ_EXIT": "" if rc is None else str(rc),
                "GQ_HOSTS": ",".join(p["host"] for p in job["placement"]),
            }
        try:
            subprocess.Popen(
                ["bash", "-c", self.cfg.notify_cmd],
                env=env,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=open(self.cfg.state_dir / "notify.log", "a"),  # noqa: SIM115
            )
        except OSError:
            log.exception("notify")

    # ---- usage history, idle / hung / offline watch, transfer routes -------------------------------------------
    def sample(self, snaps: dict) -> None:
        """One row per host per minute; rows older than history_days are dropped once an hour."""
        minute = int(time.time() // 60)
        if minute == self.last_sample_min:
            return
        self.last_sample_min = minute
        snap = db.kv_get(self.con, "hosts", {})
        rows = []
        for h in snap.get("hosts", []):
            if not h["up"]:
                cfgh = self.cfg.host(h["name"])
                if cfgh and cfgh.compute:
                    rows.append((minute * 60, h["name"], 0, 0, 0, 0, 1, 0.0))
                continue
            g = h["gpus"]
            if not g:
                continue
            ours = sum(1 for x in g if x["jobs"])
            idle = sum(1 for x in g if x["idle"])
            rows.append((minute * 60, h["name"], len(g), idle, ours, len(g) - idle - ours, 0,
                         sum(x["util"] for x in g) / len(g)))  # fmt: skip
        self.con.executemany("INSERT OR REPLACE INTO samples VALUES(?,?,?,?,?,?,?,?)", rows)
        if minute % 60 == 0:
            self.con.execute("DELETE FROM samples WHERE t < ?", (time.time() - self.cfg.history_days * 86400,))

    def watch(self, snaps: dict) -> None:
        now = time.time()
        snap = db.kv_get(self.con, "hosts", {})
        pending = db.jobs(self.con, ("pending",))
        for h in snap.get("hosts", []):
            name, cfgh = h["name"], self.cfg.host(h["name"])
            # offline
            if not h["up"]:
                self.down_since.setdefault(name, now)
                if self.cfg.offline_alert_min and now - self.down_since[name] > self.cfg.offline_alert_min * 60:
                    if ("down", name) not in self.alerted:
                        self.alerted.add(("down", name))
                        self.alert(
                            "host_down",
                            f"⚠️ gq：{name} 连不上已经 {_dur(now - self.down_since[name])}\n{h['err'][:200]}",
                        )
                continue
            if ("down", name) in self.alerted:
                self.alerted.discard(("down", name))
                self.alert("host_up", f"🟢 gq：{name} 恢复连接（断了 {_dur(now - self.down_since[name])}）")
            self.down_since.pop(name, None)
            # whole machine idle
            if not cfgh or cfgh.opt_in or not cfgh.compute or not h["gpus"]:
                continue
            if not all(g["idle"] for g in h["gpus"]):
                self.busy_since[name] = now
                self.alerted.discard(("idle", name))
                continue
            since = self.busy_since.setdefault(name, now)
            if (
                self.cfg.idle_alert_min
                and now - since > self.cfg.idle_alert_min * 60
                and ("idle", name) not in self.alerted
            ):
                self.alerted.add(("idle", name))
                waiting = [j for j in pending if any(x.name == name for x in self.cfg.allowed(j["hosts"]))]
                tail = f"，队列里有 {len(waiting)} 个任务在等（可能在等依赖）" if waiting else "，队列里没有等它的任务"
                self.alert("host_idle", f"💤 gq：{name} 的 {len(h['gpus'])} 张卡已经全空 {_dur(now - since)}{tail}")
        # hung jobs: running GPU jobs whose GPUs all sit near 0 %
        stats = db.kv_get(self.con, "jobstats", {})
        for j in db.jobs(self.con, ("running",)):
            st = stats.get(str(j["id"]))
            if not st or j["gpus"] == 0:
                continue
            if st["util"] >= HUNG_UTIL or now - j["started"] < 600:
                self.active_at[j["id"]] = now
                self.alerted.discard(("hung", j["id"]))
                continue
            last = self.active_at.setdefault(j["id"], now)
            if (
                self.cfg.hung_alert_min
                and now - last > self.cfg.hung_alert_min * 60
                and ("hung", j["id"]) not in self.alerted
            ):
                self.alerted.add(("hung", j["id"]))
                self.alert("job_hung", f"🐢 gq 任务 #{j['id']} {j['name']} 的 GPU 利用率 {_dur(now - last)} 都接近 0，"
                           f"可能卡住了（没有动它）", job=j)  # fmt: skip

    def maybe_routes(self, snaps: dict, force: bool = False) -> None:
        """Probe who can ssh to whom, in the background, every ROUTES_EVERY_S or when `gq net --probe` asks."""
        if self.routes_thread and self.routes_thread.is_alive():
            return
        r = db.kv_get(self.con, "routes", {})
        asked = db.kv_get(self.con, "routes_req", 0)
        if not force and r.get("t", 0) > max(time.time() - ROUTES_EVERY_S, asked):
            return

        def run():
            from . import transfer

            t0 = time.time()
            con = db.connect(self.cfg.db_path)
            db.kv_set(con, "routes", {"t": t0, "routes": transfer.probe_routes(self.cfg), "took": time.time() - t0})
            con.close()
            log.info("routes probed in %.0f s", time.time() - t0)

        self.routes_thread = threading.Thread(target=run, daemon=True, name="gq-routes")
        self.routes_thread.start()

    # ---- pending jobs ------------------------------------------------------------------------------------------
    def schedule(self, snaps: dict) -> None:
        pending = db.jobs(self.con, ("pending",))
        if not pending:
            return
        dep_ids = {d for j in pending for d in j["after"]}
        dep_state = {}
        if dep_ids:
            q = f"SELECT id, state FROM jobs WHERE id IN ({','.join('?' * len(dep_ids))})"
            dep_state = {r["id"]: r["state"] for r in self.con.execute(q, list(dep_ids))}
        ready, reasons = [], {}
        for j in pending:
            why = self.wait_reason(j, dep_state)
            if why is None:
                ready.append(j)
            elif why.startswith("!"):
                db.update(self.con, j["id"], "pending", state="failed", ended=time.time(), reason=why[1:])
                self.notify(j, "failed", None, why[1:])
            else:
                reasons[j["id"]] = why
        view = sched.build_view(self.cfg.hosts, snaps, db.jobs(self.con, ("running", "starting")))
        starts, more = sched.plan(ready, lambda j: self.cfg.allowed(j["hosts"]), view)
        reasons |= more
        for j in pending:
            if j["id"] in reasons and reasons[j["id"]] != j["reason"]:
                db.update(self.con, j["id"], "pending", reason=reasons[j["id"]])
        for job, ranks in starts:
            self.start(job, ranks)

    def wait_reason(self, j: dict, dep_state: dict) -> str | None:
        """None = ready; '!msg' = can never run; else why it waits."""
        waiting = []
        for d in j["after"]:
            s = dep_state.get(d)
            if s is None:
                return f"!依赖的任务 #{d} 不存在"
            if s in ("failed", "cancelled") and not j["after_any"]:
                return f"!依赖的任务 #{d} {s}"
            if s not in db.FINAL:
                waiting.append(f"#{d}")
        if waiting:
            return f"等依赖 {' '.join(waiting)}"
        for spec in j["wait_files"]:
            hname, _, path = spec.partition(":")
            h = self.cfg.host(hname)
            if h is None:
                return f"!--wait-file 的机器 {hname} 不在 hosts.toml"
            if remote.exists(h, path) is not True:
                return f"等文件 {spec}"
        return None

    def start(self, job: dict, ranks: list[tuple[Host, list[int]]]) -> None:
        if not db.update(self.con, job["id"], "pending", state="starting", reason="启动中"):
            return
        attempt = job["attempt"] + 1
        port = self.cfg.port_base + job["id"] % 1000
        placement = []
        try:
            for r, (h, gpus) in enumerate(ranks):
                d = f"j{job['id']}_a{attempt}_r{r}"
                pid = remote.launch(h, d, run_script(job, h, r, gpus, ranks, port), job["cmd"] + "\n")
                placement.append({"host": h.name, "rank": r, "gpus": gpus, "pid": pid, "dir": d})
        except Exception as e:  # noqa: BLE001
            for p in placement:
                remote.kill(self.cfg.host(p["host"]), p["pid"], "KILL")
            db.update(self.con, job["id"], state="pending", reason=f"启动失败，下轮再试：{e}"[:400])
            log.warning("job %s launch failed: %s", job["id"], e)
            return
        db.update(
            self.con,
            job["id"],
            state="running",
            placement=placement,
            attempt=attempt,
            started=time.time(),
            ended=None,
            killed_at=None,
            exit_code=None,
            reason="",
        )
        log.info("job %s started: %s", job["id"], [(p["host"], p["gpus"]) for p in placement])

    # ---- snapshot for the CLI and the web page -----------------------------------------------------------------
    def publish(self, snaps: dict) -> None:
        running = db.jobs(self.con, ("running", "starting"))
        names = {j["id"]: j["name"] for j in running}
        view = sched.build_view(self.cfg.hosts, snaps, running)
        ours = {(p["host"], p["pid"]) for j in running for p in j["placement"]}
        out = []
        for h in self.cfg.hosts:
            s = snaps.get(h.name, {})
            gpus = []
            for g in view.get(h.name) or []:
                apps = [a for a in s.get("apps", []) if a["uuid"] == g.uuid and (h.name, a["sid"]) not in ours]
                gpus.append(
                    {
                        "idx": g.idx,
                        "total": g.total,
                        "used": g.used,
                        "util": g.util,
                        "jobs": [{"id": jid, "name": names.get(jid, ""), "mem_mib": m} for jid, m in g.held],
                        "foreign": [
                            {
                                "pid": a["pid"],
                                "user": a["user"],
                                "mem": a["mem"],
                                "args": a["args"][:160],
                                "short": short(a["args"]),
                            }
                            for a in apps
                        ],
                        "idle": not g.held and g.used < h.idle_mib,
                    }
                )
            out.append(
                {
                    "name": h.name,
                    "tags": h.tags,
                    "group": h.group,
                    "opt_in": h.opt_in,
                    "share": h.share,
                    "up": bool(s.get("up")),
                    "err": s.get("err", ""),
                    "t": s.get("t"),
                    "gpus": gpus,
                }
            )
        db.kv_set(self.con, "hosts", {"t": time.time(), "hosts": out, "external": external(self.cfg, snaps, ours)})
        stats = {}
        for j in running:
            utils, mem = [], 0
            for p in j["placement"]:
                s = snaps.get(p["host"], {})
                byidx = {g["idx"]: g for g in s.get("gpus", [])}
                utils += [byidx[i]["util"] for i in p["gpus"] if i in byidx]
                mem += sum(a["mem"] for a in s.get("apps", []) if a["sid"] == p["pid"])
            stats[str(j["id"])] = {"util": max(utils) if utils else 0, "mem_mib": mem}
        db.kv_set(self.con, "jobstats", stats)


STATE_ZH = {"done": "完成", "failed": "失败", "cancelled": "取消"}


def _dur(s: float) -> str:
    s = int(max(0, s))
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min" if m else f"{s} s"


SHELLS = {"bash", "-bash", "sh", "-sh", "zsh", "-zsh", "tmux", "screen", "sshd", "systemd", "init", "su", "sudo"}


def external(cfg: Config, snaps: dict, ours: set) -> list[dict]:
    """GPU work outside the queue, one entry per (host, session): the processes of one training run share a session
    (torchrun workers, a tmux pane, a nohup'd launch script), so they show up as one task."""
    out = []
    for h in cfg.hosts:
        s = snaps.get(h.name, {})
        if not s.get("up"):
            continue
        idx = {g["uuid"]: g["idx"] for g in s.get("gpus", [])}
        groups: dict[int, list] = {}
        for a in s.get("apps", []):
            if (h.name, a["sid"]) in ours:
                continue
            groups.setdefault(a["sid"] or -a["pid"], []).append(a)
        for sid, apps in groups.items():
            if sum(a["mem"] for a in apps) < 1024:  # desktops, sunshine, small contexts: not a task
                continue
            main = max(apps, key=lambda a: a["mem"])
            lead = apps[0].get("lead", "")
            lead_prog = lead.split()[0].rsplit("/", 1)[-1] if lead else ""
            use_lead = lead and lead_prog not in SHELLS or (lead_prog in ("bash", "sh") and len(lead.split()) > 1)
            gpus = sorted({idx[a["uuid"]] for a in apps if a["uuid"] in idx})
            per_gpu: dict[int, int] = {}
            for a in apps:
                if a["uuid"] in idx:
                    per_gpu[idx[a["uuid"]]] = per_gpu.get(idx[a["uuid"]], 0) + a["mem"]
            out.append(
                {
                    "host": h.name,
                    "sid": sid,
                    "user": main["user"],
                    "name": short(main["args"]),
                    "gpus": gpus,
                    "mem_mib": sum(a["mem"] for a in apps),
                    "mem_per_gpu": max(per_gpu.values()) if per_gpu else 0,
                    "procs": len({a["pid"] for a in apps}),
                    "elapsed": max(a.get("etime", 0) for a in apps),
                    "cmd": lead if use_lead else main["args"],
                }
            )
    return sorted(out, key=lambda e: (-len(e["gpus"]), e["host"]))


def wake_path(cfg: Config):
    return cfg.state_dir / "wake"


def poke(cfg: Config) -> None:
    """Make the daemon run its next tick now (submit, cancel, ...)."""
    try:
        wake_path(cfg).touch()
    except OSError:
        pass


def main(cfg: Config, web: bool = True, once: bool = False) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(cfg.state_dir / "daemon.log"), logging.StreamHandler()],
    )
    lock = open(cfg.state_dir / "daemon.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another gq daemon holds {cfg.state_dir}/daemon.lock") from None
    lock.write(str(os.getpid()))
    lock.flush()
    tok = cfg.state_dir / "token"
    if not tok.exists():  # the web page's submit form asks for it: gq token
        tok.write_text(secrets.token_urlsafe(9) + "\n")
        tok.chmod(0o600)
    d = Daemon(cfg)
    d.recover()
    if web and not once:
        from . import web as webmod

        webmod.serve_in_thread(cfg)
        log.info("web page on http://%s:%s/", cfg.web_host, cfg.web_port)
    log.info("gq daemon up, %d hosts, poll %.0f s", len(cfg.hosts), cfg.poll_s)
    wp = wake_path(cfg)
    while True:
        t0 = time.time()
        try:
            d.tick()
        except Exception:
            log.exception("tick")
        if once:
            return
        while time.time() - t0 < cfg.poll_s:
            time.sleep(1)
            try:
                if wp.stat().st_mtime > t0:
                    break
            except FileNotFoundError:
                pass
