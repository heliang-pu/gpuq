"""gq: a GPU queue over the fleet. `gq -h` and `gq <command> -h` for help."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import unicodedata

from . import config
from . import db
from . import ops
from . import remote
from .daemon import poke

EPILOG = """examples:
  gq submit -n sim_eval -g 1 -H ws1 -- bash eval.sh task_a 0
  gq submit -n finetune -g 8 -H a100 -C /data/myproject -- bash launch.sh
  gq submit -n two_node -g 8 -N 2 -H 'node3,node4' -- bash launch.sh             # NNODES/NODE_RANK/MASTER_* set
  gq submit -n export -g 0 -a 12 -- python export.py                              # CPU job after #12 succeeded
  gq submit -n feat -g 1 -m 12 -H ws0 -- python extract.py                         # shares a GPU (12 GB)
  gq submit -w node4:/data/dataset/x.ready -g 8 -H node4 -- bash train.sh          # waits for a file
  gq submit -t 8h -g 8 -H a100 -- bash train.sh                                    # stopped after 8 h
  gq cp ws0:/home/me/data/ds node3:/data/dataset/                                  # transfer job (see gq cp -h)
  gq ls | gq show 12 | gq log 12 -f | gq cancel 12 | gq hosts | gq net | gq daemon
"""


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _w(s))


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(_w(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return "\n".join("  ".join([*(_pad(c, w) for c, w in zip(r[:-1], widths)), r[-1]]).rstrip() for r in rows)


def dur(s: float) -> str:
    s = int(max(0, s))
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s" if m else f"{s}s"


os.environ.setdefault("TZ", "Asia/Shanghai")  # every time shown is Beijing time
time.tzset()


def ts(t) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else ""


STATE = {"pending": "排队", "held": "暂缓", "starting": "启动", "running": "运行", "done": "完成", "failed": "失败",
         "cancelled": "取消"}  # fmt: skip


def cmd_submit(a, cfg, con):
    if not a.cmd:
        sys.exit("gq submit: the command goes after --")
    env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep:
            sys.exit(f"gq submit: --env {kv}: expected KEY=VALUE")
        env[k] = v
    hosts = [p for x in a.host for p in x.split(",") if p]
    cwd = a.cwd
    if cwd is None and hosts and all(not h.ssh for h in cfg.allowed(hosts)):
        cwd = os.getcwd()  # a job pinned to the scheduler host: run where it was submitted
    spec = {
        "cmd": a.cmd[0] if len(a.cmd) == 1 else shlex.join(a.cmd),
        "name": a.name or a.cmd[-1].split("/")[-1][:40],
        "gpus": a.gpus,
        "nodes": a.nodes,
        "mem_gb": a.mem,
        "hosts": hosts,
        "priority": a.priority,
        "after": [x for s in a.after for x in s.split(",") if x],
        "after_any": a.after_any,
        "wait_files": a.wait_file,
        "cwd": cwd,
        "env": env,
        "cvd": int(not a.no_cvd),
        "retries": a.retry,
        "hold": a.hold,
        "time_limit": a.time,
        "submit_info": _who(),
    }
    try:
        jid = ops.submit(cfg, con, spec)
    except ValueError as e:
        sys.exit(f"gq submit: {e}")
    poke(cfg)
    print(jid if a.quiet else f"#{jid} 已提交（{'暂缓' if a.hold else '排队'}）：gq show {jid}")


def _who() -> str:
    return f"{os.environ.get('USER', '?')}@{socket.gethostname()} {os.getcwd()}"


def cmd_cp(a, cfg, con):
    spec = {
        "name": a.name,
        "priority": a.priority,
        "after": [x for s in a.after for x in s.split(",") if x],
        "after_any": a.after_any,
        "retries": a.retry,
        "hold": a.hold,
        "delete": a.delete,
        "bwlimit": a.bwlimit,
        "time_limit": a.time,
        "submit_info": _who(),
    }
    routes = db.kv_get(con, "routes")
    if not routes:
        print("（还没有探测过机器间的连通，先按中转处理；gq net --probe 可以探测）")
    try:
        jid, how = ops.submit_transfer(cfg, con, a.src, a.dst, spec)
    except ValueError as e:
        sys.exit(f"gq cp: {e}")
    poke(cfg)
    j = db.get(con, jid)
    explain = {"same": "同一台机器内复制", "push": f"在 {j['hosts'][0]} 上 rsync 推过去", "pull": f"在 {j['hosts'][0]} 上 rsync 拉过来",
               "relay": f"两台互相连不上，经调度机 {j['hosts'][0]} 用 tar 中转"}[how]  # fmt: skip
    print(jid if a.quiet else f"#{jid} 已提交：{explain}。gq log {jid} -f 看进度；训练可以用 -a {jid} 等它传完")


def cmd_net(a, cfg, con):
    from . import transfer

    if a.authorize:
        src, dst = (cfg.host(x) for x in a.authorize)
        if not src or not dst:
            sys.exit(f"gq net --authorize: 机器是 {', '.join(h.name for h in cfg.hosts)} 之一")
        print(transfer.authorize(cfg, src, dst))
        print("重新探测连通…")
        a.probe = True
    if a.probe:
        t0 = time.time()
        db.kv_set(con, "routes", {"t": t0, "routes": transfer.probe_routes(cfg), "took": time.time() - t0})
    r = db.kv_get(con, "routes")
    if not r:
        sys.exit("还没有探测过：gq net --probe")
    names = [h.name for h in cfg.hosts]
    rows = [["从 \\ 到", *names]]
    for s in names:
        rows.append([s, *("·" if s == d else ("✓" if (r["routes"].get(s) or {}).get(d) else "✗") for d in names)])
    print(f"机器间能否直接 ssh（✓ 直连 rsync，✗ 要反向拉或经调度机中转），探测于 {ts(r['t'])}\n")
    print(table([[*row, ""] for row in rows]))
    if a.verbose:
        for s in names:
            for d, via in sorted((r["routes"].get(s) or {}).items()):
                if via:
                    print(f"  {s} → {d}: {via}")


def cmd_token(a, cfg, con):
    print((cfg.state_dir / "token").read_text().strip())


def where(j) -> str:
    return " + ".join(f"{p['host']}:{','.join(map(str, p['gpus']))}" for p in j["placement"])


def cmd_ls(a, cfg, con):
    now = time.time()
    if a.all:
        rows = db.jobs(con, limit=a.limit)
    else:
        rows = db.jobs(con, db.ACTIVE) + db.jobs(con, db.FINAL, limit=a.limit, since=now - 86400)
    stats = db.kv_get(con, "jobstats", {})
    out = [["ID", "状态", "P", "名称", "资源", "位置", "时间", "原因"]]
    for j in rows:
        res = (
            f"{j['gpus']}卡"
            + (f"×{j['nodes']}" if j["nodes"] > 1 else "")
            + (f"/{j['mem_gb']:g}G" if j["mem_gb"] else "")
        )
        if j["state"] == "running":
            t = "运行 " + dur(now - j["started"])
        elif j["ended"]:
            t = ts(j["ended"])
        else:
            t = "等 " + dur(now - j["submitted"])
        if j.get("kind") == "transfer":
            res = "传输"
        elif j["state"] == "running" and str(j["id"]) in stats:
            res += f" {stats[str(j['id'])]['util']}%"
        st = STATE.get(j["state"], j["state"]) + (f"({j['exit_code']})" if j["state"] == "failed" else "")
        out.append([str(j["id"]), st, str(j["priority"]), j["name"][:28], res, where(j) or "-", t, j["reason"][:80]])
    print(table(out) if len(out) > 1 else "队列是空的")
    snap = db.kv_get(con, "hosts")
    ext = (snap or {}).get("external") or []
    if ext:
        rows = [["机器", "GPU", "用户", "名称", "显存", "已运行", "命令"]]
        for e in ext:
            rows.append([e["host"], ",".join(map(str, e["gpus"])), e["user"], e["name"][:28],
                         f"{e['mem_mib'] / 1024:.0f}G", dur(e["elapsed"]), e["cmd"][:90]])  # fmt: skip
        print("\n队列外正在运行（自动发现，调度会避开）\n" + table(rows))
    if snap and now - snap["t"] > 3 * cfg.poll_s + 30:
        print(f"\n!! 守护进程 {dur(now - snap['t'])} 没有采样了：gq daemon 还在跑吗？")
    elif not snap:
        print("\n!! 还没有守护进程的采样：先启动 gq daemon")


def cmd_show(a, cfg, con):
    j = db.get(con, a.id)
    if not j:
        sys.exit(f"#{a.id} 不存在")
    if a.json:
        print(json.dumps(j, ensure_ascii=False, indent=1))
        return
    for k in ("id", "kind", "name", "state", "reason", "priority", "gpus", "nodes", "mem_gb", "hosts", "after",
              "after_any", "wait_files", "cwd", "env", "retries", "attempt", "time_limit", "exit_code", "submit_info"):  # fmt: skip
        print(f"{k:12} {j[k]}")
    for k in ("submitted", "started", "ended"):
        print(f"{k:12} {ts(j[k])}")
    print(f"{'cmd':12} {j['cmd']}")
    for p in j["placement"]:
        h = cfg.host(p["host"])
        print(
            f"{'rank ' + str(p['rank']):12} {p['host']} GPU {p['gpus']} pid {p['pid']}  log {h.jobdir}/{p['dir']}/log"
        )
    if j["tail"]:
        print("---- 结束时日志末尾 ----\n" + j["tail"].rstrip())


def cmd_log(a, cfg, con):
    j = db.get(con, a.id)
    if not j:
        sys.exit(f"#{a.id} 不存在")
    if not j["placement"]:
        print(j["tail"] or f"#{a.id} 还没开始（{j['reason']}）")
        return
    p = j["placement"][min(a.rank, len(j["placement"]) - 1)]
    h = cfg.host(p["host"])
    if a.follow:
        os.execvp(remote.follow_argv(h, p["dir"], a.n)[0], remote.follow_argv(h, p["dir"], a.n))
    print(remote.tail(h, p["dir"], a.n), end="")


def cmd_hosts(a, cfg, con):
    snap = db.kv_get(con, "hosts")
    if not snap:
        sys.exit("还没有采样：先启动 gq daemon")
    print(f"采样于 {ts(snap['t'])}（{dur(time.time() - snap['t'])} 前）")
    for h in snap["hosts"]:
        if not h["up"]:
            print(f"\n{h['name']}  离线  {h['err'][:100]}")
            continue
        idle = sum(g["idle"] for g in h["gpus"])
        print(
            f"\n{h['name']}  {idle}/{len(h['gpus'])} 空闲  {' '.join(h['tags'])}{'  (需点名)' if h['opt_in'] else ''}"
        )
        for g in h["gpus"]:
            who = [f"#{x['id']} {x['name']}" for x in g["jobs"]]
            who += [f"[外] {f['user']}:{f['pid']} {f.get('short', '')}" for f in g["foreign"]]
            if not who and not g["idle"]:
                who = ["[外] 看不到进程"]
            print(
                f"  {g['idx']}  {g['used'] / 1024:5.1f}/{g['total'] / 1024:4.0f}G {g['util']:3d}%  "
                + (" · ".join(who) or "空闲")
            )


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gq", description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", help="hosts.toml (default: next to gq, or $GQ_CONFIG)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "submit", help="queue a job", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG
    )
    s.add_argument("-n", "--name", default="")
    s.add_argument("-g", "--gpus", type=int, default=1, help="GPUs per node (0 = CPU job; default 1)")
    s.add_argument("-N", "--nodes", type=int, default=1, help="nodes, of one group (default 1)")
    s.add_argument("-m", "--mem", type=float, default=None, help="GB per GPU: may share a GPU that has this much free")
    s.add_argument("-H", "--host", action="append", default=[], help="host names / tags, globs, comma-separated")
    s.add_argument("-p", "--priority", type=int, default=0, help="higher first (default 0)")
    s.add_argument("-a", "--after", action="append", default=[], help="job ids that must succeed first")
    s.add_argument("--after-any", action="store_true", help="start once --after jobs ended, whatever the outcome")
    s.add_argument("-w", "--wait-file", action="append", default=[], help="<host>:<path> that must exist first")
    s.add_argument("-C", "--cwd", default=None, help="working dir on the host (default: the job dir; local jobs: here)")
    s.add_argument("-e", "--env", action="append", default=[], help="KEY=VALUE")
    s.add_argument("--retry", type=int, default=0, help="rerun this many times if it fails")
    s.add_argument("--no-cvd", action="store_true", help="don't set CUDA_VISIBLE_DEVICES (use $GQ_GPUS yourself)")
    s.add_argument("--hold", action="store_true", help="queue held; gq release <id> to let it run")
    s.add_argument("-t", "--time", default=None, help="time limit (90m, 8h, 2d): stopped and marked failed after it")
    s.add_argument("-q", "--quiet", action="store_true", help="print only the id")
    s.add_argument("cmd", nargs=argparse.REMAINDER)

    s = sub.add_parser(
        "cp",
        help="queue a transfer between machines (rsync / tar)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""paths follow rsync: SRC dir copies the directory into DST, SRC dir/ copies its contents. DST is a directory.
the transfer runs on the source (push) or the destination (pull) if one can ssh to the other, else through the
scheduler (tar over ssh). gq net shows who reaches whom; gq net --authorize A B lets A ssh to B.
examples:
  gq cp ws0:/home/me/data/clean_v1 node3:/data/dataset/
  gq cp -a 12 node4:/data/train_out/run_x/ ws1:/data/ckpt/run_x
  id=$(gq cp -q ws0:/home/me/ds node3:/data/dataset/) && gq submit -a $id -g 8 -H node3 -- bash train.sh""",
    )
    s.add_argument("src", help="<host>:/abs/path")
    s.add_argument("dst", help="<host>:/abs/dir")
    s.add_argument("-n", "--name", default="")
    s.add_argument("-p", "--priority", type=int, default=0)
    s.add_argument("-a", "--after", action="append", default=[], help="job ids that must succeed first")
    s.add_argument("--after-any", action="store_true")
    s.add_argument("--retry", type=int, default=2, help="reruns if it fails (rsync resumes; default 2)")
    s.add_argument("--delete", action="store_true", help="rsync --delete: remove files in DST that are not in SRC")
    s.add_argument("--bwlimit", default=None, help="rsync --bwlimit, e.g. 50m")
    s.add_argument("-t", "--time", default=None, help="time limit")
    s.add_argument("--hold", action="store_true")
    s.add_argument("-q", "--quiet", action="store_true", help="print only the id")

    s = sub.add_parser("net", help="who can ssh to whom (used to pick push / pull / relay for gq cp)")
    s.add_argument("--probe", action="store_true", help="test every pair now (~30 s)")
    s.add_argument("--authorize", nargs=2, metavar=("A", "B"), help="add A's key to B's authorized_keys, then probe")
    s.add_argument("-v", "--verbose", action="store_true", help="show the address used for each pair")
    sub.add_parser("token", help="the password of the web page's submit form")

    s = sub.add_parser("ls", help="the queue (active + last 24 h)")
    s.add_argument("-a", "--all", action="store_true")
    s.add_argument("-l", "--limit", type=int, default=20)
    s = sub.add_parser("show", help="one job")
    s.add_argument("id", type=int)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("log", help="a job's log")
    s.add_argument("id", type=int)
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("-n", type=int, default=100)
    s.add_argument("-r", "--rank", type=int, default=0)
    for name, hlp in (("cancel", "cancel (running: TERM, KILL after 30 s)"), ("hold", "keep a pending job from starting"),
                      ("release", "let a held job run"), ("resubmit", "queue a copy of a job")):  # fmt: skip
        s = sub.add_parser(name, help=hlp)
        s.add_argument("ids", type=int, nargs="+")
    s = sub.add_parser("prio", help="set a job's priority")
    s.add_argument("id", type=int)
    s.add_argument("priority", type=int)
    sub.add_parser("hosts", help="GPUs of every host (from the daemon's last probe)")
    s = sub.add_parser("daemon", help="run the scheduler (and the web page) in the foreground")
    s.add_argument("--no-web", action="store_true")
    s.add_argument("--once", action="store_true", help="one tick, then exit")
    sub.add_parser("web", help="only the web page")

    a = p.parse_args(argv)
    if a.command == "submit" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    cfg = config.load(a.config)
    if a.command == "daemon":
        from . import daemon

        return daemon.main(cfg, web=not a.no_web, once=a.once)
    if a.command == "web":
        from . import web

        print(f"http://{cfg.web_host}:{cfg.web_port}/")
        return web.serve(cfg)
    con = db.connect(cfg.db_path)
    if a.command in ("cancel", "hold", "release", "resubmit"):
        for i in a.ids:
            print(getattr(ops, a.command)(con, i))
        poke(cfg)
        return
    if a.command == "prio":
        print(ops.prio(con, a.id, a.priority))
        poke(cfg)
        return
    cmds = {"submit": cmd_submit, "cp": cmd_cp, "net": cmd_net, "token": cmd_token, "ls": cmd_ls, "show": cmd_show,
            "log": cmd_log, "hosts": cmd_hosts}  # fmt: skip
    cmds[a.command](a, cfg, con)


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except subprocess.TimeoutExpired as e:
        sys.exit(f"timeout: {e}")
