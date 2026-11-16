"""Queue operations shared by the CLI and the web page. They only change the database; the daemon acts on it."""

from __future__ import annotations

import time

from . import db

COPY = ("name", "cmd", "cwd", "gpus", "nodes", "mem_gb", "hosts", "priority", "after_any", "wait_files", "env", "cvd", "kind", "time_limit",
        "retries", "submit_info")  # fmt: skip


def cancel(con, job_id: int) -> str:
    j = db.get(con, job_id)
    if j is None:
        return f"#{job_id} 不存在"
    if j["state"] in ("pending", "held"):
        db.update(con, job_id, ("pending", "held"), state="cancelled", ended=time.time(), reason="排队时取消")
        return f"#{job_id} 已取消"
    if j["state"] in ("running", "starting"):
        db.update(con, job_id, cancel_req=time.time(), reason="取消中")
        return f"#{job_id} 正在终止（先 TERM，30 s 后 KILL）"
    return f"#{job_id} 已经是 {j['state']}"


def hold(con, job_id: int) -> str:
    ok = db.update(con, job_id, "pending", state="held", reason="已暂缓（gq release 放行）")
    return f"#{job_id} " + ("已暂缓" if ok else "不在排队，不能暂缓")


def release(con, job_id: int) -> str:
    ok = db.update(con, job_id, "held", state="pending", reason="")
    return f"#{job_id} " + ("已放行" if ok else "不是暂缓状态")


def prio(con, job_id: int, priority: int) -> str:
    ok = db.update(con, job_id, priority=int(priority))
    return f"#{job_id} 优先级 = {priority}" if ok else f"#{job_id} 不存在"


def resubmit(con, job_id: int) -> str:
    j = db.get(con, job_id)
    if j is None:
        return f"#{job_id} 不存在"
    f = {k: j[k] for k in COPY}
    f["submit_info"] = (f["submit_info"] + f" (重跑 #{job_id})").strip()
    new = db.insert(con, **f)
    return f"#{job_id} 重新提交为 #{new}"


def parse_duration(s) -> float | None:
    """'90m', '8h', '1.5d', '3600' (s) -> seconds; None / '' -> None."""
    if s in (None, ""):
        return None
    s = str(s).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(s[-1])
    try:
        return float(s[:-1]) * mult if mult else float(s)
    except ValueError:
        raise ValueError(f"时长 {s!r} 看不懂：写成 90m / 8h / 1.5d") from None


def check_linked(cfg, con, allowed, gpus: int, nodes: int) -> None:
    """Multi-GPU jobs only get GPUs that are linked: the GPUs of one machine (the A100s are NVSwitch all-to-all), or
    `nodes` machines of one IB group. Single-GPU machines are never pooled. Reject a job no machine could ever hold
    (it would wait forever). GPU counts come from the daemon's last probe; machines not seen yet are given the benefit
    of the doubt."""
    if gpus <= 1 and nodes == 1:
        return
    counts = {h["name"]: len(h["gpus"]) for h in db.kv_get(con, "hosts", {}).get("hosts", []) if h["up"]}
    fits = [h for h in allowed if h.compute and counts.get(h.name, gpus) >= gpus]
    if nodes > 1:
        groups = {h.group for h in fits if h.group and sum(x.group == h.group for x in fits) >= nodes}
        if not groups:
            raise ValueError(
                f"{nodes} 机 × {gpus} 卡：多机只能用 hosts.toml 里同一组的机器，"
                f"候选 {[h.name for h in allowed]} 里没有 {nodes} 台同组、每台至少 {gpus} 卡的"
            )
        return
    if not fits:
        have = ", ".join(f"{h.name} {counts[h.name]} 卡" for h in allowed if h.name in counts)
        raise ValueError(
            f"{gpus} 卡任务只能用同一台机器里互联的卡，不会把不同机器的卡拼起来（{have}）。"
            f"要更多卡请用 -N 多机（只限 IB 级联的同组 A100）"
        )


def submit(cfg, con, spec: dict) -> int:
    """Validate and queue a GPU/CPU job. spec: cmd (str), name, gpus, nodes, mem_gb, hosts (list), priority,
    after (list of ids), after_any, wait_files (list), cwd, env (dict), retries, cvd, hold, time_limit, submit_info.
    Raises ValueError with a message for the user."""
    cmd = (spec.get("cmd") or "").strip()
    if not cmd:
        raise ValueError("命令是空的")
    raw = spec.get("hosts") or []
    hosts = [y.strip() for x in ([raw] if isinstance(raw, str) else raw) for y in x.split(",") if y.strip()]
    allowed = cfg.allowed(hosts)
    if not allowed:
        raise ValueError(f"--host {hosts} 没有匹配的机器（{', '.join(h.name for h in cfg.hosts)}）")
    gpus, nodes = int(spec.get("gpus", 1)), int(spec.get("nodes", 1))
    if gpus < 0 or nodes < 1:
        raise ValueError("卡数 >= 0，机器数 >= 1")
    if nodes > 1 and not any(sum(x.group == h.group for x in allowed) >= nodes for h in allowed if h.group):
        raise ValueError(
            f"{nodes} 机任务只能用 hosts.toml 里同一组的机器，候选 {[h.name for h in allowed]} 里没有 {nodes} 台同组的"
        )
    if gpus > 0 and not any(h.compute for h in allowed):
        raise ValueError("这些机器不是算力机，只能跑 -g 0 的任务")
    check_linked(cfg, con, allowed, gpus, nodes)
    raw = spec.get("after") or []
    after = [int(y) for x in ([raw] if isinstance(raw, (str, int)) else raw) for y in str(x).replace(",", " ").split()]
    for d in after:
        if db.get(con, d) is None:
            raise ValueError(f"依赖的任务 #{d} 不存在")
    for w in spec.get("wait_files") or []:
        if ":" not in w or cfg.host(w.split(":", 1)[0]) is None:
            raise ValueError(f"等待文件 {w}：要写成 <hosts.toml 里的机器>:<路径>")
    mem = spec.get("mem_gb")
    jid = db.insert(
        con,
        name=(spec.get("name") or cmd.split()[-1].split("/")[-1])[:60],
        cmd=cmd,
        cwd=spec.get("cwd") or "",
        gpus=gpus,
        nodes=nodes,
        mem_gb=None if mem in (None, "") else float(mem),
        hosts=hosts,
        priority=int(spec.get("priority", 0)),
        after=after,
        after_any=int(bool(spec.get("after_any"))),
        wait_files=spec.get("wait_files") or [],
        env=spec.get("env") or {},
        cvd=int(spec.get("cvd", 1)),
        retries=int(spec.get("retries", 0)),
        state="held" if spec.get("hold") else "pending",
        time_limit=parse_duration(spec.get("time_limit")),
        submit_info=spec.get("submit_info", ""),
        kind="run",
    )
    return jid


def submit_transfer(cfg, con, src: str, dst: str, spec: dict) -> tuple[int, str]:
    """Queue a transfer src_host:/path -> dst_host:/dir. Returns (id, how it will run)."""
    from . import transfer

    ends = []
    for end in (src, dst):
        host, sep, path = end.partition(":")
        h = cfg.host(host)
        if not sep or h is None:
            raise ValueError(f"{end}：要写成 <机器>:<绝对路径>，机器是 {', '.join(x.name for x in cfg.hosts)} 之一")
        if not path.startswith("/"):
            raise ValueError(f"{end}：路径要写绝对路径")
        ends.append((h, path))
    (sh, sp), (dh, dp) = ends
    routes = db.kv_get(con, "routes", {}).get("routes", {})
    runner, cmd, how = transfer.plan(cfg, routes, sh, sp, dh, dp, spec.get("delete"), spec.get("bwlimit"))
    jid = db.insert(
        con,
        name=(spec.get("name") or f"传输 {sh.name}→{dh.name} {sp.rstrip('/').rsplit('/', 1)[-1]}")[:60],
        cmd=cmd,
        gpus=0,
        hosts=[runner.name],
        priority=int(spec.get("priority", 0)),
        after=[int(x) for x in spec.get("after") or []],
        after_any=int(bool(spec.get("after_any"))),
        retries=int(spec.get("retries", 2)),
        state="held" if spec.get("hold") else "pending",
        time_limit=parse_duration(spec.get("time_limit")),
        submit_info=(spec.get("submit_info", "") + f" {src} -> {dst} ({how})").strip(),
        kind="transfer",
        cvd=0,
    )
    return jid, how
