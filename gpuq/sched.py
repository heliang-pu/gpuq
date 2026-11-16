"""Placement: pure functions of the host snapshots and the queue, so they can be tested without machines.

Rules
- Exclusive job (mem_gb None): a GPU qualifies if no gq job holds it and it uses < host.idle_mib. Memory used by
  processes outside the queue therefore makes a GPU busy: we wait for them, never touch them.
- Shared job (mem_gb set, host.share): a GPU qualifies if no exclusive gq job holds it and
  total - used - reserve - headroom >= mem, where reserve = memory promised to gq jobs on it and not yet allocated
  (attributed per job through the session id of its processes, = the pid of its run.sh).
- Order: priority (high first), then id. A job that could run on some hosts but finds no room reserves those hosts:
  lower jobs may not start there (no starvation of 8- or 16-GPU jobs by a stream of small ones). Jobs whose hosts are
  not reserved still start (backfill on other machines).
- Single node: best fit, i.e. the host with the fewest qualifying GPUs that still has enough (keeps whole machines free).
- Multi node: `nodes` hosts of one group, each with `gpus` qualifying GPUs; rank 0 = first of them in config order.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from .config import Host

MiB = 1024


@dataclass
class Gpu:
    idx: int
    total: int  # MiB
    used: int
    util: int = 0
    uuid: str = ""
    held: list = field(default_factory=list)  # [(job_id, mem_mib or None)]
    reserve: int = 0  # MiB promised to gq jobs, not yet allocated

    def exclusive_held(self) -> bool:
        return any(m is None for _, m in self.held)

    def free(self) -> int:
        return self.total - self.used - self.reserve


def build_view(hosts: list[Host], snaps: dict, running: list[dict]) -> dict[str, list[Gpu] | None]:
    """name -> GPUs gq may use (None = host down), with the GPUs held by running gq jobs marked."""
    view: dict[str, list[Gpu] | None] = {}
    for h in hosts:
        s = snaps.get(h.name)
        if not s or not s.get("up"):
            view[h.name] = None
            continue
        if not s.get("gpus"):
            view[h.name] = None if h.compute else []  # a transfer endpoint: up, no GPUs, takes CPU jobs
            continue
        gpus = [Gpu(g["idx"], g["total"], g["used"], g.get("util", 0), g.get("uuid", "")) for g in s["gpus"]]
        gpus.sort(key=lambda g: g.idx)
        if h.max_gpus:
            gpus = gpus[: h.max_gpus]
        view[h.name] = gpus
    for j in running:
        mem = None if j.get("mem_gb") is None else int(j["mem_gb"] * MiB)
        for p in j.get("placement", []):
            gpus = view.get(p["host"])
            if not gpus:
                continue
            apps = snaps.get(p["host"], {}).get("apps", [])
            for g in gpus:
                if g.idx not in p["gpus"]:
                    continue
                g.held.append((j["id"], mem))
                if mem is not None:
                    seen = sum(a["mem"] for a in apps if a["uuid"] == g.uuid and p.get("pid") and a["sid"] == p["pid"])
                    g.reserve += max(0, mem - seen)
    return view


def qualifying(host: Host, gpus: list[Gpu], job: dict) -> list[Gpu]:
    mem = job.get("mem_gb")
    if mem is None or not host.share:
        return [g for g in gpus if not g.held and g.used < host.idle_mib]
    need = (mem + host.headroom_gb) * MiB
    ok = [g for g in gpus if not g.exclusive_held() and g.free() >= need]
    return sorted(ok, key=lambda g: (g.free(), g.idx))  # pack the fullest card that still fits


def feasible(host: Host, gpus: list[Gpu] | None, job: dict) -> bool:
    """Could this host ever take the job (enough GPUs, big enough)?"""
    if gpus is None:
        return False
    if job["gpus"] == 0:
        return True
    if len(gpus) < job["gpus"]:
        return False
    mem = job.get("mem_gb")
    return mem is None or max(g.total for g in gpus) >= (mem + host.headroom_gb) * MiB


def take(gpus: list[Gpu], chosen: list[Gpu], job: dict) -> None:
    mem = None if job.get("mem_gb") is None else int(job["mem_gb"] * MiB)
    for g in chosen:
        g.held.append((job["id"], mem))
        if mem is not None:
            g.reserve += mem


def plan(jobs: list[dict], allowed, view: dict) -> tuple[list[tuple[dict, list[tuple[Host, list[int]]]]], dict]:
    """jobs: ready to run (dependencies met). allowed(job) -> [Host] in config order.
    Returns ([(job, [(host, gpu indices) per rank])], {job id: why it waits}). Mutates view."""
    starts, reasons, reserved = [], {}, set()
    for job in sorted(jobs, key=lambda j: (-j["priority"], j["id"])):
        hosts = allowed(job)
        if not hosts:
            reasons[job["id"]] = f"--host {','.join(job['hosts'])} 没有匹配的机器"
            continue
        feas = [h for h in hosts if feasible(h, view.get(h.name), job)]
        if job["nodes"] > 1:  # only groups with enough feasible members
            feas = [h for h in feas if h.group and sum(x.group == h.group for x in feas) >= job["nodes"]]
        if not feas:
            down = [h.name for h in hosts if view.get(h.name) is None]
            reasons[job["id"]] = "没有能放下它的在线机器" + (f"（离线：{','.join(down)}）" if down else "")
            continue
        open_ = [h for h in feas if h.name not in reserved]
        placed = _place_multi(job, open_, view) if job["nodes"] > 1 else _place_single(job, open_, view)
        if placed:
            for h, chosen in placed:
                take(view[h.name], chosen, job)
            starts.append((job, [(h, [g.idx for g in chosen]) for h, chosen in placed]))
            continue
        reserved |= {h.name for h in feas}
        what = f"{job['gpus']} 卡" + (f" × {job['nodes']} 机" if job["nodes"] > 1 else "")
        if job.get("mem_gb") is not None:
            what += f"（每卡 {job['mem_gb']:g} GB，可共享）"
        reasons[job["id"]] = f"等 GPU：{what}，候选 {','.join(h.name for h in feas)}"
    return starts, reasons


def _place_single(job, hosts, view):
    best = None
    for h in hosts:
        q = qualifying(h, view[h.name], job)
        if len(q) >= job["gpus"] and (best is None or len(q) < best[2]):
            best = (h, q[: job["gpus"]], len(q))
    return [(best[0], best[1])] if best else None


def _place_multi(job, hosts, view):
    groups: dict[str, list] = {}
    for h in hosts:
        if not h.group:
            continue
        q = qualifying(h, view[h.name], job)
        if len(q) >= job["gpus"]:
            groups.setdefault(h.group, []).append((h, q[: job["gpus"]]))
    for members in groups.values():
        if len(members) >= job["nodes"]:
            return members[: job["nodes"]]
    return None
