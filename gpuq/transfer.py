"""Transfers between machines: who can ssh to whom (probed), and the rsync / tar command of a transfer job.

A transfer is an ordinary CPU job (gpus 0) pinned to the machine that runs it:
- push: rsync on the source, when the source can ssh to the destination;
- pull: rsync on the destination, when it can ssh to the source;
- relay: neither can reach the other -> tar over two ssh connections through the scheduler host (which reaches every
  machine). Works always, but the bytes go through the scheduler's network.
Same machine: rsync locally.

Paths follow rsync: `src/dir` copies the directory into DST (DST/dir), `src/dir/` copies its contents into DST.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import posixpath
import shlex

from . import remote
from .config import Config
from .config import Host

RSYNC = "rsync -aHs --no-owner --no-group --partial --mkpath --info=progress2,stats1 --no-inc-recursive"
SSH_E = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"


def split(reach: str) -> tuple[str, str, int]:
    """user@host[:port] -> (user, host, port)."""
    user, _, rest = reach.rpartition("@")
    host, _, port = rest.partition(":")
    return user, host, int(port or 22)


def candidates(src: Host, dst: Host) -> list[str]:
    """Ways src may ssh to dst, best first: the fast internal address inside a group, then dst.reach."""
    out = []
    if src.group and src.group == dst.group and dst.addr and dst.reach:
        user, _, port = split(dst.reach[0])
        out += [f"{user}@{dst.addr}:22"] + ([f"{user}@{dst.addr}:{port}"] if port != 22 else [])
    return out + [r for r in dst.reach if r not in out]


def probe_routes(cfg: Config) -> dict[str, dict[str, str | None]]:
    """{src: {dst: the first working user@host:port or None}} for every pair, tested from src itself."""

    def one(src: Host) -> tuple[str, dict]:
        tests = {d.name: candidates(src, d) for d in cfg.hosts if d is not src}
        lines = []
        for dname, cands in tests.items():
            tries = " || ".join(
                f"{{ timeout 12 {SSH_E} -o ConnectTimeout=6 -p {p} {shlex.quote(u + '@' + h)} true </dev/null "
                f">/dev/null 2>&1 && echo 'R {dname} {c}'; }}"
                for c in cands
                for u, h, p in [split(c)]
            )
            lines.append(f"( {tries or 'true'} ) &")
        script = "\n".join(lines) + "\nwait\necho @end\n"
        try:
            r = remote.run(src, script, timeout=90)
        except Exception:  # noqa: BLE001
            return src.name, {d: None for d in tests}
        if "@end" not in r.stdout:
            return src.name, {d: None for d in tests}
        found = {d: None for d in tests}
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) == 3 and p[0] == "R":
                found[p[1]] = p[2]
        return src.name, found

    with ThreadPoolExecutor(max_workers=len(cfg.hosts)) as ex:
        return dict(ex.map(one, cfg.hosts))


def _q(path: str) -> str:
    return shlex.quote(path)


def _ssh_e(port: int) -> str:
    return f"{SSH_E} -p {port}"


def plan(cfg: Config, routes: dict, src: Host, spath: str, dst: Host, dpath: str, delete=False, bwlimit=None):
    """-> (runner Host, shell command, how: same|push|pull|relay)."""
    opts = RSYNC + (" --delete" if delete else "") + (f" --bwlimit={bwlimit}" if bwlimit else "")
    dest = dpath if dpath.endswith("/") else dpath + "/"  # rsync into this directory
    if src is dst:
        return src, f"{opts} {_q(spath)} {_q(dest)}", "same"
    via = (routes.get(src.name) or {}).get(dst.name)
    if via:
        u, h, p = split(via)
        return src, f"{opts} -e {_q(_ssh_e(p))} {_q(spath)} {_q(f'{u}@{h}:{dest}')}", "push"
    via = (routes.get(dst.name) or {}).get(src.name)
    if via:
        u, h, p = split(via)
        return dst, f"{opts} -e {_q(_ssh_e(p))} {_q(f'{u}@{h}:{spath}')} {_q(dest)}", "pull"
    hub = cfg.local
    if hub is None:
        raise ValueError(f"{src.name} 和 {dst.name} 互相连不上，也没有能中转的调度机")
    if spath.endswith("/"):
        tar_c = f"tar -C {_q(spath)} -cf - ."
    else:
        tar_c = f"tar -C {_q(posixpath.dirname(spath.rstrip('/')) or '/')} -cf - {_q(posixpath.basename(spath))}"
    # drain after tar: it stops at the end-of-archive marker, and the unread padding would SIGPIPE the sender
    tar_x = f"mkdir -p {_q(dest)} && tar -C {_q(dest)} --no-same-owner -xf - && cat > /dev/null"

    def on(h: Host, cmd: str) -> str:
        return f"{{ {cmd}; }}" if not h.ssh else f"{SSH_E} {_q(h.ssh)} {_q(cmd)}"

    return hub, f"set -o pipefail; {on(src, tar_c)} | {on(dst, tar_x)}", "relay"


def authorize(cfg: Config, src: Host, dst: Host) -> str:
    """Let src ssh to dst: append src's ~/.ssh/id_ed25519.pub (created if missing, no passphrase) to dst's
    authorized_keys. Both are reached from the scheduler."""
    r = remote.run(
        src,
        "[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519 -C gq-$(hostname); "
        "cat ~/.ssh/id_ed25519.pub",
        timeout=30,
    )
    key = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if not key.startswith("ssh-"):
        raise RuntimeError(f"拿不到 {src.name} 的公钥：{r.stderr.strip()[-200:]}")
    r = remote.run(
        dst,
        f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
        f"(grep -qF {_q(key)} ~/.ssh/authorized_keys && echo already || (echo {_q(key)} >> ~/.ssh/authorized_keys "
        f"&& echo added))",
        timeout=30,
    )
    return f"{src.name} 的公钥 → {dst.name}：{r.stdout.strip() or r.stderr.strip()[-200:]}"
