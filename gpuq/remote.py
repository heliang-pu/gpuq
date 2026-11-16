"""Everything that touches a host: one `bash -s` per call (over ssh, or locally for ssh = ""), the script on stdin, so
nothing is quoted twice and `pgrep -f` style self-matches cannot happen."""

from __future__ import annotations

import base64
import shlex
import subprocess
import time

from .config import Host

SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=2",
]


def argv(host: Host, tty: bool = False) -> list[str]:
    if not host.ssh:
        return ["bash", "-s"]
    return ["ssh", *SSH_OPTS, *(["-t"] if tty else []), host.ssh, "bash -s"]


def run(host: Host, script: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(argv(host), input=script, capture_output=True, text=True, timeout=timeout)


def jd(host: Host) -> str:
    """The job root as a shell word (a leading ~ becomes $HOME)."""
    if host.jobdir.startswith("~/"):
        return '"$HOME"/' + shlex.quote(host.jobdir[2:])
    return shlex.quote(host.jobdir)


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def probe(host: Host, entries: list[tuple[str, int]]) -> dict:
    """GPUs, the processes on them (with session id, user and command) and the state of the given job directories
    (dir, pid): ('exit', rc) | ('run',) | ('gone',)."""
    pairs = " ".join(f"{d}:{p}" for d, p in entries)
    script = f"""JD={jd(host)}
echo @gpu
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1
echo @app
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r pid uuid mem; do
  [ -n "$pid" ] || continue
  read -r sid user et args <<< "$(ps -o sid=,user=,etimes=,args= -p "$pid" 2>/dev/null)"
  lead=$(ps -o args= -p "${{sid:-0}}" 2>/dev/null | head -c 300)
  printf '%s\\x1f%s\\x1f%s\\x1f%s\\x1f%s\\x1f%s\\x1f%s\\x1f%s\\n' "$pid" "$uuid" "$mem" "${{sid:-0}}" "${{user:-?}}" "${{et:-0}}" "${{args:0:300}}" "$lead"
done
echo @job
for e in {pairs}; do
  d=${{e%%:*}}; p=${{e##*:}}; f="$JD/$d"
  if [ -f "$f/exit_code" ]; then echo "$d exit $(cat "$f/exit_code")"
  elif [ -r /proc/$p/cmdline ] && tr '\\0' ' ' < /proc/$p/cmdline | grep -qF "$d/run.sh"; then echo "$d run"
  elif [ -f "$f/exit_code" ]; then echo "$d exit $(cat "$f/exit_code")"
  else echo "$d gone"; fi
done
echo @end
"""
    t0 = time.time()
    try:
        r = run(host, script, timeout=40)
    except subprocess.TimeoutExpired:
        return {"up": False, "err": "timeout", "t": t0}
    if "@end" not in r.stdout:
        return {"up": False, "err": (r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}", "t": t0}
    out = {"up": True, "err": "", "t": t0, "gpus": [], "apps": [], "jobs": {}}
    sec = None
    for line in r.stdout.splitlines():
        if line.startswith("@"):
            sec = line[1:]
            continue
        if sec == "gpu":
            p = [x.strip() for x in line.split(",")]
            if len(p) == 5 and p[0].isdigit():
                out["gpus"].append(
                    {"idx": int(p[0]), "uuid": p[1], "used": _int(p[2]), "total": _int(p[3]), "util": _int(p[4])}
                )
            elif line.strip():
                out["err"] = line.strip()[:200]
        elif sec == "app":
            p = line.split("\x1f")
            if len(p) < 8 or not p[0].isdigit():
                continue
            out["apps"].append(
                {
                    "pid": int(p[0]),
                    "uuid": p[1],
                    "mem": _int(p[2]),
                    "sid": _int(p[3]),
                    "user": p[4],
                    "etime": _int(p[5]),  # seconds since the process started
                    "args": p[6],
                    "lead": p[7],  # command of the session leader (often the launch script)
                }
            )
        elif sec == "job":
            p = line.split()
            if len(p) >= 2:
                out["jobs"][p[0]] = ("exit", _int(p[2]) if len(p) > 2 else -1) if p[1] == "exit" else (p[1],)
    return out


def _int(s: str) -> int:
    try:
        return int(float(s))
    except ValueError:
        return 0


def launch(host: Host, dirname: str, run_sh: str, cmd_sh: str) -> int:
    """Write run.sh / cmd.sh into <jobdir>/<dirname> and start run.sh in its own session; returns its pid (= the
    session and process group id, so a kill of -pid reaches every child)."""
    script = f"""set -e
D={jd(host)}/{dirname}
mkdir -p "$D"
rm -f "$D/exit_code" "$D/pid"
echo {_b64(run_sh)} | base64 -d > "$D/run.sh"
echo {_b64(cmd_sh)} | base64 -d > "$D/cmd.sh"
cd "$D"
setsid nohup bash "$D/run.sh" > "$D/log" 2>&1 < /dev/null &
for i in $(seq 100); do [ -s "$D/pid" ] && break; sleep 0.1; done
cat "$D/pid"
"""
    r = run(host, script, timeout=40)
    pid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if r.returncode != 0 or not pid.isdigit():
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}")
    return int(pid)


def kill(host: Host, pid: int, sig: str = "TERM") -> None:
    run(host, f"kill -{sig} -- -{int(pid)} 2>/dev/null; kill -{sig} {int(pid)} 2>/dev/null; true", timeout=20)


def tail(host: Host, dirname: str, n: int = 200) -> str:
    try:
        r = run(host, f"tail -n {int(n)} {jd(host)}/{dirname}/log 2>&1", timeout=20)
        return r.stdout
    except subprocess.TimeoutExpired:
        return "(timeout)"


def exists(host: Host, path: str) -> bool | None:
    """True / False, or None if the host could not be reached."""
    try:
        r = run(host, f"test -e {shlex.quote(path)} && echo yes || echo no", timeout=20)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout.strip()
    return True if out == "yes" else False if out == "no" else None


def follow_argv(host: Host, dirname: str, n: int) -> list[str]:
    cmd = f"tail -n {int(n)} -F {host.jobdir}/{dirname}/log"
    return ["bash", "-c", cmd] if not host.ssh else ["ssh", *SSH_OPTS, host.ssh, cmd]
