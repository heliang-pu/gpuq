"""The machine pool and defaults, read from hosts.toml (or $GQ_CONFIG)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import fnmatch
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Host:
    name: str
    ssh: str = ""  # "" = this machine
    tags: list[str] = field(default_factory=list)
    group: str = ""
    addr: str = ""
    jobdir: str = "~/gq"
    share: bool = True
    opt_in: bool = False
    idle_mib: int = 2000
    headroom_gb: float = 4.0
    max_gpus: int = 0  # 0 = every GPU of the host
    env: dict[str, str] = field(default_factory=dict)
    reach: list[str] = field(default_factory=list)  # how other machines ssh here: user@host[:port], tried in order
    compute: bool = True  # False: a transfer endpoint only (NAS, ...), never gets GPU jobs

    def matches(self, pattern: str) -> bool:
        return fnmatch.fnmatch(self.name, pattern) or any(fnmatch.fnmatch(t, pattern) for t in self.tags)


@dataclass
class Config:
    hosts: list[Host]
    state_dir: Path
    poll_s: float = 20
    web_host: str = "0.0.0.0"
    web_port: int = 8796
    port_base: int = 29500
    notify_cmd: str = ""
    web_url: str = ""  # put in notifications
    idle_alert_min: float = 30  # a whole compute host idle this long -> notify (0 = off)
    hung_alert_min: float = 30  # a running GPU job at ~0 % utilisation this long -> notify (0 = off)
    offline_alert_min: float = 10  # a host unreachable this long -> notify (0 = off)
    history_days: float = 14

    def host(self, name: str) -> Host | None:
        return next((h for h in self.hosts if h.name == name), None)

    def allowed(self, patterns: list[str]) -> list[Host]:
        """Hosts a job may use: those matching one of --host, or every host that is not opt_in."""
        if not patterns:
            return [h for h in self.hosts if not h.opt_in and h.compute]
        return [h for h in self.hosts if any(h.matches(p) for p in patterns)]

    @property
    def local(self) -> Host | None:
        """The host the scheduler runs on (ssh = "")."""
        return next((h for h in self.hosts if not h.ssh), None)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "gq.db"


def load(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("GQ_CONFIG") or ROOT / "hosts.toml")
    raw = tomllib.loads(path.read_text())
    d = raw.get("defaults", {})
    groups = raw.get("group", {})
    hosts = []
    for h in raw.get("host", []):
        h = dict(h)
        env = dict(groups.get(h.get("group", ""), {}).get("env", {}))
        env.update(h.pop("env", {}))
        base = {k: d[k] for k in ("idle_mib", "headroom_gb") if k in d}
        hosts.append(Host(**{**base, **h, "env": env}))
    names = [h.name for h in hosts]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host names in {path}")
    return Config(
        hosts=hosts,
        state_dir=Path(os.path.expanduser(d.get("state_dir", "~/.local/share/gq"))),
        poll_s=float(d.get("poll_s", 20)),
        web_host=d.get("web_host", "0.0.0.0"),
        web_port=int(d.get("web_port", 8796)),
        port_base=int(d.get("port_base", 29500)),
        notify_cmd=d.get("notify_cmd", ""),
        web_url=d.get("web_url", ""),
        idle_alert_min=float(d.get("idle_alert_min", 30)),
        hung_alert_min=float(d.get("hung_alert_min", 30)),
        offline_alert_min=float(d.get("offline_alert_min", 10)),
        history_days=float(d.get("history_days", 14)),
    )
