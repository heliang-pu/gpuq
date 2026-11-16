"""gq tests: placement rules (pure) and an end-to-end run of CPU jobs on this machine (no GPU, no ssh)."""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from gpuq import config
from gpuq import daemon
from gpuq import db
from gpuq import ops
from gpuq import sched
from gpuq.config import Host

G = 1024  # MiB per GB


def snap(n, used=0, total=80 * G, apps=()):
    gp = [{"idx": i, "uuid": f"u{i}", "used": used[i] if isinstance(used, list) else used, "total": total, "util": 0}
          for i in range(n)]  # fmt: skip
    return {"up": True, "gpus": gp, "apps": list(apps), "jobs": {}}


def job(i, gpus=1, nodes=1, mem=None, prio=0, hosts=()):
    return {"id": i, "gpus": gpus, "nodes": nodes, "mem_gb": mem, "priority": prio, "hosts": list(hosts)}


A = [Host("a1", "x", ["a100"], group="g", addr="10.0.0.1"), Host("a2", "x", ["a100"], group="g"),
     Host("p6", "x", ["ws"]), Host("b0", "", ["bench"], opt_in=True, headroom_gb=6)]  # fmt: skip
CFG = config.Config(hosts=A, state_dir=Path("/nonexistent"))


def run_plan(jobs, snaps, running=()):
    view = sched.build_view(A, snaps, list(running))
    return sched.plan(jobs, lambda j: CFG.allowed(j["hosts"]), view)


def test_foreign_memory_makes_gpu_busy():
    snaps = {"a1": snap(8, used=[15 * G] * 7 + [0]), "a2": None, "p6": None, "b0": None}
    starts, why = run_plan([job(1, gpus=1), job(2, gpus=2)], snaps)
    assert [(j["id"], [(h.name, g) for h, g in r]) for j, r in starts] == [(1, [("a1", [7])])]
    assert "等 GPU" in why[2]


def test_best_fit_keeps_whole_machine_free():
    snaps = {"a1": snap(8), "a2": snap(8, used=[30 * G] * 6 + [0, 0]), "p6": None, "b0": None}
    starts, _ = run_plan([job(1, gpus=2)], snaps)
    assert starts[0][1][0][0].name == "a2" and starts[0][1][0][1] == [6, 7]


def test_reservation_blocks_small_jobs_behind_big_one_but_backfills_elsewhere():
    snaps = {"a1": snap(8, used=[0] * 4 + [20 * G] * 4), "a2": None, "p6": snap(1), "b0": None}
    starts, why = run_plan([job(1, gpus=8, prio=5), job(2, gpus=1), job(3, gpus=1, hosts=["ws"])], snaps)
    # #2 may not take a1's free GPUs from #1, so it backfills p6; #3 then waits for p6
    assert [(j["id"], r[0][0].name) for j, r in starts] == [(2, "p6")]
    assert "a1" in why[1] and "等 GPU" in why[3]


def test_shared_mem_fits_next_to_foreign_and_counts_unallocated_promise():
    apps = [{"pid": 50, "uuid": "u0", "mem": 4 * G, "sid": 50, "user": "root", "args": "x"}]
    snaps = {"a1": None, "a2": None, "p6": snap(1, used=12 * G, total=96 * G, apps=apps), "b0": None}
    # a running gq job (pid 50) promised 30 GB and holds 4 so far: 26 GB more are reserved
    running = [{"id": 9, "mem_gb": 30, "placement": [{"host": "p6", "gpus": [0], "pid": 50}]}]
    starts, why = run_plan([job(1, mem=60, hosts=["p6"])], snaps, running)  # 96-12-26-4 = 54 < 60
    assert not starts
    starts, _ = run_plan([job(2, mem=50, hosts=["p6"])], snaps, running)  # 54 >= 50
    assert starts
    starts, _ = run_plan([job(3, hosts=["p6"])], snaps, running)  # exclusive: never on a used GPU
    assert not starts


def test_opt_in_host_only_when_named():
    snaps = {"a1": None, "a2": None, "p6": None, "b0": snap(1, used=20 * G, total=48 * G)}
    starts, why = run_plan([job(1, mem=10)], snaps)
    assert not starts and "没有能放下" in why[1]
    starts, _ = run_plan([job(2, mem=10, hosts=["bench"])], snaps)  # 48-20-6 = 22 >= 10
    assert starts and starts[0][1][0][0].name == "b0"
    starts, _ = run_plan([job(3, mem=30, hosts=["b0"])], snaps)
    assert not starts


def test_multi_node_takes_one_group_rank0_first_in_config():
    snaps = {"a1": snap(8), "a2": snap(8), "p6": snap(1), "b0": None}
    starts, _ = run_plan([job(1, gpus=8, nodes=2)], snaps)
    assert [h.name for h, _ in starts[0][1]] == ["a1", "a2"]
    snaps["a2"] = snap(8, used=[0] * 7 + [10 * G])
    starts, why = run_plan([job(2, gpus=8, nodes=2)], snaps)
    assert not starts and "a1" in why[2]


def test_run_script_multi_node_env():
    j = {"id": 7, "name": "t", "attempt": 0, "cvd": 1, "nodes": 2, "env": {"X": "a b"}, "cwd": ""}
    s = daemon.run_script(j, A[1], 1, [0, 1], [(A[0], [0, 1]), (A[1], [0, 1])], 29507)
    assert "export MASTER_ADDR=10.0.0.1" in s and "export NODE_RANK=1" in s and "export GQ_NODES=a1,a2" in s
    assert "export CUDA_VISIBLE_DEVICES=0,1" in s and "export X='a b'" in s


# ---- end to end on this machine ----------------------------------------------------------------------------------


@pytest.fixture
def local(tmp_path):
    h = Host("me", "", ["local"], jobdir=str(tmp_path / "jobs"))
    cfg = config.Config(hosts=[h], state_dir=tmp_path / "state", poll_s=1)
    d = daemon.Daemon(cfg)
    return cfg, d


def wait_until(d, cond, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d.tick()
        if cond():
            return
        time.sleep(0.3)
    raise AssertionError("timeout")


def test_end_to_end_deps_retry_cancel(local, tmp_path):
    cfg, d = local
    con = d.con
    out = tmp_path / "out.txt"
    a = db.insert(con, name="a", cmd=f"echo A-$GQ_JOB_ID >> {out}", gpus=0)
    b = db.insert(con, name="b", cmd=f"echo B >> {out}", gpus=0, after=[a])
    flaky = tmp_path / "flaky"
    c = db.insert(con, name="c", cmd=f"test -f {flaky} || {{ touch {flaky}; exit 3; }}", gpus=0, retries=1)
    bad = db.insert(con, name="bad", cmd="exit 5", gpus=0)
    dep_bad = db.insert(con, name="dep", cmd="true", gpus=0, after=[bad])
    slow = db.insert(con, name="slow", cmd="sleep 300", gpus=0)
    wait_until(d, lambda: db.get(con, slow)["state"] == "running")
    ops.cancel(con, slow)
    final = lambda i: db.get(con, i)["state"] in db.FINAL  # noqa: E731
    wait_until(d, lambda: all(final(i) for i in (a, b, c, bad, dep_bad, slow)), timeout=60)
    s = {i: db.get(con, i) for i in (a, b, c, bad, dep_bad, slow)}
    assert s[a]["state"] == s[b]["state"] == "done"
    assert out.read_text() == f"A-{a}\nB\n"
    assert s[c]["state"] == "done" and s[c]["attempt"] == 2
    assert s[bad]["state"] == "failed" and s[bad]["exit_code"] == 5
    assert s[dep_bad]["state"] == "failed" and f"#{bad}" in s[dep_bad]["reason"]
    assert s[slow]["state"] == "cancelled"
    assert "[gq] job" in s[a]["tail"]


def test_wait_file(local, tmp_path):
    cfg, d = local
    flag = tmp_path / "ready"
    j = db.insert(d.con, name="w", cmd="true", gpus=0, wait_files=[f"me:{flag}"])
    d.tick()
    assert db.get(d.con, j)["state"] == "pending" and "等文件" in db.get(d.con, j)["reason"]
    flag.touch()
    wait_until(d, lambda: db.get(d.con, j)["state"] == "done")


# ---- transfers, time limits, submit validation, history ----------------------------------------------------------

from gpuq import transfer  # noqa: E402

T = [Host("s", "s", reach=["u@s.lan"], group="g", addr="10.0.0.1"), Host("d", "d", reach=["root@d.pub:2222"], group="g",
     addr="10.0.0.2"), Host("x", "x", reach=["u@x"]), Host("hub", "", reach=["root@hub"])]  # fmt: skip
TCFG = config.Config(hosts=T, state_dir=Path("/nonexistent"))


def test_candidates_prefer_group_internal_address():
    assert transfer.candidates(T[0], T[1]) == ["root@10.0.0.2:22", "root@10.0.0.2:2222", "root@d.pub:2222"]
    assert transfer.candidates(T[0], T[2]) == ["u@x"]


def test_transfer_plan_push_pull_relay_same():
    s, d, x, hub = T
    runner, cmd, how = transfer.plan(TCFG, {"s": {"d": "root@10.0.0.2:22"}}, s, "/a/ds", d, "/data")
    assert (runner.name, how) == ("s", "push") and "root@10.0.0.2:/data/" in cmd and "-p 22" in cmd
    runner, cmd, how = transfer.plan(TCFG, {"d": {"s": "u@s.lan:22"}}, s, "/a/ds/", d, "/data/")
    assert (runner.name, how) == ("d", "pull") and "u@s.lan:/a/ds/" in cmd
    runner, cmd, how = transfer.plan(TCFG, {}, s, "/a/ds", x, "/b")
    assert (runner.name, how) == ("hub", "relay")
    assert (
        "tar -C /a -cf - ds" in cmd
        and "mkdir -p /b/ && tar -C /b/ --no-same-owner -xf - && cat > /dev/null" in cmd
        and "pipefail" in cmd
    )
    runner, cmd, how = transfer.plan(TCFG, {}, s, "/a/ds/", hub, "/b")
    assert "tar -C /a/ds/ -cf - ." in cmd and how == "relay"
    runner, cmd, how = transfer.plan(TCFG, {}, x, "/a", x, "/b")
    assert (runner.name, how) == ("x", "same")


def test_relay_really_copies_on_this_machine(tmp_path):
    src = tmp_path / "src" / "ds"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("hello")
    hub = Host("hub", "", reach=["me@hub"])
    cfg = config.Config(hosts=[hub], state_dir=tmp_path)
    other = Host("other", "", reach=["me@other"])  # ssh "" too: both ends local, forced through relay
    _, cmd, how = transfer.plan(cfg, {}, hub, str(src), other, str(tmp_path / "dst"))
    assert how == "relay"
    import subprocess

    subprocess.run(["bash", "-c", cmd], check=True)
    assert (tmp_path / "dst" / "ds" / "sub" / "f.txt").read_text() == "hello"


def test_parse_duration():
    assert ops.parse_duration("90m") == 5400 and ops.parse_duration("1.5d") == 129600
    assert ops.parse_duration("") is None and ops.parse_duration("30") == 30
    with pytest.raises(ValueError):
        ops.parse_duration("8x")


def test_submit_validation(local):
    cfg, d = local
    with pytest.raises(ValueError, match="命令是空的"):
        ops.submit(cfg, d.con, {"cmd": " "})
    with pytest.raises(ValueError, match="没有匹配"):
        ops.submit(cfg, d.con, {"cmd": "true", "hosts": ["nope"]})
    with pytest.raises(ValueError, match="不存在"):
        ops.submit(cfg, d.con, {"cmd": "true", "after": [999]})
    jid = ops.submit(cfg, d.con, {"cmd": "true", "gpus": 0, "time_limit": "2h", "hosts": "me"})
    assert db.get(d.con, jid)["time_limit"] == 7200


def test_time_limit_stops_job(local):
    cfg, d = local
    jid = db.insert(d.con, name="long", cmd="sleep 300", gpus=0, time_limit=1)
    wait_until(d, lambda: db.get(d.con, jid)["state"] in db.FINAL, timeout=60)
    j = db.get(d.con, jid)
    assert j["state"] == "failed" and "超过时限" in j["reason"]


def test_transfer_job_end_to_end(local, tmp_path):
    cfg, d = local
    (tmp_path / "a" / "ds").mkdir(parents=True)
    (tmp_path / "a" / "ds" / "x.bin").write_bytes(b"\0" * 1000)
    jid, how = ops.submit_transfer(cfg, d.con, f"me:{tmp_path}/a/ds", f"me:{tmp_path}/b/", {})
    assert how == "same"
    wait_until(d, lambda: db.get(d.con, jid)["state"] in db.FINAL, timeout=60)
    assert db.get(d.con, jid)["state"] == "done", db.get(d.con, jid)["tail"]
    assert (tmp_path / "b" / "ds" / "x.bin").stat().st_size == 1000


def test_history_aggregates(local):
    from gpuq import web

    cfg, d = local
    now = int(time.time() // 60) * 60
    rows = [(now - 120, "h", 8, 8, 0, 0, 0, 0.0), (now - 60, "h", 8, 0, 4, 4, 0, 90.0)]
    d.con.executemany("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)", rows)
    h = web.history(d.con, 1)["hosts"]["h"]
    assert abs(h["idle_h"] - 8 / 60) < 1e-9 and abs(h["ours_h"] - 4 / 60) < 1e-9 and h["buckets"]


def test_web_submit_needs_token(local, tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request

    from gpuq import web

    cfg, d = local
    cfg.web_host, cfg.web_port = "127.0.0.1", 0
    (cfg.state_dir / "token").write_text("sekret\n")
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(cfg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/submit"

    def post(body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    assert post({"token": "x", "cmd": "true"})[0] == 403
    code, res = post({"token": "sekret", "cmd": "true", "gpus": 0, "hosts": "me", "after": ""})
    assert code == 200 and db.get(d.con, res["id"])["cmd"] == "true"
    code, res = post({"token": "sekret", "kind": "transfer", "src": "me:rel", "dst": "me:/x"})
    assert code == 400 and "绝对路径" in res["error"]
    srv.shutdown()


def test_multi_gpu_only_on_linked_gpus(local):
    """GPUs of different machines are never pooled; multi-node only inside an IB group."""
    cfg, d = local
    hosts = [Host("a1", "x", ["a100"], group="g"), Host("a2", "x", ["a100"], group="g"), Host("a3", "x", ["a100"], group="h"),
             Host("g1", "x", ["4090"]), Host("g2", "x", ["4090"])]  # fmt: skip
    cfg.hosts = hosts
    db.kv_set(d.con, "hosts", {"hosts": [{"name": h.name, "up": True, "gpus": [{}] * (8 if h.name[0] == "a" else 1)}
                                         for h in hosts]})  # fmt: skip
    with pytest.raises(ValueError, match="同一台机器里互联的卡"):
        ops.submit(cfg, d.con, {"cmd": "x", "gpus": 2, "hosts": ["4090"]})  # two 4090s are two machines
    with pytest.raises(ValueError, match="同一组的机器"):
        ops.submit(cfg, d.con, {"cmd": "x", "gpus": 8, "nodes": 2, "hosts": ["a1", "a3"]})  # different groups
    assert ops.submit(cfg, d.con, {"cmd": "x", "gpus": 8, "hosts": ["a100"]})
    assert ops.submit(cfg, d.con, {"cmd": "x", "gpus": 8, "nodes": 2, "hosts": ["a100"]})
    # and the placer never mixes groups either
    snaps = {h.name: snap(8 if h.name[0] == "a" else 1) for h in hosts}
    view = sched.build_view(hosts, snaps, [])
    snaps_busy = dict(snaps, a2=snap(8, used=[30 * G] * 8))
    view = sched.build_view(hosts, snaps_busy, [])
    starts, why = sched.plan([job(1, gpus=8, nodes=2)], lambda j: hosts, view)
    assert not starts  # a1 is free and a3 is free, but they are not linked
    starts, _ = sched.plan([job(2, gpus=2)], lambda j: [h for h in hosts if h.name.startswith("g")],
                           sched.build_view(hosts, snaps, []))  # fmt: skip
    assert not starts  # g1 + g2 would be 2 GPUs, but on two machines
