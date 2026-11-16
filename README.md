# gpuq

**English** | [中文](README.zh-CN.md)

A small GPU job queue for a handful of machines you reach over ssh. You submit a command; `gq` picks the machine and
the GPUs, and handles queueing, dependencies, waiting for files, retries, multi-node jobs and data transfers.
One daemon, one SQLite file, one web page. Python standard library only.

It is for a lab-sized pool (a few multi-GPU servers and workstations) where Slurm is too much and a shared
spreadsheet is too little. The remote machines need nothing installed beyond bash, base64 and nvidia-smi.

## What it does

- **Queue with priorities and reservations.** A large job that does not fit yet reserves its candidate machines, so
  small jobs cannot starve it; small jobs still backfill other machines.
- **Whole-GPU and shared-GPU jobs.** `-g 8` takes eight idle cards; `-g 1 -m 12` takes 12 GB on a card that other
  processes may also use.
- **Respects what is already running.** Processes that were not started through the queue count as busy. Nothing is
  ever preempted.
- **Dependencies.** Run after another job succeeds (`-a ID`), after it ends either way (`--after-any`), or when a file
  appears on a host (`-w host:/path`).
- **Multi-node jobs** inside a configured group, with `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and the
  group's own environment exported.
- **Transfers as jobs.** `gq cp hostA:/path hostB:/dir` queues an rsync between any two hosts, picks push, pull or a
  relay through the scheduler depending on who can ssh to whom, and can be a dependency of a training job.
- **Alerts** through any command you configure: job done or failed, a whole machine idle for too long, a running job
  stuck near 0 % utilisation, a host offline.
- **Web page** with the queue, per-GPU state, the last 24 hours of usage, and forms to submit jobs and transfers.

## Quick start

Requires Python 3.11+ on the scheduler host. No installation: clone and run.

```bash
git clone https://github.com/heliang-pu/gpuq && cd gpuq
cp hosts.example.toml hosts.toml     # describe your machines; hosts.toml is git-ignored
./gq daemon                          # scheduler loop + web page on :8796 (see gq.service for systemd)
```

In another shell:

```bash
./gq hosts                                              # every GPU of every host: memory, utilisation, who uses it
./gq submit -n finetune -g 8 -H a100 -C /data/myproject -- bash launch.sh
./gq ls                                                 # the queue
./gq log 1 -f
```

The config path can also be given with `GQ_CONFIG=/path/to/hosts.toml`.

## Commands

```bash
# one machine, eight whole GPUs, any host tagged a100; -C sets the remote working directory
gq submit -n finetune -g 8 -H a100 -C /data/myproject -- bash launch.sh

# two nodes, 16 GPUs: only inside one group; NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT are exported
gq submit -n two_node -g 8 -N 2 -H a100 -- bash launch.sh '$NODE_RANK'

# dependencies and waiting
gq submit -n export -g 0 -a 12 -- python export.py          # after job 12 succeeds; -g 0 = CPU only
gq submit -n eval -a 12 --after-any -- bash eval.sh         # after job 12 ends, whatever the result
gq submit -w node2:/data/dataset/x.ready -g 8 -H node2 -- bash train.sh

# share a card (12 GB of it)
gq submit -n feat -g 1 -m 12 -H ws0 -- python extract.py

gq ls                  # active jobs and those that ended in the last 24 h; -a for all
gq show 12             # hosts, GPUs, pids, remote log paths, the end of the log
gq log 12 [-f] [-r 1]
gq cancel 12           # queued: removed; running: TERM to the process group, KILL after 30 s
gq hold 12 / gq release 12 / gq prio 12 5 / gq resubmit 12
gq hosts               # per-GPU state; processes outside the queue are marked
```

Other options: `-p` priority (higher first), `-t 8h` time limit, `--retry N`, `-e K=V`, `--hold`,
`--no-cvd` (do not set `CUDA_VISIBLE_DEVICES`; read `$GQ_GPUS` yourself).

Environment inside a job: `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so indices match nvidia-smi),
`GQ_JOB_ID`, `GQ_JOB_NAME`, `GQ_ATTEMPT`, `GQ_HOST`, `GQ_GPUS`, `GQ_NUM_GPUS`; multi-node jobs also get `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and `GQ_NODES`.

Each job runs in `<jobdir>/j<id>_a<attempt>_r<rank>/` on its host, which holds `run.sh` (the wrapper), `cmd.sh`
(your command), `log`, `pid` and `exit_code`. Jobs are started with `setsid`, so restarting the daemon does not
disturb them; it picks them up again by pid and exit code.

## Transfers

```bash
gq cp ws0:/home/me/data/clean_v1 node3:/data/dataset/          # node3 gets /data/dataset/clean_v1/
gq cp ws0:/home/me/data/clean_v1/ node3:/data/dataset/v1       # trailing slash: only the contents
id=$(gq cp -q ws0:/home/me/ds node3:/data/dataset/) && gq submit -a $id -g 8 -H node3 -- bash train.sh
gq cp -a 12 node4:/data/train_out/run_x/ ws1:/data/ckpt/run_x  # bring results home after job 12
```

Options: `--delete`, `--bwlimit 50m`, `-a`, `-p`, `-t`. An interrupted rsync resumes; a failed transfer is retried twice.

The route follows from who can ssh to whom (`gq net` shows the table, `gq net --probe` measures it now):

| Situation | Route |
|---|---|
| The source reaches the destination | rsync pushed from the source |
| Only the destination reaches the source | rsync pulled from the destination |
| Neither reaches the other | tar relayed through the scheduler host |
| Same machine | local rsync |

`gq net --authorize A B` installs A's public key on B so that later transfers go directly.
File ownership is not preserved; files belong to the receiving account.

## Allocation rules

- **Whole-GPU jobs** use only cards that no queue job holds and whose memory use is below `idle_mib`.
- **Shared jobs** (`-m GB`) need `total - used - memory promised to queue jobs - headroom_gb >= GB` on the card.
- **Order:** priority first, then submission order, with reservations for jobs that do not fit yet.
- **GPUs of different machines are never pooled** into one single-node job. Multi-node jobs stay inside one group.
- **Single-node jobs** go to the machine with the fewest idle cards that still fits them, keeping whole machines free
  for large jobs.
- **`opt_in` hosts** (for example a workstation someone is also using) only take jobs that name them and give `-m`.
- An unreachable host gets no new jobs; jobs already on it are tracked again when it comes back.

## Configuration

Everything is in `hosts.toml`; `hosts.example.toml` documents each key. Per host: ssh alias, tags, group, job
directory, how other hosts reach it for transfers, and optional `opt_in`, `share`, `idle_mib`, `headroom_gb`,
`max_gpus`, `compute = false` (a transfer endpoint only, such as a NAS).

`notify_cmd` runs on the scheduler host for every alert with `GQ_EVENT` and `GQ_TEXT` set (plus `GQ_JOB_ID`,
`GQ_NAME`, `GQ_STATE`, `GQ_EXIT`, `GQ_HOSTS` for jobs), so any chat or push service with a command-line client works.
Set a threshold to 0 to switch that kind of alert off.

## Web page and its password

The page (`web_port`, default 8796) shows the queue, the GPUs and the usage history to anyone who can reach it, and
lets them cancel, hold, release, re-prioritise and resubmit. **Submitting jobs and transfers needs a password**, shown
by `gq token` on the scheduler host. There is no user management; put the page behind a VPN or a firewall.

## Tests

```bash
PYTHONPATH=. python3 -m pytest -q tests     # no GPU and no ssh needed
```

## Limits

- One scheduler, no high availability; the queue is a local SQLite file.
- No preemption, no fair-share accounting, no per-user quotas.
- The web page and the CLI messages are in Chinese.
