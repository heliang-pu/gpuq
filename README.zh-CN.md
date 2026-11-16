# gpuq

[English](README.md) | **中文**

给几台用 ssh 能连上的机器用的小型 GPU 任务队列。提交一条命令，`gq` 自动挑机器和卡，排队、依赖、等文件、失败重试、
多机训练和数据传输都由它处理。一个守护进程、一个 SQLite 文件、一个网页，只用 Python 标准库。

适合实验室规模的机器池（几台多卡服务器加几台工作站）：用 Slurm 太重，靠表格登记又不够。
远端机器除了 bash、base64 和 nvidia-smi 之外不需要装任何东西。

## 功能

- **带优先级和预约的队列。** 大任务暂时放不下时会预约它的候选机器，小任务不能一直插队把它饿死；小任务仍然可以去别的机器。
- **整卡任务和共享任务。** `-g 8` 占八张空闲的卡；`-g 1 -m 12` 在一张卡上占 12 GB，可以和别的进程同卡。
- **尊重已经在跑的东西。** 不是通过队列启动的进程一律算占用。从不抢占正在运行的任务。
- **依赖。** 等另一个任务成功后再跑（`-a ID`）、等它结束不管成败（`--after-any`）、等某台机器上出现某个文件（`-w 机器:/路径`）。
- **多机任务。** 只在配置好的同一组机器内进行，自动导出 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 和该组的环境变量。
- **传输也是任务。** `gq cp 机器A:/路径 机器B:/目录` 把一次 rsync 放进队列，按"谁能 ssh 到谁"自动选择推、拉或经调度机中转，
  并且可以作为训练任务的依赖。
- **提醒。** 通过你配置的任意命令发出：任务完成或失败、整台机器空闲太久、运行中的任务利用率接近 0、机器掉线。
- **网页。** 显示队列、每张卡的状态、过去 24 小时的使用情况，并可以提交任务和传输。

## 快速开始

调度机上需要 Python 3.11 以上。不用安装，克隆下来直接运行。

```bash
git clone https://github.com/heliang-pu/gpuq && cd gpuq
cp hosts.example.toml hosts.toml     # 写上你自己的机器；hosts.toml 不进仓库
./gq daemon                          # 调度循环 + 网页（端口 8796）；常驻运行见 gq.service
```

另开一个终端：

```bash
./gq hosts                                              # 每台机器每张卡：显存、利用率、谁在用
./gq submit -n finetune -g 8 -H a100 -C /data/myproject -- bash launch.sh
./gq ls                                                 # 队列
./gq log 1 -f
```

配置文件的位置也可以用 `GQ_CONFIG=/路径/hosts.toml` 指定。

## 命令

```bash
# 单机：八张整卡，任意一台带 a100 标签的机器；-C 指定远端工作目录
gq submit -n finetune -g 8 -H a100 -C /data/myproject -- bash launch.sh

# 双机 16 卡：只会落在同一组内；NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT 自动导出
gq submit -n two_node -g 8 -N 2 -H a100 -- bash launch.sh '$NODE_RANK'

# 依赖与等待
gq submit -n export -g 0 -a 12 -- python export.py          # 12 号成功后才跑；-g 0 是纯 CPU 任务
gq submit -n eval -a 12 --after-any -- bash eval.sh         # 12 号结束就跑，不管成败
gq submit -w node2:/data/dataset/x.ready -g 8 -H node2 -- bash train.sh

# 共享一张卡（只占 12 GB）
gq submit -n feat -g 1 -m 12 -H ws0 -- python extract.py

gq ls                  # 活跃任务和最近 24 小时结束的任务；-a 显示全部
gq show 12             # 机器、卡、进程号、远端日志路径、日志末尾
gq log 12 [-f] [-r 1]
gq cancel 12           # 排队中：直接取消；运行中：先 TERM 整个进程组，30 秒后 KILL
gq hold 12 / gq release 12 / gq prio 12 5 / gq resubmit 12
gq hosts               # 每张卡的状态；队列外的进程会标出来
```

其他参数：`-p` 优先级（越大越先）、`-t 8h` 时限、`--retry N` 失败重跑、`-e K=V` 环境变量、`--hold` 提交后先暂缓、
`--no-cvd` 不设置 `CUDA_VISIBLE_DEVICES`（自己读 `$GQ_GPUS`）。

任务里能用的环境变量：`CUDA_VISIBLE_DEVICES`（同时设了 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，编号和 nvidia-smi 一致）、
`GQ_JOB_ID`、`GQ_JOB_NAME`、`GQ_ATTEMPT`、`GQ_HOST`、`GQ_GPUS`、`GQ_NUM_GPUS`；多机任务另有 `NNODES`、`NODE_RANK`、
`MASTER_ADDR`、`MASTER_PORT` 和 `GQ_NODES`。

每个任务在远端的目录是 `<jobdir>/j<任务号>_a<第几次>_r<rank>/`，里面有 `run.sh`（包装脚本）、`cmd.sh`（你的命令）、
`log`、`pid` 和 `exit_code`。任务用 `setsid` 启动，重启守护进程不影响它们，重启后按进程号和退出码接着跟踪。

## 传输数据

```bash
gq cp ws0:/home/me/data/clean_v1 node3:/data/dataset/          # node3 上得到 /data/dataset/clean_v1/
gq cp ws0:/home/me/data/clean_v1/ node3:/data/dataset/v1       # 末尾带斜杠：只传里面的内容
id=$(gq cp -q ws0:/home/me/ds node3:/data/dataset/) && gq submit -a $id -g 8 -H node3 -- bash train.sh
gq cp -a 12 node4:/data/train_out/run_x/ ws1:/data/ckpt/run_x  # 12 号成功后把结果拉回来
```

可选参数：`--delete`、`--bwlimit 50m`、`-a`、`-p`、`-t`。rsync 断了会续传，传输失败默认重试两次。

传输怎么走由"谁能 ssh 到谁"决定（`gq net` 查看，`gq net --probe` 立即探测）：

| 情况 | 走法 |
|---|---|
| 源机能连上目标机 | 在源机上用 rsync 推过去 |
| 只有目标机能连上源机 | 在目标机上用 rsync 拉过来 |
| 两边互相都连不上 | 经调度机用 tar 中转 |
| 同一台机器内 | 本地 rsync |

`gq net --authorize A B` 把 A 的公钥装到 B 上，之后两台机器就能直接互传。
文件属主不保留，落地后属于接收方的账号。

## 分配规则

- **整卡任务**只用满足两个条件的卡：没有队列任务占着，且显存占用低于 `idle_mib`。
- **共享任务**（`-m GB`）的条件是：这张卡的 `总显存 - 已用 - 已承诺给队列任务的显存 - headroom_gb >= GB`。
- **顺序**：先按优先级，再按提交顺序；放不下的任务会预约机器。
- **不同机器的卡不会拼成一个单机任务。** 多机任务只在同一组内。
- **单机任务**去"空闲卡最少、但仍然够用"的那台机器，把整台空机留给大任务。
- **`opt_in` 的机器**（比如有人同时在用的工作站）只接受点名要它、并且带 `-m` 的任务。
- 机器连不上时不再往它上面派任务；已经在它上面跑的任务等它恢复后接着跟踪。

## 配置

全部在 `hosts.toml` 里，`hosts.example.toml` 里每一项都有说明。每台机器可以配置：ssh 别名、标签、组、任务目录、
别的机器怎么连它（用于传输），以及可选的 `opt_in`、`share`、`idle_mib`、`headroom_gb`、`max_gpus`、
`compute = false`（只当传输端点，比如 NAS）。

`notify_cmd` 在每次提醒时在调度机上执行，能拿到 `GQ_EVENT` 和 `GQ_TEXT`（任务类提醒另有 `GQ_JOB_ID`、`GQ_NAME`、
`GQ_STATE`、`GQ_EXIT`、`GQ_HOSTS`），所以任何有命令行客户端的聊天或推送服务都能接。把某个阈值设为 0 就关掉那一类提醒。

## 网页和口令

网页（`web_port`，默认 8796）对所有能访问到它的人显示队列、显卡和使用历史，并允许取消、暂缓、放行、调优先级和重跑。
**提交任务和传输需要口令**，在调度机上执行 `gq token` 查看。没有用户管理，请把网页放在 VPN 或防火墙后面。

## 测试

```bash
PYTHONPATH=. python3 -m pytest -q tests     # 不需要 GPU，也不需要 ssh
```

## 限制

- 只有一个调度进程，没有高可用；队列是本地的一个 SQLite 文件。
- 不抢占，不做公平份额统计，没有按用户的配额。
- 网页和命令行的提示信息是中文的。
