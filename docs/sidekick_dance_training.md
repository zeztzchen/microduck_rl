# 侧踢舞步训练

训练原理、61 维观测、14 维动作和仿真到真机的差距，见
[Microduck 强化学习训练与仿真到真机迁移](rl_training_and_sim2real.md)。

## 共享仓库与本地推理接入

服务器与本地统一使用 <https://github.com/zeztzchen/microduck_rl> 的 `develop`
分支；官方项目 <https://github.com/pollen-robotics/microduck_rl> 保留为 `upstream`。
首次在本地获取代码和测试模型：

```bash
git clone --branch develop https://github.com/zeztzchen/microduck_rl.git
cd microduck_rl
```

已有本地 checkout 时，先检查 `git status`，保存本地 agent 的修改；确认 `origin`
指向共享仓库后，在干净的 `develop` 上执行 `git pull --ff-only origin develop`。
两端编辑前先同步；功能改动在各自分支提交，合入 `develop` 后另一端再拉取，
不要通过强推覆盖对方提交。

最新共享版本为训练中的 **model_6000**，直接随 Git 保存：

- `policies/sidekick_reference_v2/model_6000/model_6000.pt`：原始训练 checkpoint，4843445 字节。
- 同目录 `policy.onnx`：通过官方 `scripts/export.py` 从该 checkpoint 导出，包含观测归一化。
- 同目录 `metadata.json`：checkpoint 与 ONNX 的 SHA256、训练来源、完整部署接口。

之前的 **model_3000** 保留用于对照：

- `policies/sidekick_reference_v2/model_3000/policy.onnx`：793813 字节，包含观测归一化。
- 同目录 `metadata.json`：模型哈希、训练来源、完整观测布局、关节顺序、HOME、BAM 参数。
- 模型 SHA256：`573a10dc28711f30f48fdc71451843587cea708114f3cedc1163af38fbcebeb2`。

新测试版本使用新的目录并明确加入 `.gitignore` 例外，保留旧版本便于比较。
日常训练的 `logs/`、`artifacts/`、`.pt`、虚拟环境不进入 Git；
只有上述显式允许的选定 checkpoint 随版本发布。
仓库的 `uv.lock` 使用公共包源；服务器本机镜像地址的重写不随任务代码提交。

### 本地推理接口（已接入）

复用已有 macOS uv 项目环境 `.venv` 和 `mjpython`，读取上述 `metadata.json`。
独立推理需要 MuJoCo、CPU ONNX Runtime、NumPy 和 BAM，不需要启动训练栈。
训练对应版本为 Python 3.12、MuJoCo 3.10.0、ONNX Runtime 1.24.4；
BAM 来自 `Rhoban/bam` 的 `mjlab_frictionloss` 分支，提交
`62bd8ce12154340be97e06f7f41a0ca8f116d967`。
本地 viewer 不沿用服务器的 `MUJOCO_GL=disable` 或 `DISPLAY=:99`。

`scripts/infer_policy.py --dance` 已接入舞蹈模式，自动读取 ONNX 同目录的
`metadata.json`（也可通过 `--dance-metadata` 指定），检查模型 SHA256、
输入输出、关节顺序、HOME、控制周期和 BAM 参数，复用原有观测构造：

1. 每 0.02 秒从仿真获取真实状态，构造 `float32 [1,61]` 的 `obs`。
   头/身体命令为零，`obs[3:6]` 使用单位 projected gravity；
   现有命令行入口默认使用 projected gravity，不加 `--raw-accelerometer`。
   使用 `--new-cmd-obs` 对应的 13 维命令块，外部不重复归一化。
2. 每次推理前写入 `obs[48:51] = [cos(2*pi*t/1.2), sin(2*pi*t/1.2), 0]`。
   `t` 是开始舞蹈后的仿真时间，暂停时不推进。正半周期右腿、负半周期左腿；
   停止舞蹈时只将 13 维命令清零，继续闭环执行策略。
   避免 walking 的速度输入或自动策略切换覆盖舞蹈命令。
3. 输出 `actions [1,14]`，执行 `q_target = HOME + actions`，
   上一动作观测保存上一轮原始网络输出，初始为零。
   使用 `scene.xml`（包含 `robot_groundcontact.xml`）和配套资产，
   从 `STAND` 初始化，BAM XL330/M6 每 0.005 秒更新，4 个物理步对应一次推理。
   HOME、动作顺序、电压和刚度按 metadata，不新增动作滤波。
4. 提供实际可执行的 macOS 启动命令，以及舞蹈启停、暂停、重置按键；
   保留现有行走、坐下、翻滚功能。
5. 先检查输入输出形状、有限值、相位推进，再观察抬脚、支撑脚接触、滑动和跌倒。
   记录具体模型版本及观察结果，不能以接口通过代替动作质量评估。

可参考 `scripts/eval_sidekick_reference.py` 的节拍注入和 BAM 执行循环。
该脚本现在复用同一个舞蹈适配器，仅在自己的进程内关闭渲染。
仅下载模型并执行 `--walking policy.onnx --new-cmd-obs` 不会自动开始舞蹈。

### 本机启动与操作

从项目根目录执行，使用已有 uv 环境，不重新同步训练依赖、不需要 CUDA：

```bash
PYTHON_LIBDIR=$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

env -u MUJOCO_GL -u DISPLAY \
  DYLD_FALLBACK_LIBRARY_PATH="$PYTHON_LIBDIR:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}" \
  uv run --no-sync mjpython scripts/infer_policy.py \
    --walking alpha_walking.onnx \
    --sitstand alpha_sitstand.onnx \
    --roulade roulade.onnx \
    --dance policies/sidekick_reference_v2/model_6000/policy.onnx \
    --new-cmd-obs
```

前三个 ONNX 是已有本地文件，没有这些文件时删除对应参数即可；仅保留
`--dance ... --new-cmd-obs` 也能运行。macOS 不能用普通 `python` 启动 passive
viewer；`DYLD_FALLBACK_LIBRARY_PATH` 用于找到 uv Python 的动态库。
启动时模型从 `STAND` 进入舞蹈策略自身的 idle，等待站稳后开始。

**所有按键均在启动程序的终端输入，不是在 MuJoCo 查看器窗口输入。**

| 按键 | 行为 |
| --- | --- |
| D | 开始舞蹈，phase 从 0 开始；再次按下进入同一舞蹈模型的 idle |
| 空格 | 舞蹈命令清零，保持同一模型闭环，不是暂停推理 |
| X | 显式退出舞蹈模型，交回零命令 standing/walking；只有 sitstand 时交回其站立状态 |
| T | 暂停/恢复推理和节拍；物理仍继续，最后舵机目标保持，可能失衡 |
| 0（数字零） | 重置 STAND、速度、外力、BAM 状态、动作历史和节拍，进入 dance idle 并恢复推理 |
| Y / R | 坐站 / 翻滚；先按 X 退出舞蹈模型，且需要加载对应 ONNX |
| Q | 退出 |

D/空格停止后，自动 walking/standing 切换、速度键、头部/身体命令仍不能
抢占舞蹈模型；X 才释放控制权。只有舞蹈模型时 X 给出提示，不切换到不存在的策略。
T 不是安全的平衡保持功能；需要停止动作但保留反馈时使用 D/空格。
默认周期来自 metadata；指定 `--dance-period` 时必须与该模型一致。
没有 sidecar 的旧模型保留 1.2 秒默认约定，并在启动时明确提示未校验元数据。

CPU 接口、模式切换、BAM 和 viewer 启动检查：

```bash
uv run --no-sync --with pytest python -m pytest \
  tests/test_infer_policy_dance.py tests/test_infer_policy_bam.py \
  tests/test_infer_policy_viewer.py -q

uv run --no-sync python scripts/eval_sidekick_reference.py \
  --onnx-file policies/sidekick_reference_v2/model_6000/policy.onnx \
  --trials 3 --seconds 8 --output /tmp/sidekick-model6000-dance.json
```

评估命令加 `--idle` 可检查同模型的零命令站立。测试包括模型哈希、8 个相位与
idle 的有限输出、单位重力、原始动作历史、仿真时间推进/暂停/重置、会话名称、
命令互斥和坐站/翻滚回归；这些测试不代表抬脚质量或真机部署已经合格。

### model_6000 的服务器无渲染验证（2026-10-16）

原始 checkpoint 哈希与训练目录一致；官方导出包含观测归一化，
`float32 [1,61] -> [1,14]`、8 个相位与 idle 有限输出检查通过。
使用刚同步的舞蹈适配器完成 Linux CPU MuJoCo/BAM 复测：seed=42，
3 次各 8 秒、忽略首秒、关节初始噪声 ±0.01 rad、无 DR、不渲染。
原始统计在 `policies/sidekick_reference_v2/model_6000/evaluation.json`。

- 跌倒 0/3；踢腿峰值窗口两侧支撑脚接地比例均为 100%。
- 左脚峰值窗口离地比例 73.21%–77.27%，右脚为 100%。
- 左/右峰值窗口平均脚底最低点离地约 3.93–4.15 mm / 7.65–7.71 mm。
- 每次试验相对初始位置的最大水平偏移为 9.02–11.83 cm。

这份快照仍存在左右抬脚不对称和水平漂移；没有对 model_6000 做本地 viewer
或真机验证。下面 model_3000 的 macOS 观察保留为历史对照，不能视为新版本的验收结果。

### model_3000 的 macOS 本地验证（2026-10-16）

模型 SHA256 为上文的 `573a10dc…beb2`，不是之前的 v1 `model_7999`。
CPU 测试套件（推理、BAM、viewer 启动、两种舞蹈配置）共 76 项通过，1 项
Linux 无 DISPLAY 检查在 macOS 跳过；训练 CLI 解析有两条已有的 tuple/list 类型警告。
这次没有启动训练或改动模型权重。

上述无渲染命令：seed=42，3 次各 8 秒，关节初始噪声 ±0.01 rad，
左右起始相位交替，忽略首秒后每次统计 350 个控制步。名义 BAM、平地，无 DR：

| 指标 | 三次试验范围 |
| --- | --- |
| 跌倒 | 0/3；阈值为躯干倾斜 >55° 或高度 <0.065 m |
| 双脚同时接地 | 62.57%–63.71% |
| 左脚踢腿峰值窗口离地比例 | 71.43%–72.73% |
| 右脚踢腿峰值窗口离地比例 | 66.07%–69.70% |
| 两侧峰值窗口的支撑脚接地比例 | 均为 100% |
| 左/右摆动脚峰值窗口平均最低点离地 | 4.25–4.58 mm / 3.13–3.32 mm |
| 平均躯干倾斜 | 1.36°–1.40° |
| 相对重置位置的最大水平偏移 | 9.24–11.68 cm |
| 左/右脚接地时平均接触点滑动速度 | 1.95–2.07 cm/s / 1.86–2.01 cm/s |

峰值窗口定义为 `abs(sin(2*pi*phase)) > 0.85`，离地比例只以该侧峰值窗口的
采样数为分母，不是整个回合的成功率。高度使用整个脚底碰撞网格的最低点，
不是训练脚底 site 的目标高度。滑动速度使用静止地面上各接触点的水平速度，
先在同一脚的接触点间平均，再对该脚接地的采样平均。
全零 idle 的另外 3 次 8 秒测试没有跌倒，双脚接地比例均为 100%。

实际 `mjpython` 窗口已验证 D/空格启停、X 退出、Y 坐站、R 翻滚返回、0 重置和 Q
退出。T 暂停前后节拍值保持一致，但一次约 0.3 秒暂停导致失衡：物理仍在继续，
不能把暂停推理当作站稳功能。数字 0 可恢复初始状态。
独立离屏近景确认左右脚能交替离地，但幅度较小，头部摆动明显；结合上述漂移与
滑动数据，当前结论是“部署接口通过，模型已经能抬脚，但尚未完成理想原地侧踢的
动作质量验收”，不构成真机安全或长期稳定性的保证。

## 参考任务 v2：明确要求抬脚侧踢（2026-10-16 修正）

用户指定的舞蹈参考是
[pezzonovante7/microduck-sidekick-dance](https://github.com/pezzonovante7/microduck-sidekick-dance)，
最初依据提交 `02c5d96704d3ac266acb29fbb29957335ee680bc` 的配置。
任务仍为 **`Mjlab-SideKickReference-Flat-MicroDuck`**，配置为
`src/mjlab_microduck/tasks/microduck_sidekick_reference_env_cfg.py`。
下面原有 `SideKickDance` 任务仍是脚部位置目标，旧训练和模型不被替换。

公开参考提供的是任务代码，不能据此认定已经训练出可见的侧踢。
旧 v1 同时给双脚接地奖励，主奖励仅跟踪 hip_yaw，没有约束摆动脚离地。
对 v1 `model_7999` 的 ONNX 做 CPU MuJoCo/BAM 无渲染复测：3 次各 8 秒、
关节初始噪声 ±0.01 rad、忽略首秒后，双脚同时接地占 99.43%/99.71%/100%；
左右踢腿峰值窗口的摆动脚离地比例均为 0%。这与用户本机观察一致。
原始数据保存在
`artifacts/sidekick_reference/20261016_000325_811cbc/headless_baseline.json`。
这只是名义参数下的部署复测，不是完整 DR 评估。

v2 修改的是奖励目标，旧 ONNX 的行为不会因此改变，需要重新训练：

- **抬脚 → 侧伸 → 收回 → 落地**：脚底 site 最高离静止位置 15 mm，
  两脚横向间距最多增加 15 mm。抬脚脉冲先于侧伸开始，收回先于落地结束；
  相位交界处保留双脚支撑。目标位于世界竖直方向和身体 yaw 坐标系，
  允许躯干横向调整重心。
- **删除左右脚独立接地奖励**。接触奖励根据相位要求相应支撑脚接地；
  抬脚目标超过峰值的 20% 时，要求摆动脚离地。低于 5% 时要求双脚接地，
  中间短过渡不限制摆动脚接触状态。空闲命令要求双脚站立。
- 主脚部轨迹奖励 **8**、接触奖励 **2**、髋角风格奖励 **1** 都受正确接触
  条件约束；贴地转脚、抬错脚或双脚腾空不能获得踢腿阶段的这些奖励。
  另有权重 **2** 的连续脚高引导，帮助离地前探索，但不能代替主奖励。
- hip_yaw 使用平滑的伸出/收回脉冲，峰值仍为两侧 +0.40 rad；
  原作者的 0.55 rad 超出当前关节限位，不扩大限位。髋角从主要目标降为辅助项。
  腿 HOME 权重降为 0.2，且排除 hip_yaw；头颈 HOME 为 0.3，避免抵消踢腿目标。
- 直立和站高合成平衡奖励 **1**，并用于门控正奖励；站高目标 0.1155 m，
  v1 闭环实测约 0.11517–0.11530 m、平均倾斜约 0.51–0.54°。
  HOME 开环角度本身并不保证静态稳定。
- 增加双脚腾空成本 **−2**、漂移成本 **−1**、接触脚滑动成本 **−0.2**；
  保留关节限位、自碰撞保护。早期关闭推力和平滑成本，平滑成本在
  1000/2000/3000 轮逐步到 −0.1/−0.3/−0.5，若技能尚未形成需推迟课程。
- 保留 v1 已修复的显式节拍，仍为 61D：
  `obs[48:51] = [cos(2πt/1.2), sin(2πt/1.2), 0]`，正半周期右腿、负半周期左腿。
  10% 回合全零命令训练 idle，头/身体命令部署时均为零。
- 从当前 walking 配方继承 BAM、DR、编码器偏差、观测噪声和 NaN 防护；
  关节跟踪与 actor 使用相同的编码器视图，兼容 passive/backlash 关节。
  去掉 walking 的奖励和课程，动作仍为 14 维、50 Hz、无额外滤波。

v2 是修正版配方；本仓库附带 model_3000 和 model_6000，分别已完成的验证及局限见前文。
运动学可达、测试通过和 smoke 完成均不代表已经学会稳定侧踢。

在无图形服务器上启动两张卡，每卡 1024 个环境，从零训练：

```bash
uv run scripts/train_sidekick_dance.py \
  --task Mjlab-SideKickReference-Flat-MicroDuck \
  --num-gpus 2 --envs-per-gpu 1024 --iterations 8000
```

入口自动执行每卡 64 环境 × 5 轮 smoke；NaN 终止必须为零，惩罚符号正确，
ONNX 校验通过后才进入正式训练。输出独立保存在
`logs/rsl_rl/sidekick_reference_v2/` 和 `artifacts/sidekick_reference_v2/`，
后者还保留启动时的源码快照和参考提交信息。日志默认 TensorBoard。
每 250 轮保存 checkpoint 并更新运行目录 ONNX，结束后生成验证过的 `train.onnx`。
整个流程 `MUJOCO_GL=disable`，不渲染、不录视频，自动导出包含观测归一化。
旧 v1 的 `logs/rsl_rl/sidekick_reference/`、模型和结果保留。
默认从零训练；不要直接把 v1 的 `model_7999` 当作 v2 的断点续训，
否则旧课程进度和已学会的贴地动作会混入实验。

仅验证构建、5 轮训练和自动导出：

```bash
uv run scripts/train_sidekick_dance.py \
  --task Mjlab-SideKickReference-Flat-MicroDuck --smoke --num-gpus 1
```

TensorBoard：

```bash
uv run tensorboard --logdir logs/rsl_rl/sidekick_reference_v2 --host 127.0.0.1 --port 6006
```

除 `Episode_Reward/sidekick_foot_track` 和跌倒次数外，检查
`Episode_Metrics/dance/` 下的 `both_grounded`、`both_airborne`、`support_lost`、
左右 `*_site_height_m` 和 `lateral_extension_m`。
`airborne_kick` 和 `kick_window` 都以所有控制步为分母；两者相除才是踢腿窗口内
正确单脚支撑的比例，不能把任意一个单独当成功率。空闲回合没有踢腿窗口。
这些指标不乘奖励权重，也不按最大回合时长缩放。

服务器无渲染复测导出的策略（脚底离地高度使用整个碰撞网格的最低点）：

```bash
uv run scripts/eval_sidekick_reference.py \
  --onnx-file artifacts/sidekick_reference_v2/实际运行目录/train.onnx \
  --trials 3 --seconds 8 --output artifacts/sidekick_reference_v2/eval.json
```

这是 CPU MuJoCo/BAM 的名义参数复测，默认直接启动舞蹈、左右起始相位交替。
报告接地占比、峰值窗口的支撑/离地比例、脚底间距、倾斜和跌倒；
`--idle` 可单独检查全零命令站立。最终还要在本地看动作，确认不是拖脚或轻微摆动。

本机推理使用前文 `--dance` 接入的显式节拍；不能直接套参考 README 的零速度命令。
新旧任务 ONNX 的输入/输出和节拍接口相同，学到的动作目标不同。

## 原有脚部位置版本

任务：`Mjlab-SideKickDance-Flat-MicroDuck`。这是一个待训练、待评估的初版
PPO 配方，目标是原地左右交替抬脚并向侧面伸出。通过训练检查不代表已经学会舞步。

## 安装与启动

在有 NVIDIA 驱动的单台机器上执行，A6000 / A100 / H800 使用同一套入口：

```bash
uv sync

# A6000：64 个环境、5 轮训练，然后导出并检查 ONNX。
uv run scripts/train_sidekick_dance.py --smoke --num-gpus 1

# 两张卡验证分布式通信；每卡 64 个环境、5 轮。
uv run scripts/train_sidekick_dance.py --smoke --num-gpus 2

# 单机 8 卡 A100 / H800：先自动执行同样的 smoke，再正式训练。
uv run scripts/train_sidekick_dance.py \
  --num-gpus 8 --envs-per-gpu 1024 --iterations 4000
```

使用 mjlab 的 torchrunx + rsl_rl 分布式 PPO，八张卡共同训练同一个策略。
`--envs-per-gpu` 是**每卡**环境数：上面的设置共 8192 个环境，每轮收集
`8 × 1024 × 24 = 196608` 个 transition。增加卡数也会改变 PPO 的全局 batch，
不保证收敛速度线性提升。首先观察舞步指标和回放，再决定是否增加环境数或训练轮数。
当前脚本支持单机多卡，不负责跨节点启动。

如果调度器设置了 `CUDA_VISIBLE_DEVICES`，`--num-gpus` 从这些可见卡中选取前 N 张。
也可以手动限制，例如：

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run scripts/train_sidekick_dance.py --num-gpus 2
```

默认日志使用本地 TensorBoard，无需 W&B 登录。需要 W&B 时加 `--logger wandb`，
并事先完成自己的账号配置。训练不需要 DISPLAY 或 VNC。
脚本复用 mjlab 的 `TrainConfig` 和 `run_train`，在所有 worker 导入 MuJoCo 前关闭
渲染，因此计算服务器不需要 EGL 图形库。此入口不录制视频；训练后用 `play` 回放。

## 输出与恢复

- checkpoint 和 TensorBoard：`logs/rsl_rl/sidekick_dance/<运行目录>/`
- 启动日志、导出模型、结果索引：`artifacts/sidekick_dance/<时间戳>/`
- `result.json` 记录每阶段 checkpoint、ONNX 路径、GPU 和耗时。
- `smoke.onnx` 仅供管线验证；完成正式训练才会产生 `train.onnx`。
- 每 250 轮保存 checkpoint，中断时可以从最近一次保存继续。
- 现有 runner 保存 checkpoint 时也会更新运行目录中的 ONNX（同名覆盖为最新策略）；
  阶段结束后脚本另行导出并验证 `smoke.onnx` / `train.onnx`。

```bash
uv run scripts/train_sidekick_dance.py --num-gpus 8 \
  --envs-per-gpu 1024 --iterations 2000 \
  --resume-run '这里填写 logs/rsl_rl/sidekick_dance 下的运行目录名' \
  --resume-checkpoint model_1000.pt
```

恢复时 `--iterations` 表示**额外训练轮数**；保持 GPU 数、每卡环境数和配置一致，
方便比较。这里只支持同任务恢复，不把行走 ONNX 当作可继续训练的 checkpoint。
不要为恢复设置 `MICRODUCK_WARM_START=1`，它会重置课程计数。

```bash
uv run tensorboard --logdir logs/rsl_rl/sidekick_dance --port 6006
```

重点看 `Episode_Reward/dance_tracking`、`dance_contact`、episode length 和跌倒次数。
`dance_drift`、`dance_foot_slip`、`dof_pos_limits`、`action_rate_l2` 等成本的加权日志
应当 ≤ 0。平滑成本在第 1000、2000、3000 轮逐渐加入；若边界处舞步退化，先评估
checkpoint，再调整课程节奏。

## 回放与导出

以下回放命令在**有图形环境的本地电脑**执行，不在无 EGL 的训练服务器上执行。
复制 checkpoint 后，使用任务自身的命令发生器回放，它会提供正确节拍：

```bash
MUJOCO_GL=glfw uv run play Mjlab-SideKickDance-Flat-MicroDuck \
  --checkpoint-file 'logs/rsl_rl/sidekick_dance/实际运行目录/model_1000.pt' \
  --num-envs 1 --viewer native
```

脚本在 smoke 和正式训练结束后都会通过仓库的 `scripts/export.py` 导出，保留观察量
归一化，然后检查 ONNX 61 维输入、14 维输出及多个相位的有限输出。
**导出 ONNX 不涉及渲染，不需要 EGL；服务器上的自动导出保持开启。** 中途 checkpoint
也可单独导出：

```bash
MUJOCO_GL=disable uv run scripts/export.py Mjlab-SideKickDance-Flat-MicroDuck \
  --checkpoint-file 'logs/rsl_rl/sidekick_dance/实际运行目录/model_1000.pt' \
  --onnx-file dance.onnx --num-envs 1 --device cuda:0
```

**部署命令约定：**观察量仍为 61D，但 `[48:51]` 槽表示
`[cos(2πt/1.2), sin(2πt/1.2), 0]`，全零表示站立。头部和身体命令部署时为零。
不能直接把这个 ONNX 当普通 walking 模型喂速度指令。现有
`infer_policy.py --walking dance.onnx` 不会自动生成此舞步节拍；ONNX 本地回放使用
前文的 `--dance`，checkpoint 可用 `play` 检查。真机使用前还需要在对应运行时
接入连续相位及启停，并做部署回放。
现有 publish 的 constant-command 合约也不适用于这个相位策略。

## 配方与验证范围

- 周期 1.2 秒，左右交替；相位交界处有双脚支撑时间。
- 脚部目标：最高抬起 12 mm，向外移动 12 mm。奖励在世界竖直方向和身体 yaw
  坐标系下计算，不固定髋关节角，也不限制平衡所需的躯干横向移动。
- 奖励要求支撑脚接触地面，并约束脚高、脚间距、前后错位、身体姿态和接触滑动。
- 10% 训练环境使用全零节拍命令，学习空闲站立；回放默认持续跳舞。
- 继承行走环境的 BAM、编码器偏差、IMU 噪声、延迟、摩擦/质量等随机化以及 NaN
  防护。保留初始随机化范围，不继承行走任务的速度、头部姿态和 CoM 扩大课程。
- 初始不加推力、不加动作平滑成本；后者在技能探索后逐步加入。
- 另注册了匹配 groundcontact 模型的 Backlash 版本；此启动脚本默认训练普通版本。

CPU 测试检查相位/空闲行为、奖励不会偏好拖脚或双脚腾空、成本符号、模型脚部
顺序、真实关节限位内的目标脚位可达性、原行走配置没有被修改以及多卡参数。
脚位可达仅是运动学检查，不代表单脚动态平衡已经学会。

```bash
uv run python -m unittest discover -s tests -p test_sidekick_dance_cfg.py -v
```

物理预检：默认 HOME 关节目标在 BAM 下并不能保持站立，因此没有把它设为严格
姿态目标。当前 `alpha_walking.onnx` 在 CPU MuJoCo/BAM、零命令下的 4 秒回放中，
最后约 2 秒躯干高度 0.1154–0.1159 m，倾斜约 1.08–1.30°；新任务以 0.1155 m
为宽松站立高度目标。正式训练仍需观看视频，确认左右脚实际离地、侧移、回收，
而不是只看总奖励上涨。

2026-10-15 本机验证结果：11 项 CPU 测试通过；A6000 单卡（64 环境）和双卡
（每卡 64 环境）均完成 5 轮 PPO，并成功自动导出带归一化的 61→14 ONNX。
训练日志中的 NaN 终止计数均为 0。结果索引分别位于
`artifacts/sidekick_dance/20261015_190202_9827ab/result.json` 和
`artifacts/sidekick_dance/20261015_191016_08d6ea/result.json`。
双卡日志有 IPv6 回退和 agent 退出清理的 SIGTERM 提示；两个训练 worker 完成，
整个脚本退出码为 0。未进行 8 卡实测或长时间技能训练，测试模型仍会跌倒。
