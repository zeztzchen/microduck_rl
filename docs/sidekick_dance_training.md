# 侧踢舞步训练

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

本次选定的测试版本直接随 Git 保存：

- `policies/sidekick_reference_v2/model_3000/policy.onnx`：793813 字节，包含观测归一化。
- 同目录 `metadata.json`：模型哈希、训练来源、完整观测布局、关节顺序、HOME、BAM 参数。
- 模型 SHA256：`573a10dc28711f30f48fdc71451843587cea708114f3cedc1163af38fbcebeb2`。

新测试版本使用新的目录并明确加入 `.gitignore` 例外，保留旧版本便于比较。
日常训练的 `logs/`、`artifacts/`、`.pt`、虚拟环境不进入 Git。
仓库的 `uv.lock` 使用公共包源；服务器本机镜像地址的重写不随任务代码提交。

### 本地 agent 的任务

复用已有 macOS `.venv-infer` 和 `mjpython` 启动方式，读取上述 `metadata.json`。
独立推理需要 MuJoCo、CPU ONNX Runtime、NumPy 和 BAM，不需要启动训练栈。
训练对应版本为 Python 3.12、MuJoCo 3.10.0、ONNX Runtime 1.24.4；
BAM 来自 `Rhoban/bam` 的 `mjlab_frictionloss` 分支，提交
`62bd8ce12154340be97e06f7f41a0ca8f116d967`。
本地 viewer 不沿用服务器的 `MUJOCO_GL=disable` 或 `DISPLAY=:99`。

当前 `scripts/infer_policy.py` 尚无专用舞蹈参数。请本地 agent 接入舞蹈模式或
独立 viewer 入口，复用已有 BAM 和观测构造，并将实现提交到这个共享仓库：

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
该脚本用于无渲染评估，接入 viewer 时需要处理它关闭渲染的设置。
仅下载模型并执行 `--walking policy.onnx --new-cmd-obs` 不会自动开始舞蹈。

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

v2 是待训练、待行为评估的修正版。运动学可达、测试通过和 smoke 完成
均不代表已经学会稳定侧踢。

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

本机推理仍需接入前文约定的显式舞蹈节拍；不能直接套参考 README 的零速度命令。
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
`infer_policy.py --walking dance.onnx` 不会自动生成此舞步节拍；本次先用 `play`
检查训练结果。真机使用前还需要在对应运行时接入连续相位及启停，并做部署回放。
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
