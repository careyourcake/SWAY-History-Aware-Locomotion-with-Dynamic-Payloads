# B2 MJX 动态负载控制

基于 JAX、MuJoCo MJX 与 Brax PPO 的 Unitree B2 动态负载研究工程。模型使用无碰撞二维摆表示流体晃动与悬吊物的低阶周期性扰动；它不是 CFD 或真实自由液面的等价模拟。

策略输出 12 维关节位置偏移，50 Hz 策略外接 500 Hz PD 控制。动态摆额外包含两个内部力矩执行器，用于每个物理子步计算线性、三次弹簧和阻尼力矩，策略无法控制或直接观测摆角。

## 安装

需要 Python 3.11–3.13。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plots]'
```

Linux NVIDIA CUDA 12 训练服务器：

```bash
pip install -e '.[dev,plots,cuda12]'
python -c "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
```

## 四种方法

当前 `B0–B3` 是本项目自定义的环境诊断和内部参考，不是已发表方法 baseline，也不应被写成 SOTA 对比。它们只能用于验证模型、奖励、动作映射和统计是否可行。正式论文 baseline 以 `configs/baselines.yaml` 为准：每个外部方法必须记录来源、观测权限、实现状态和复现实验后才能进入结果表。旧 `runs/gpu_matrix` 仅作为失败诊断，不纳入主结果。

诊断配置位于 `configs/diagnostics.yaml` 及其引用的四份 YAML。先运行 B0：

```bash
python -m b2_mjx.train --config configs/b0_sanity.yaml --run-dir runs/diagnostics/b0_sanity/seed_0
python -m b2_mjx.evaluate --checkpoint runs/diagnostics/b0_sanity/seed_0 --scenario unloaded --episodes 100
python scripts/check_stage.py runs/diagnostics/b0_sanity/seed_0/evaluation/metrics.csv
```

只有门禁输出 `PASS` 才进入下一阶段。训练结束会在 run 目录写入 `best_checkpoint.json`；评估命令可直接接收 run 目录，并自动载入该最佳 checkpoint，而非最后一个 checkpoint。

服务器上可运行 `bash scripts/run_diagnostics.sh B0 0` 做诊断；脚本失败只说明自定义环境尚未通过可行性检查，不能被解释为文献 baseline 失败。不要用 B0–B3 结果宣称优于已有方法。正式方法比较需要先实现或严谨复现 `configs/baselines.yaml` 中的文献方法，再运行内部消融。

## 能力门控课程

正式课程由 `configs/curriculum.yaml` 定义，按 C0 locomotion、C1 light swing、C2 target swing、C3 robust swing 递增难度。每阶段调用 Brax `train()` 训练 25M 步，四阶段合计 100M，与无课程对照一致；manifest 同时记录累计 `25M/50M/75M/100M`。每阶段使用上一阶段的最佳 checkpoint，并以固定 0.8 m/s、100 个 episode 离线评估：10 m 成功率至少 80%、平均速度 RMSE 不超过 0.45 m/s、跌倒率不超过 20%。成功到达 10 m 的提前结束不会再受 episode length 门槛影响。

```bash
bash scripts/run_curriculum.sh mlp_dynamic 0 runs/internal_curriculum
bash scripts/run_curriculum.sh stack5_dynamic 0 runs/internal_curriculum
bash scripts/run_curriculum.sh gru_dynamic 0 runs/internal_curriculum
```

先只运行 seed 0。确认三个方法均完成 C0-C3 且 `curriculum_manifest.json` 状态为 `complete` 后，再运行 seed 1 和 2。runner 可重复执行：已通过阶段会跳过，中断阶段会从该阶段最新 checkpoint 恢复，门禁失败则记录结果并立即停止。每个 manifest 保存阶段难度、来源 checkpoint、最佳 checkpoint、评估指标、pass/fail 和训练退化检测结果。

无课程对照继续使用 `scripts/run_ablation.sh`，但 MLP/stack5/GRU 只属于内部结构消融，输出目录默认是 `runs/internal_ablation`。正式论文表格必须将它们与文献复现方法、privileged upper bound 和明确的实现/观测权限说明分开；奖励修复和终止条件仅作为实现细节。

当前已登记但尚未实现的文献方法包括 Online Learning of Unknown Dynamics、Unknown Payload Adaptive Control、MULE 和 Adaptive MPC。它们必须逐一复现或明确标记为 unavailable，不能用自定义 B0–B3 结果替代。

## PDB-PRC prototype

The `pdb_prc` policy is an experimental phase/disturbance-belief gated residual actor. A GRU history encoder produces a latent belief and uncertainty scalar; nominal and residual action heads are combined with reduced residual authority under high uncertainty. The checkpoint also contains disturbance-prediction and phase heads for the next auxiliary-loss trainer. The stock Brax PPO loop currently optimizes only the action distribution, so this prototype is an interface scaffold and must not be reported as the final PDB-PRC method until the auxiliary losses are implemented and validated.

```bash
python -m b2_mjx.train --config configs/pdb_prc_smoke.yaml --run-dir runs/pdb_prc_smoke
```

- `gru_dynamic`：50 个、间隔两个策略步采样的历史观测，约覆盖 2 秒；GRU(64) 后接 MLP(128, 64)。
- `mlp_dynamic`：相同动态环境，只使用当前帧。
- `stack5_dynamic`：最近 5 帧直接拼接给 MLP。
- `mlp_static`：把随机负载合并进基座质量、质心和惯量，不使用摆动。

Actor 永远看不到质量、摆长、摆角或摆速。非对称 Critic 使用 63 维特权观测。

## 本地冒烟测试

```bash
pytest
scripts/run_cpu_smoke.sh runs/smoke_matrix
scripts/run_curriculum_cpu_smoke.sh runs/curriculum_smoke
```

CPU 配置只有 2 个并行环境和一次极小 PPO 更新，只验证编译、反向传播、checkpoint 与流水线，不用于判断是否学会步态。

## GPU 正式训练

单个任务可覆盖方法和种子：

```bash
python -m b2_mjx.train --config configs/ppo_b2_payload.yaml \
  --method gru_dynamic --seed 0 --run-dir runs/gru_dynamic/seed_0
```

四方法 × 三种子的矩阵：

```bash
scripts/run_gpu_matrix.sh configs/ppo_b2_payload.yaml runs/gpu_matrix
```

脚本启动前强制检查 JAX GPU 后端，不允许静默回退 CPU。若使用 Slurm，可把上述单任务命令放入作业模板，方法与种子通过数组参数传入。

每个运行目录保存 `config.yaml`、`metadata.json`、训练指标和 checkpoint，可用 `--resume` 从指定 checkpoint 恢复。

## 评估与汇总

```bash
python -m b2_mjx.evaluate --checkpoint runs/gru_dynamic/seed_0/checkpoints/STEP --episodes 100
python -m b2_mjx.aggregate runs/*/seed_*/evaluation/metrics.csv --output-dir results
```

默认扫描 1–5 kg 和 0.1/0.2/0.5 m 摆长，记录 10 m 成功率、距离、速度与姿态误差、机械能量、CoT、峰值力矩和推理耗时。汇总程序输出均值、标准差、95% bootstrap 置信区间以及成功率曲线。

机械运输成本定义为：

```text
CoT = sum(|torque * joint_velocity| * dt) / (total_mass * gravity * distance)
```

成功率、原始能量和 CoT 必须同时报告，避免失败早停形成虚假的低能耗结论。

## 研究边界

实验不预设 GRU 在 5 kg 下必然成功，也不预设节能百分比。若要声称 latent 明确编码晃动相位，需要额外完成冻结 latent 的摆角/摆速线性探针；否则只应声称历史信息改善了任务表现。

机器人 mesh 和基础 MJCF 来自 Unitree `unitree_mujoco` 提交 `ae6a8403e272733e9996ef59990880330496177f`，许可证见 `b2_mjx/assets/unitree_b2/UNITREE_LICENSE`。
