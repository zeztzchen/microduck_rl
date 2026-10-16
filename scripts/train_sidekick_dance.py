#!/usr/bin/env python3
"""Single-node dance training: GPU preflight -> 5-iteration smoke -> PPO -> ONNX.

Examples:
  uv run scripts/train_sidekick_dance.py --smoke
  uv run scripts/train_sidekick_dance.py --num-gpus 8 --envs-per-gpu 1024
  CUDA_VISIBLE_DEVICES=2,3 uv run scripts/train_sidekick_dance.py --num-gpus 2

Use mjlab's torchrunx launcher, not eight independent training jobs. Environment
count is PER GPU; 8 x 1024 x 24 = 196608 transitions per PPO iteration.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASK = "Mjlab-SideKickDance-Flat-MicroDuck"
EXPERIMENT = "sidekick_dance"
REFERENCE_TASK = "Mjlab-SideKickReference-Flat-MicroDuck"
TASK_SETTINGS = {
    TASK: (EXPERIMENT, 4000),
    REFERENCE_TASK: ("sidekick_reference_v2", 8000),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=tuple(TASK_SETTINGS), default=TASK,
                        help="Foot-position dance (default) or the corrected reference side-kick v2")
    parser.add_argument("--smoke", action="store_true", help="Only 64 envs/GPU, 5 PPO iterations, then export/check ONNX")
    parser.add_argument("--num-gpus", type=int, default=1, help="First N GPUs within CUDA_VISIBLE_DEVICES (single node)")
    parser.add_argument("--envs-per-gpu", type=int, default=None, help="Default: 4096 on one GPU; 1024 on multiple GPUs")
    parser.add_argument("--iterations", type=int, default=None,
                        help="PPO iterations (foot-position: 4000; reference: 8000); additional iterations when resuming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--resume-run", help="Exact directory name under this task's logs/rsl_rl experiment (same task only)")
    parser.add_argument("--resume-checkpoint", default="model_.*.pt", help="Checkpoint filename/regex within --resume-run")
    parser.add_argument("--dry-run", action="store_true", help="Print launch commands without importing CUDA or starting jobs")
    args = parser.parse_args(argv)
    if args.iterations is None:
        args.iterations = TASK_SETTINGS[args.task][1]
    if args.num_gpus < 1 or args.iterations < 1 or (args.envs_per_gpu is not None and args.envs_per_gpu < 1):
        parser.error("GPU, environment and iteration counts must be positive")
    if args.resume_run and (Path(args.resume_run).name != args.resume_run or args.resume_run in (".", "..")):
        parser.error("--resume-run must be an existing run directory name, not a path")
    return args


def train_command(args, tag, smoke):
    count = 64 if smoke else (args.envs_per_gpu or (4096 if args.num_gpus == 1 else 1024))
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "_train", args.task,
        "--gpu-ids", str(list(range(args.num_gpus))),
        "--env.scene.num-envs", str(count),
        "--agent.max-iterations", str(5 if smoke else args.iterations),
        "--agent.num-steps-per-env", "24",
        "--agent.save-interval", "250",
        "--agent.seed", str(args.seed),
        "--agent.logger", "tensorboard" if smoke else args.logger,
        "--agent.run-name", tag,
    ]
    if not smoke and args.resume_run:
        cmd += ["--agent.resume", "True", "--agent.load-run", re.escape(args.resume_run),
                "--agent.load-checkpoint", args.resume_checkpoint]
    return cmd


def train_worker(task_id, cfg, log_dir):
    # Set this BEFORE importing mjlab/MuJoCo, including in fresh DDP workers.
    # CUDA physics and ONNX export need no graphics context or libEGL.
    os.environ["MUJOCO_GL"] = "disable"
    from mjlab.scripts.train import run_train
    run_train(task_id, cfg, log_dir)


def launch_headless_training(task_id):
    """mjlab's TrainConfig/run_train with a renderer-free torchrunx launcher.

    mjlab 1.3's launch_training forces MUJOCO_GL=egl, including in DDP workers.
    Keep its actual environment, PPO and checkpoint code; change only launch
    setup so compute-only servers don't need EGL just to import MuJoCo.
    """
    os.environ["MUJOCO_GL"] = "disable"
    import logging
    from datetime import datetime
    import mjlab
    from mjlab.scripts.train import TrainConfig
    from mjlab.utils.gpu import select_gpus
    import tyro

    cfg = tyro.cli(TrainConfig, default=TrainConfig.from_task(task_id), config=mjlab.TYRO_FLAGS)
    selected, count = select_gpus(cfg.gpu_ids)
    if not selected:
        raise RuntimeError("Dance training requires at least one CUDA GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
    log_dir = ROOT / "logs" / "rsl_rl" / cfg.agent.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + cfg.agent.run_name
    )
    if count == 1:
        train_worker(task_id, cfg, log_dir)
    else:
        import torchrunx
        logging.basicConfig(level=logging.INFO)
        os.environ.setdefault("TORCHRUNX_LOG_DIR", str(log_dir / "torchrunx"))
        print(f"[INFO] Launching one distributed PPO run on {count} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"], workers_per_host=count, backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + (
                "MUJOCO*", "OMP_*", "MKL_*", "LD_LIBRARY_PATH", "WANDB_*",
                "MICRODUCK_*", "MPLCONFIGDIR", "XDG_CACHE_HOME",
            ),
        ).run(train_worker, task_id, cfg, log_dir)


def run(cmd, log_file, child_env):
    print("+ " + shlex.join(cmd), flush=True)
    with log_file.open("w") as output:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
    if code:
        raise RuntimeError(f"Command exited {code}; see {log_file}")


def checkpoint_for_tag(tag, experiment=EXPERIMENT):
    runs = list((ROOT / "logs" / "rsl_rl" / experiment).glob(f"*_{tag}"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run for {tag}, found {runs}")
    checkpoints = [p for p in runs[0].glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
    if not checkpoints:
        raise RuntimeError(f"No saved checkpoint in {runs[0]}")
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))


def validate_smoke_log(log_file):
    """Do not graduate to a long run if numerical guards fired in the smoke."""
    content = log_file.read_text()
    values = re.findall(r"Episode_Termination/nan_state:\s*(\S+)", content)
    if not values or any(float(value) != 0.0 for value in values):
        raise RuntimeError(f"Smoke must log zero NaN terminations; see {log_file}")
    for term in ("dof_pos_limits", "self_collisions", "action_rate_l2", "body_ang_vel",
                 "dance_hop", "dance_drift", "dance_foot_slip"):
        values = re.findall(rf"Episode_Reward/{term}:\s*(\S+)", content)
        if any(not math.isfinite(float(value)) or float(value) > 0 for value in values):
            raise RuntimeError(f"Smoke logged an invalid penalty {term}; see {log_file}")


def validate_onnx(path):
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if not any("normalizer" in item.name for item in model.graph.initializer):
        raise RuntimeError("Export is missing its observation normalizer")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise RuntimeError("Expected the shared 61D observation / 14D action contract")
    rng = np.random.default_rng(42)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        obs = rng.normal(0, 0.01, (1, 61)).astype(np.float32)
        obs[0, 3:6] = [0, 0, -1]
        obs[0, 48:51] = [np.cos(angle), np.sin(angle), 0]
        actions = session.run(None, {session.get_inputs()[0].name: obs})[0]
        if not np.isfinite(actions).all():
            raise RuntimeError("ONNX produced non-finite actions")
    print(f"ONNX passed: normalized 61D -> 14D, finite over eight phases: {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    experiment = TASK_SETTINGS[args.task][0]
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    smoke_tag, train_tag = f"{experiment}_smoke_{stamp}", f"{experiment}_{stamp}"
    if args.dry_run:
        print(shlex.join(train_command(args, smoke_tag, True)))
        if not args.smoke:
            print(shlex.join(train_command(args, train_tag, False)))
        return

    os.environ["MUJOCO_GL"] = "disable"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise SystemExit(f"Need {args.num_gpus} visible CUDA GPUs, found {torch.cuda.device_count()}; check driver and CUDA_VISIBLE_DEVICES")
    if args.resume_run and not (ROOT / "logs" / "rsl_rl" / experiment / args.resume_run).is_dir():
        raise SystemExit(f"Resume directory does not exist: {args.resume_run}")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(args.num_gpus)]
    print(f"GPUs: {gpu_names}", flush=True)
    output_dir = ROOT / "artifacts" / experiment / stamp
    output_dir.mkdir(parents=True)
    child_env = os.environ.copy()
    child_env.setdefault("OMP_NUM_THREADS", "1")
    child_env.setdefault("MKL_NUM_THREADS", "1")
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["MUJOCO_GL"] = "disable"
    results = {"task": args.task, "arguments": vars(args), "gpus": gpu_names, "stages": {}}
    if args.task == REFERENCE_TASK:
        from mjlab_microduck.tasks.microduck_sidekick_reference_env_cfg import (
            DANCE_PERIOD_S, FOOT_LIFT_M, RECIPE_VERSION, SIDE_STEP_M, YAW_TARGET,
        )
        results["reference"] = {
            "repository": "https://github.com/pezzonovante7/microduck-sidekick-dance",
            "commit": "02c5d96704d3ac266acb29fbb29957335ee680bc",
            "recipe_version": RECIPE_VERSION,
            "yaw_target_rad": YAW_TARGET, "period_s": DANCE_PERIOD_S,
            "foot_lift_m": FOOT_LIFT_M, "side_extension_m": SIDE_STEP_M,
            "main_reward_requires_phase_correct_contact": True,
            "command": "obs[48:51] = [cos(phase), sin(phase), 0]; all-zero = idle",
        }
    # Capture uncommitted task sources too; runner git logs omit untracked files.
    import shutil
    snapshot = output_dir / "sources"
    for relative in (
        "scripts/train_sidekick_dance.py", "scripts/eval_sidekick_reference.py",
        "src/mjlab_microduck/tasks/mdp.py",
        "src/mjlab_microduck/tasks/__init__.py",
        "src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py",
        "src/mjlab_microduck/tasks/microduck_sidekick_dance_env_cfg.py",
        "src/mjlab_microduck/tasks/microduck_sidekick_reference_env_cfg.py",
    ):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    started = time.monotonic()
    stages = [("smoke", smoke_tag, True)]
    if not args.smoke:
        stages.append(("train", train_tag, False))
    for name, tag, smoke in stages:
        stage_start = time.monotonic()
        count = 64 if smoke else (args.envs_per_gpu or (4096 if args.num_gpus == 1 else 1024))
        print(f"{name}: {count} envs/GPU, {count * args.num_gpus} total, {24 * count * args.num_gpus} transitions/iteration", flush=True)
        run(train_command(args, tag, smoke), output_dir / f"{name}.log", child_env)
        if smoke:
            validate_smoke_log(output_dir / f"{name}.log")
        checkpoint = checkpoint_for_tag(tag, experiment)
        onnx_path = output_dir / f"{name}.onnx"
        run([sys.executable, "scripts/export.py", args.task,
             "--checkpoint-file", str(checkpoint), "--onnx-file", str(onnx_path),
             "--num-envs", "1", "--device", "cuda:0"],
            output_dir / f"{name}_export.log", child_env)
        validate_onnx(onnx_path)
        results["stages"][name] = {
            "checkpoint": str(checkpoint), "onnx": str(onnx_path),
            "elapsed_s": round(time.monotonic() - stage_start, 2),
        }
        results["elapsed_s"] = round(time.monotonic() - started, 2)
        (output_dir / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Finished. Results: {output_dir / 'result.json'}")
    if args.smoke:
        print("Smoke only: this verifies the pipeline, not a learned dance.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_train":
        del sys.argv[1]
        task_id = sys.argv.pop(1)
        if task_id not in TASK_SETTINGS:
            raise SystemExit(f"Unknown dance task: {task_id}")
        launch_headless_training(task_id)
    else:
        main()
