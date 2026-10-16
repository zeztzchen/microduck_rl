"""CPU checks for the dance reward, real model geometry and launch contract."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sidekick_dance_env_cfg import (
    FOOT_LIFT_M, FOOT_REST_HEIGHT_M, HOME_FOOT_WIDTH_M, SIDE_STEP_M,
    STAND_HEIGHT_M, make_microduck_sidekick_dance_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

ROOT = Path(__file__).resolve().parents[1]


class Scene(dict):
    pass


def reward_env(sine=1.0):
    cmd = torch.tensor([[0.0, sine, 0.0]])
    swing = mdp.sidekick_dance_swing(cmd)
    feet = torch.zeros(1, 2, 3)
    feet[0, :, 1] = torch.tensor([HOME_FOOT_WIDTH_M / 2, -HOME_FOOT_WIDTH_M / 2])
    feet[0, :, 1] += torch.tensor([SIDE_STEP_M, -SIDE_STEP_M]) * swing[0]
    feet[:, :, 2] = FOOT_REST_HEIGHT_M + FOOT_LIFT_M * swing
    data = SimpleNamespace(
        site_pos_w=feet, root_link_pos_w=torch.tensor([[0.0, 0.0, STAND_HEIGHT_M]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]), heading_w=torch.zeros(1),
        root_link_lin_vel_b=torch.zeros(1, 3), root_link_ang_vel_b=torch.zeros(1, 3),
        site_lin_vel_w=torch.zeros(1, 2, 3),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = torch.zeros(1, 3)
    scene.sensors = {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=(swing < 0.1).unsqueeze(-1)))}
    return SimpleNamespace(num_envs=1, scene=scene, command_manager=SimpleNamespace(get_command=lambda _: cmd))


class DanceTests(unittest.TestCase):
    def test_observation_contract_and_base_safeguards(self):
        cfg = make_microduck_sidekick_dance_env_cfg()
        self.assertEqual(list(cfg.observations["actor"].terms), [
            "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
            "command", "head_command", "body_command",
        ])
        self.assertEqual(cfg.decimation * cfg.sim.mujoco.timestep, 0.02)
        self.assertIn("expand_bam_friction_fields", cfg.events)
        self.assertIn("encoder_bias", cfg.events)
        self.assertIn("nan_state", cfg.terminations)
        for name in ("foot_height", "foot_air_time", "foot_contact_forces"):
            self.assertTrue(cfg.observations["critic"].terms[name].func.__name__.endswith("_safe"))
        self.assertNotIn("standing_envs", cfg.curriculum)
        self.assertNotIn("track_linear_velocity", cfg.rewards)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, 0)
        for name in ("dance_drift", "dance_foot_slip", "self_collisions", "dof_pos_limits"):
            self.assertLess(cfg.rewards[name].weight, 0)

    def test_factory_does_not_mutate_walking(self):
        before = deepcopy(make_microduck_velocity_env_cfg())
        make_microduck_sidekick_dance_env_cfg()
        after = make_microduck_velocity_env_cfg()
        self.assertEqual(before.commands["twist"].ranges, after.commands["twist"].ranges)
        self.assertEqual(before.rewards["action_rate_l2"].weight, after.rewards["action_rate_l2"].weight)
        self.assertEqual(before.events["reset_base"].params, after.events["reset_base"].params)

    def test_actual_models_resolve_feet_and_servos_in_order(self):
        for task in ("Mjlab-SideKickDance-Flat-MicroDuck", "Mjlab-SideKickDance-Flat-Backlash-MicroDuck"):
            cfg = load_env_cfg(task)
            model = cfg.scene.entities["robot"].spec_fn().compile()
            names = [model.joint(i).name for i in range(model.njnt)]
            servo = [n for n in names if n and not n.startswith("passive_") and n != "floating_base"]
            # Count actuators by their target names; the root free joint is not a servo.
            self.assertEqual(model.nu, 14)
            self.assertLess(model.geom("left_foot_collision").id, model.geom("right_foot_collision").id)
            self.assertLess(model.site("left_foot").id, model.site("right_foot").id)
            self.assertIn("left_hip_roll", servo)
            self.assertIn("right_hip_roll", servo)

    def test_targets_are_reachable_with_actual_joint_limits(self):
        cfg = make_microduck_sidekick_dance_env_cfg()
        model = cfg.scene.entities["robot"].spec_fn().compile()
        data = mujoco.MjData(model)
        data.qpos[:7] = [0, 0, STAND_HEIGHT_M, 1, 0, 0, 0]
        for i in range(model.njnt):
            name = model.joint(i).name
            for pattern, value in HOME_FRAME.joint_pos.items():
                if re.fullmatch(pattern, name):
                    data.qpos[model.jnt_qposadr[i]] = value
                    break
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.site("left_foot").xpos[1] - data.site("right_foot").xpos[1], HOME_FOOT_WIDTH_M, delta=0.002)
        for side, sign in (("left", 1), ("right", -1)):
            joints = [model.joint(f"{side}_{suffix}").id for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
            qids = model.jnt_qposadr[joints]
            home = data.qpos[qids].copy()
            target = np.array([0.0, sign * (HOME_FOOT_WIDTH_M / 2 + SIDE_STEP_M), FOOT_REST_HEIGHT_M + FOOT_LIFT_M])

            def residual(q):
                data.qpos[qids] = q
                mujoco.mj_forward(model, data)
                return data.site(f"{side}_foot").xpos - target

            fit = least_squares(residual, home, bounds=(model.jnt_range[joints, 0] + 0.01, model.jnt_range[joints, 1] - 0.01),
                                gtol=1e-10, ftol=1e-10, xtol=1e-10)
            self.assertLess(np.linalg.norm(residual(fit.x)), 0.001)

    def test_clock_reset_and_period_are_observable(self):
        env = SimpleNamespace(num_envs=4, device="cpu", step_dt=0.02)
        cmd = mdp.SideKickDanceCommandCfg(resampling_time_range=(8, 8), idle_probability=0, randomize_start_side=False).build(env)
        cmd.reset(None)
        self.assertTrue(torch.allclose(cmd.command[:, :2], torch.tensor([[1.0, 0.0]]).expand(4, -1)))
        cmd.compute(0.3)
        self.assertTrue(torch.allclose(cmd.command[:, 1], torch.ones(4)))
        old = cmd.command[1:].clone()
        cmd.reset(torch.tensor([0]))
        self.assertTrue(torch.allclose(cmd.command[1:], old))
        self.assertAlmostEqual(cmd.command[0, 0].item(), 1)

    def test_idle_and_phase_crossings_have_both_feet_down(self):
        cmd = torch.tensor([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        self.assertTrue(torch.equal(mdp.sidekick_dance_swing(cmd), torch.tensor([[0., 0.], [0., 0.], [0., 0.], [0., 1.], [1., 0.]])))
        clock = mdp.SideKickDanceCommandCfg(resampling_time_range=(8, 8), idle_probability=1).build(SimpleNamespace(num_envs=8, device="cpu"))
        clock.reset(None)
        clock.compute(0.1)
        self.assertEqual(clock.command.count_nonzero().item(), 0)

    def test_correct_step_beats_standing_dragging_and_jumping(self):
        feet_cfg = SimpleNamespace(site_ids=[0, 1])
        for sine in (-1., 1.):
            env = reward_env(sine)
            ideal = mdp.sidekick_dance_tracking(env, feet_cfg).item()
            self.assertAlmostEqual(ideal, 1.0, places=5)
            self.assertAlmostEqual(mdp.sidekick_dance_contact(env).item(), 1.0)
            env.scene["robot"].data.site_pos_w[:, :, 2] = FOOT_REST_HEIGHT_M
            self.assertLess(mdp.sidekick_dance_tracking(env, feet_cfg).item(), ideal / 5)
            env.scene.sensors["feet_ground_contact"].data.found[:] = True
            self.assertEqual(mdp.sidekick_dance_contact(env).item(), 0)
            env.scene.sensors["feet_ground_contact"].data.found[:] = False
            self.assertEqual(mdp.sidekick_dance_tracking(env, feet_cfg).item(), 0)

    def test_no_flop_rewards_and_cost_signs(self):
        env = reward_env()
        env.scene["robot"].data.root_link_pos_w[:, 2] = 0.03
        self.assertEqual(mdp.sidekick_dance_balance(env).item(), 0)
        self.assertEqual(mdp.sidekick_dance_contact(env).item(), 0)
        env.scene["robot"].data.root_link_lin_vel_b[:] = 1
        env.scene["robot"].data.site_lin_vel_w[:] = 1
        self.assertGreater(mdp.sidekick_dance_drift_cost(env).item(), 0)
        self.assertGreater(mdp.sidekick_dance_slip_cost(env, SimpleNamespace(site_ids=[0, 1])).item(), 0)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("dance_launcher", ROOT / "scripts/train_sidekick_dance.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def test_eight_gpu_command_is_one_distributed_run(self):
        args = self.launcher.parse_args(["--num-gpus", "8"])
        cmd = self.launcher.train_command(args, "test", False)
        start = cmd.index("--gpu-ids") + 1
        self.assertEqual(cmd[start], "[0, 1, 2, 3, 4, 5, 6, 7]")
        self.assertEqual(cmd[cmd.index("--env.scene.num-envs") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--agent.num-steps-per-env") + 1], "24")

    def test_smoke_never_inherits_long_run_or_resume(self):
        args = self.launcher.parse_args(["--num-gpus", "2", "--iterations", "9999", "--envs-per-gpu", "8192", "--resume-run", "existing"])
        cmd = self.launcher.train_command(args, "test", True)
        self.assertEqual(cmd[cmd.index("--env.scene.num-envs") + 1], "64")
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], "5")
        self.assertNotIn("--agent.resume", cmd)

    def test_launch_arguments_parse_with_the_installed_mjlab_cli(self):
        import mjlab
        from mjlab.scripts.train import TrainConfig
        import tyro
        args = self.launcher.parse_args(["--num-gpus", "8", "--resume-run", "previous_run"])
        cmd = self.launcher.train_command(args, "test", False)
        self.assertEqual(cmd[3], self.launcher.TASK)
        cfg = tyro.cli(TrainConfig, args=cmd[4:], default=TrainConfig.from_task(cmd[3]), config=mjlab.TYRO_FLAGS)
        self.assertEqual(cfg.gpu_ids, list(range(8)))
        self.assertEqual(cfg.env.scene.num_envs, 1024)
        self.assertTrue(cfg.agent.resume)


if __name__ == "__main__":
    unittest.main()
