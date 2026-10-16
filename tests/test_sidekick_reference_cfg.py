"""Reference recipe: reachable targets, clock/encoder contract and launch gates."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_sidekick_reference_env_cfg import (
    REFERENCE_COMMIT, REFERENCE_YAW_TARGET, YAW_TARGET,
    EXPERIMENT_NAME, FOOT_LIFT_M, SIDE_STEP_M, HOME_FOOT_WIDTH_M,
    FOOT_REST_HEIGHT_M, STAND_Z,
    MicroduckSideKickReferenceRlCfg, make_microduck_sidekick_reference_env_cfg,
)
from mjlab_microduck.tasks.microduck_sidekick_dance_env_cfg import make_microduck_sidekick_dance_env_cfg

ROOT = Path(__file__).resolve().parents[1]
TASK = "Mjlab-SideKickReference-Flat-MicroDuck"


def encoder_env(backlash=False):
    names = ["left_hip_yaw", "right_hip_yaw"]
    ids = [0, 1]
    if backlash:
        names = ["left_hip_yaw", "passive_left_hip_yaw_backlash",
                 "right_hip_yaw", "passive_right_hip_yaw_backlash"]
        ids = [0, 2]
    raw = torch.zeros(3, len(names))
    biased = raw.clone()
    # Correct: right on positive half, left on negative half, neither at idle.
    biased[0, ids[1]] = YAW_TARGET
    biased[1, ids[0]] = YAW_TARGET
    asset = SimpleNamespace(joint_names=names, data=SimpleNamespace(
        joint_pos=raw, joint_pos_biased=biased,
        default_joint_pos=torch.zeros_like(raw), root_link_lin_vel_b=torch.zeros(3, 3),
    ))
    command = torch.tensor([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.]])
    env = SimpleNamespace(scene={"robot": asset},
                          command_manager=SimpleNamespace(get_command=lambda _: command))
    return env, SimpleNamespace(name="robot", joint_ids=ids), command


class ReferenceCfgTests(unittest.TestCase):
    def test_lifted_objective_replaces_independent_contact_rewards(self):
        cfg = make_microduck_sidekick_reference_env_cfg()
        self.assertNotIn("left_support", cfg.rewards)
        self.assertNotIn("right_support", cfg.rewards)
        self.assertIs(cfg.rewards["phase_contact"].func, mdp.sidekick_reference_contact)
        self.assertGreater(cfg.rewards["sidekick_foot_track"].weight,
                           cfg.rewards["sidekick_hip_yaw"].weight)
        self.assertIs(cfg.rewards["sidekick_hip_yaw"].func, mdp.sidekick_reference_hip_yaw_reward)
        self.assertGreater(cfg.rewards["sidekick_clearance"].weight, 0)
        for name in ("dance_hop", "dance_drift", "dance_foot_slip", "body_ang_vel",
                     "self_collisions", "dof_pos_limits"):
            self.assertLess(cfg.rewards[name].weight, 0)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, 0)
        self.assertNotIn("push_robot", cfg.events)
        self.assertIn("dance/airborne_kick", cfg.metrics)
        self.assertIn("dance/kick_window", cfg.metrics)
        self.assertEqual(set(cfg.curriculum), {"action_rate_weight"})
        self.assertEqual(cfg.curriculum["action_rate_weight"].params["weight_stages"], [
            {"step": 0, "weight": 0.}, {"step": 24000, "weight": -.1},
            {"step": 48000, "weight": -.3}, {"step": 72000, "weight": -.5},
        ])
        self.assertEqual(MicroduckSideKickReferenceRlCfg.max_iterations, 8000)
        self.assertEqual(MicroduckSideKickReferenceRlCfg.experiment_name, "sidekick_reference_v2")
        self.assertEqual(len(REFERENCE_COMMIT), 40)

    def test_observation_safeguards_and_old_task_unchanged(self):
        before = deepcopy(make_microduck_sidekick_dance_env_cfg())
        cfg = make_microduck_sidekick_reference_env_cfg()
        after = make_microduck_sidekick_dance_env_cfg()
        self.assertEqual(list(cfg.observations["actor"].terms), [
            "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
            "command", "head_command", "body_command",
        ])
        self.assertEqual(cfg.decimation * cfg.sim.mujoco.timestep, .02)
        self.assertEqual(cfg.commands["twist"].period, 1.2)
        self.assertEqual(cfg.commands["twist"].idle_probability, .1)
        self.assertIn("expand_bam_friction_fields", cfg.events)
        self.assertTrue(cfg.observations["actor"].terms["joint_pos"].params["biased"])
        for key in ("foot_height", "foot_air_time", "foot_contact_forces"):
            self.assertTrue(cfg.observations["critic"].terms[key].func.__name__.endswith("_safe"))
        self.assertEqual(before.rewards["dance_tracking"].weight, after.rewards["dance_tracking"].weight)
        self.assertEqual(before.rewards["action_rate_l2"].weight, after.rewards["action_rate_l2"].weight)
        self.assertNotIn("push_robot", after.events)
        play = make_microduck_sidekick_reference_env_cfg(play=True)
        self.assertEqual(play.commands["twist"].idle_probability, 0)
        self.assertNotIn("push_robot", play.events)

    def test_target_with_encoder_uncertainty_inside_real_joint_limits(self):
        for task in (TASK, "Mjlab-SideKickReference-Flat-Backlash-MicroDuck"):
            cfg = load_env_cfg(task)
            model = cfg.scene.entities["robot"].spec_fn().compile()
            self.assertEqual(model.nu, 14)
            # Worst-case encoder bias plus ±1 degree serial backlash.
            margin = .015 + (np.deg2rad(1.) if "Backlash" in task else 0.)
            for name in ("left_hip_yaw", "right_hip_yaw"):
                low, high = model.joint(name).range
                self.assertLess(high, REFERENCE_YAW_TARGET)
                self.assertLess(YAW_TARGET + margin, high)
                self.assertGreater(-margin, low)
            for side in ("left", "right"):
                self.assertGreaterEqual(model.geom(f"{side}_foot_collision").id, 0)
            if "Backlash" in task:
                self.assertIs(cfg.observations["actor"].terms["joint_pos"].func,
                              mdp.joint_pos_rel_backlash)

    def test_phase_and_idle_use_actor_encoder_view(self):
        env, asset_cfg, command = encoder_env()
        score = mdp.sidekick_reference_hip_yaw_track(env, asset_cfg)
        torch.testing.assert_close(score, torch.ones(3))
        command[0, 1] = -1
        self.assertLess(mdp.sidekick_reference_hip_yaw_track(env, asset_cfg)[0].item(), .01)
        # Hidden episode time must not influence the policy's visible phase.
        command[0, 1] = 1
        env.episode_length_buf = torch.tensor([57, 88, 99])
        torch.testing.assert_close(mdp.sidekick_reference_hip_yaw_track(env, asset_cfg), score)

    def test_interleaved_backlash_and_pose_share_encoder_view(self):
        env, asset_cfg, _ = encoder_env(backlash=True)
        asset = env.scene["robot"]
        asset.data.joint_pos_biased[0, 2] -= .01
        asset.data.joint_pos[0, 3] = .01
        torch.testing.assert_close(mdp.sidekick_reference_hip_yaw_track(env, asset_cfg), torch.ones(3))
        # HOME reward must see the same .4 rad as the actor, not the raw zero.
        expected = (1 + np.exp(-(.4 / .3) ** 2)) / 2
        self.assertAlmostEqual(mdp.sidekick_reference_home_pose(env, asset_cfg)[0].item(), expected, places=6)
        self.assertEqual(mdp.sidekick_reference_home_pose(env, asset_cfg)[2].item(), 1.)

    def test_stay_put_is_positive_reward_not_negative_cost(self):
        env, _, _ = encoder_env()
        env.scene["robot"].data.root_link_lin_vel_b[:, 0] = torch.tensor([0., .08, -.08])
        torch.testing.assert_close(mdp.sidekick_reference_stay_put(env),
                                   torch.tensor([1., np.exp(-1), np.exp(-1)], dtype=torch.float32))

    def test_lift_extension_and_yaw_reachable_without_sole_dragging(self):
        """Actual robot FK/IK, real limits; kinematic reachability, not balance."""
        model = make_microduck_sidekick_reference_env_cfg().scene.entities["robot"].spec_fn().compile()
        data = mujoco.MjData(model)
        data.qpos[:7] = [0, 0, STAND_Z, 1, 0, 0, 0]
        for side, sign in (("left", 1), ("right", -1)):
            joints = [model.joint(f"{side}_{suffix}").id
                      for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
            ids = model.jnt_qposadr[joints]
            geom = model.geom(f"{side}_foot_collision").id
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            low, high = model.jnt_range[joints, 0] + .005, model.jnt_range[joints, 1] - .005
            # Outward target at all stages, and positive reference yaw on both legs.
            for sine in (0., .35, .65, .9, 1.):
                command = torch.tensor([[np.sqrt(1-sine*sine), -sign*sine, 0.]])
                lift, extension = mdp.sidekick_reference_targets(command)
                l, e = float(lift.max()), float(extension.max())
                target = np.array([0., sign * (HOME_FOOT_WIDTH_M/2 + SIDE_STEP_M*e),
                                   FOOT_REST_HEIGHT_M + FOOT_LIFT_M*l])

                def residual(q):
                    data.qpos[ids] = q
                    mujoco.mj_forward(model, data)
                    sole_z = (vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])[:, 2].min()
                    return np.r_[data.site(f"{side}_foot").xpos - target,
                                 .05 * (q[0] - YAW_TARGET*e),
                                 max(0., .006*l - sole_z)]

                best = None
                for initial in (np.clip(data.qpos[ids], low+1e-6, high-1e-6), (low+high)/2):
                    fit = least_squares(residual, initial, bounds=(low, high), max_nfev=250,
                                        gtol=1e-10, ftol=1e-10, xtol=1e-10)
                    if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                        best = fit
                self.assertLess(np.linalg.norm(residual(best.x)), .001, (side, sine, best.fun))


class Scene(dict):
    pass


def kick_env():
    env, hip_cfg, command = encoder_env()
    env.scene = Scene(env.scene)
    env.scene.env_origins = torch.zeros(3, 3)
    env.num_envs, env.device = 3, "cpu"
    lift, extension = mdp.sidekick_reference_targets(command)
    feet = torch.zeros(3, 2, 3)
    feet[:, :, 1] = (HOME_FOOT_WIDTH_M/2 + SIDE_STEP_M*extension) * torch.tensor([1., -1.])
    feet[:, :, 2] = FOOT_REST_HEIGHT_M + FOOT_LIFT_M*lift
    data = env.scene["robot"].data
    data.site_pos_w = feet
    data.site_lin_vel_w = torch.zeros(3, 2, 3)
    data.heading_w = torch.zeros(3)
    data.root_link_pos_w = torch.tensor([[0., 0., STAND_Z]]).repeat(3, 1)
    data.root_link_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    data.root_link_ang_vel_b = torch.zeros(3, 3)
    env.scene.sensors = {
        f"{side}_support_contact": SimpleNamespace(data=SimpleNamespace(found=(lift[:, i] < .2).unsqueeze(-1)))
        for i, side in enumerate(("left", "right"))
    }
    return env, hip_cfg, SimpleNamespace(name="robot", site_ids=[0, 1]), command


class ReferenceRewardTests(unittest.TestCase):
    def track(self, env, feet):
        return mdp.sidekick_reference_foot_track(
            env, feet, FOOT_LIFT_M, SIDE_STEP_M, HOME_FOOT_WIDTH_M, FOOT_REST_HEIGHT_M, STAND_Z,
        )

    def test_lift_precedes_extension_and_recovery_precedes_landing(self):
        command = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.],
                                [.9, .4, 0.], [-.9, .4, 0.], [0., 1., 0.], [0., -1., 0.]])
        lift, extension = mdp.sidekick_reference_targets(command)
        self.assertEqual(lift[:3].count_nonzero(), 0)
        self.assertGreater(lift[3, 1], 0)
        self.assertEqual(extension[3, 1], 0)
        torch.testing.assert_close(lift[3], lift[4])
        torch.testing.assert_close(lift[5:], torch.tensor([[0., 1.], [1., 0.]]))
        torch.testing.assert_close(extension[5:], lift[5:])

    def test_planted_pivots_cannot_earn_main_or_hip_reward(self):
        env, hips, feet, _ = kick_env()
        torch.testing.assert_close(self.track(env, feet), torch.ones(3))
        torch.testing.assert_close(mdp.sidekick_reference_hip_yaw_reward(env, hips, STAND_Z), torch.ones(3))
        # Even a raised site on a tilted/toe-pivoted foot must fail if it contacts.
        for sensor in env.scene.sensors.values():
            sensor.data.found[:] = True
        torch.testing.assert_close(self.track(env, feet), torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(mdp.sidekick_reference_hip_yaw_reward(env, hips, STAND_Z), torch.tensor([0., 0., 1.]))
        # A dense height signal remains for exploration, but is not the main score.
        env.scene["robot"].data.site_pos_w[:, :, 2] = FOOT_REST_HEIGHT_M
        clearance = mdp.sidekick_reference_clearance(env, feet, FOOT_LIFT_M, FOOT_REST_HEIGHT_M, STAND_Z)
        self.assertTrue(torch.all((clearance[:2] > 0) & (clearance[:2] < .1)))

    def test_hops_wrong_support_and_flops_do_not_pay(self):
        env, hips, feet, _ = kick_env()
        for sensor in env.scene.sensors.values():
            sensor.data.found.logical_not_()
        self.assertEqual(self.track(env, feet).count_nonzero(), 0)
        for sensor in env.scene.sensors.values():
            sensor.data.found[:] = False
        self.assertEqual(self.track(env, feet).count_nonzero(), 0)
        torch.testing.assert_close(mdp.sidekick_reference_hop_cost(env), torch.ones(3))
        env, hips, feet, _ = kick_env()
        env.scene["robot"].data.root_link_pos_w[:, 2] = .03
        self.assertEqual(self.track(env, feet).count_nonzero(), 0)
        self.assertEqual(mdp.sidekick_reference_hip_yaw_reward(env, hips, STAND_Z).count_nonzero(), 0)
        self.assertEqual(mdp.sidekick_reference_stay_put(env, stand_height=STAND_Z).count_nonzero(), 0)

    def test_side_extension_and_recovery_are_measured(self):
        env, _, feet, command = kick_env()
        env.scene["robot"].data.site_pos_w[:, :, 1] = torch.tensor([HOME_FOOT_WIDTH_M/2, -HOME_FOOT_WIDTH_M/2])
        self.assertTrue(torch.all(self.track(env, feet)[:2] < .25))
        env, _, feet, command = kick_env()
        command[:] = torch.tensor([1., 0., 0.])  # Time to land, still holding kick.
        self.assertEqual(self.track(env, feet)[:2].count_nonzero(), 0)

    def test_metrics_are_unweighted_and_world_frame_invariant(self):
        env, _, feet, _ = kick_env()
        metric = lambda name: mdp.sidekick_reference_metric(env, feet, name, HOME_FOOT_WIDTH_M)
        torch.testing.assert_close(metric("both_grounded"), torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(metric("airborne_kick"), metric("kick_window"))
        torch.testing.assert_close(metric("lateral_extension_m"), torch.tensor([SIDE_STEP_M, SIDE_STEP_M, 0.]), atol=1e-7, rtol=1e-5)
        env.scene.env_origins[:, 2] = 2.
        data = env.scene["robot"].data
        data.site_pos_w[:, :, 2] += 2.
        data.root_link_pos_w[:, 2] += 2.
        xy = data.site_pos_w[:, :, :2].clone()
        data.site_pos_w[:, :, 0] = -xy[:, :, 1] + 7.
        data.site_pos_w[:, :, 1] = xy[:, :, 0] - 3.
        data.heading_w[:] = torch.pi/2
        data.root_link_quat_w[:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        torch.testing.assert_close(self.track(env, feet), torch.ones(3), atol=1e-5, rtol=1e-5)


class ReferenceLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("reference_launcher", ROOT / "scripts/train_sidekick_dance.py")
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def test_reference_launch_parses_and_uses_separate_experiment(self):
        import mjlab
        from mjlab.scripts.train import TrainConfig
        import tyro
        args = self.launcher.parse_args(["--task", TASK, "--num-gpus", "2"])
        cmd = self.launcher.train_command(args, "reference_test", False)
        self.assertEqual(args.iterations, 8000)
        self.assertEqual(cmd[3], TASK)
        cfg = tyro.cli(TrainConfig, args=cmd[4:], default=TrainConfig.from_task(cmd[3]), config=mjlab.TYRO_FLAGS)
        self.assertEqual(cfg.agent.experiment_name, EXPERIMENT_NAME)
        self.assertEqual(self.launcher.TASK_SETTINGS[TASK][0], EXPERIMENT_NAME)
        self.assertEqual(cfg.gpu_ids, [0, 1])
        self.assertEqual(cfg.env.scene.num_envs, 1024)
        self.assertFalse(cfg.agent.resume)
        smoke = self.launcher.train_command(args, "reference_smoke", True)
        self.assertEqual(smoke[smoke.index("--agent.max-iterations") + 1], "5")
        self.assertEqual(smoke[smoke.index("--env.scene.num-envs") + 1], "64")

    def test_nan_and_positive_penalties_cannot_pass_smoke_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "smoke.log"
            path.write_text("Episode_Termination/nan_state: 0.0000\nEpisode_Reward/action_rate_l2: -1.0\n")
            self.launcher.validate_smoke_log(path)
            for bad in ("", "Episode_Termination/nan_state: 1.0\n",
                        "Episode_Termination/nan_state: nan\n",
                        "Episode_Termination/nan_state: 0.0\nEpisode_Reward/action_rate_l2: nan\n",
                        "Episode_Termination/nan_state: 0.0\nEpisode_Reward/action_rate_l2: 1.0\n"):
                path.write_text(bad)
                with self.assertRaises(RuntimeError):
                    self.launcher.validate_smoke_log(path)


if __name__ == "__main__":
    unittest.main()
