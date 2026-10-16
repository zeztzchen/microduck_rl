"""Alternating lateral dance steps, with the beat observable by the actor.

The twist slot carries [cos(phase), sin(phase), 0], not a velocity; all-zero
commands mean idle. Foot-space objectives let PPO discover the joint motion.
No fixed hip-yaw target (yaw alone neither lifts a foot nor makes a side step).
The shared walking recipe supplies BAM, observation noise/delays and safe DR.
This is an initial training recipe, not a validated dancing policy.
"""

import math
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

DANCE_PERIOD_S = 1.2
FOOT_LIFT_M = 0.012
SIDE_STEP_M = 0.012
HOME_FOOT_WIDTH_M = 0.084
FOOT_REST_HEIGHT_M = 0.0005
# CPU MuJoCo + BAM, alpha_walking at zero commands: trunk z=0.1154..0.1159 m,
# tilt=1.08..1.30 degrees over the final 2 s of a 4 s rollout (2026-10-15).
# HOME ctrl alone falls; this is a broad closed-loop balance objective, not a
# claim that holding HOME angles is a static equilibrium.
STAND_HEIGHT_M = 0.1155
ENABLE_PUSHES = False  # Add only after the dance itself has been learned.


def make_microduck_sidekick_dance_env_cfg(play: bool = False, rough: bool = False):
    if rough:
        raise ValueError("SideKickDance currently supports flat terrain only")
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play, rough=False))
    cfg.scene.entities["robot"] = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    cfg.episode_length_s = 8.0
    cfg.sim.nconmax = 100
    cfg.commands["twist"] = mdp.SideKickDanceCommandCfg(
        resampling_time_range=(8.0, 8.0),
        period=DANCE_PERIOD_S,
        idle_probability=0.0 if play else 0.1,
        randomize_start_side=not play,
    )
    for name, size in (("head_pose", 4), ("body_pose", 6)):
        cfg.commands[name] = mdp.UniformPoseCommandCfg(
            resampling_time_range=(2.0, 5.0),
            ranges=((-0.001, 0.001),) * size,
            zero_command_prob=1.0 if play else 0.5,
        )

    # No velocity-tracking reward/curriculum may interpret the clock as m/s.
    # Keep the base DR events at their conservative initial ranges; remove its
    # walking/pose/CoM ramps, which otherwise silently alter this new task.
    cfg.curriculum = {}
    keep = ("dof_pos_limits", "self_collisions", "action_rate_l2", "head_pose_tracking")
    cfg.rewards = {name: cfg.rewards[name] for name in keep}
    cfg.rewards["head_pose_tracking"].weight = 0.3
    cfg.rewards["action_rate_l2"].weight = 0.0
    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
    cfg.rewards.update(
        dance_tracking=RewardTermCfg(
            func=mdp.sidekick_dance_tracking,
            weight=6.0,
            params={
                "feet_cfg": feet,
                "lift_height": FOOT_LIFT_M,
                "side_distance": SIDE_STEP_M,
                "foot_width": HOME_FOOT_WIDTH_M,
                "rest_height": FOOT_REST_HEIGHT_M,
                "stand_height": STAND_HEIGHT_M,
            },
        ),
        dance_contact=RewardTermCfg(
            func=mdp.sidekick_dance_contact,
            weight=1.0,
            params={"stand_height": STAND_HEIGHT_M},
        ),
        dance_balance=RewardTermCfg(
            func=mdp.sidekick_dance_balance,
            weight=1.0,
            params={"stand_height": STAND_HEIGHT_M},
        ),
        dance_drift=RewardTermCfg(func=mdp.sidekick_dance_drift_cost, weight=-0.5),
        dance_foot_slip=RewardTermCfg(
            func=mdp.sidekick_dance_slip_cost, weight=-0.2, params={"feet_cfg": feet}
        ),
    )
    cfg.curriculum["dance_smoothness"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.1},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.3},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -0.5},
            ],
        },
    )
    if not ENABLE_PUSHES or play:
        cfg.events.pop("push_robot", None)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.12, 0.125),
        "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_robot_joints"].params.update(
        position_range=(-0.015, 0.015), velocity_range=(-0.05, 0.05),
        asset_cfg=SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
    )
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55)
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.rewards["action_rate_l2"].weight = -0.5
    return cfg


MicroduckSideKickDanceRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSideKickDanceRlCfg.experiment_name = "sidekick_dance"
MicroduckSideKickDanceRlCfg.run_name = "sidekick_dance"
MicroduckSideKickDanceRlCfg.max_iterations = 4000
MicroduckSideKickDanceRlCfg.save_interval = 250
MicroduckSideKickDanceRlCfg.logger = "tensorboard"
