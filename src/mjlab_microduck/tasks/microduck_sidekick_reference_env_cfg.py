"""Lifted side-kick v2, correcting the reference recipe's planted-foot shortcut.

The public reference is task code, not evidence of a successfully trained dance.
Its hip-only v1 reward produced ~99.7% double support in our CPU rehearsal of
model_7999. V2 explicitly tracks lift/extend/recover/land, gates the main score
on phase-correct contact, and keeps hip yaw as a secondary style objective.

The 61D clock/action contract and BAM physics stay compatible with v1 ONNX.
New runs use sidekick_reference_v2; old checkpoints/artifacts are preserved.
This revised objective still needs training and behavioral validation.
"""

from copy import deepcopy
import math

from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)

REFERENCE_URL = "https://github.com/pezzonovante7/microduck-sidekick-dance"
REFERENCE_COMMIT = "02c5d96704d3ac266acb29fbb29957335ee680bc"
RECIPE_VERSION = "lifted_sidekick_v2"
EXPERIMENT_NAME = "sidekick_reference_v2"
DANCE_PERIOD_S = 1.2
REFERENCE_YAW_TARGET = 0.55
YAW_TARGET = 0.40
FOOT_LIFT_M = 0.015
SIDE_STEP_M = 0.015
HOME_FOOT_WIDTH_M = 0.084
FOOT_REST_HEIGHT_M = 0.0005
# CPU BAM v1 closed-loop stand/dance: z=0.11517..0.11530 m, tilt=0.51..0.54 deg.
# HOME ctrl alone is not a stable equilibrium; use a broad balance objective.
STAND_Z = 0.1155
EPISODE_LENGTH_S = 8.0
ENABLE_VELOCITY_PUSHES = False  # Reintroduce only after clean single support.
ENABLE_SYMMETRY = False


def make_microduck_sidekick_reference_env_cfg(play: bool = False, rough: bool = False):
    if rough:
        raise ValueError("SideKickReference supports flat terrain only")
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play, rough=False))
    cfg.scene.entities["robot"] = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    cfg.sim.nconmax = 100
    cfg.episode_length_s = EPISODE_LENGTH_S
    support_sensors = tuple(
        ContactSensorCfg(
            name=f"{side}_support_contact",
            primary=ContactMatch(
                mode="geom", pattern=f"^{side}_foot_collision$", entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",), reduce="netforce", num_slots=1,
        )
        for side in ("left", "right")
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *support_sensors)
    cfg.terminations["nan_state"].params["sensor_names"] = (
        "feet_ground_contact", "left_support_contact", "right_support_contact"
    )

    cfg.commands["twist"] = mdp.SideKickDanceCommandCfg(
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        period=DANCE_PERIOD_S,
        idle_probability=0.0 if play else 0.1,
        randomize_start_side=not play,
    )
    for name, size in (("head_pose", 4), ("body_pose", 6)):
        cfg.commands[name] = mdp.UniformPoseCommandCfg(
            resampling_time_range=(2.0, 5.0), ranges=((-0.001, 0.001),) * size,
            zero_command_prob=1.0 if play else 0.5,
        )

    keep = ("body_ang_vel", "dof_pos_limits", "action_rate_l2", "self_collisions")
    cfg.rewards = {name: cfg.rewards[name] for name in keep}
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = 0.0
    cfg.rewards["self_collisions"].weight = -1.0
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",)
    )
    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
    cfg.rewards.update(
        sidekick_foot_track=RewardTermCfg(
            func=mdp.sidekick_reference_foot_track, weight=8.0,
            params={"feet_cfg": feet, "lift_height": FOOT_LIFT_M,
                    "side_distance": SIDE_STEP_M, "foot_width": HOME_FOOT_WIDTH_M,
                    "rest_height": FOOT_REST_HEIGHT_M, "stand_height": STAND_Z},
        ),
        sidekick_clearance=RewardTermCfg(
            func=mdp.sidekick_reference_clearance, weight=2.0,
            params={"feet_cfg": feet, "lift_height": FOOT_LIFT_M,
                    "rest_height": FOOT_REST_HEIGHT_M, "stand_height": STAND_Z},
        ),
        phase_contact=RewardTermCfg(
            func=mdp.sidekick_reference_contact, weight=2.0,
            params={"stand_height": STAND_Z},
        ),
        dance_balance=RewardTermCfg(
            func=mdp.sidekick_dance_balance, weight=1.0,
            params={"stand_height": STAND_Z},
        ),
        sidekick_hip_yaw=RewardTermCfg(
            func=mdp.sidekick_reference_hip_yaw_reward, weight=1.0,
            params={
                "std": 0.25, "yaw_target": YAW_TARGET, "stand_height": STAND_Z,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("left_hip_yaw", "right_hip_yaw"),
                    preserve_order=True,
                ),
            },
        ),
        stay_put=RewardTermCfg(
            func=mdp.sidekick_reference_stay_put, weight=0.5,
            params={"std": 0.08, "stand_height": STAND_Z},
        ),
        pose_stand_legs=RewardTermCfg(
            func=mdp.sidekick_reference_home_pose, weight=0.2,
            params={"std": 0.6, "stand_height": STAND_Z, "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^(left|right)_(hip_roll|hip_pitch|knee|ankle)$",)
            )},
        ),
        pose_stand_neck=RewardTermCfg(
            func=mdp.sidekick_reference_home_pose, weight=0.3,
            params={"std": 0.3, "stand_height": STAND_Z, "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r"^(neck_pitch|head_pitch|head_yaw|head_roll)$",)
            )},
        ),
        dance_hop=RewardTermCfg(func=mdp.sidekick_reference_hop_cost, weight=-2.0),
        dance_drift=RewardTermCfg(func=mdp.sidekick_dance_drift_cost, weight=-1.0),
        dance_foot_slip=RewardTermCfg(
            func=mdp.sidekick_reference_slip_cost, weight=-0.2, params={"feet_cfg": feet},
        ),
    )
    for metric in ("both_grounded", "both_airborne", "kick_window", "airborne_kick",
                   "support_lost", "left_site_height_m", "right_site_height_m", "lateral_extension_m"):
        cfg.metrics[f"dance/{metric}"] = MetricsTermCfg(
            func=mdp.sidekick_reference_metric,
            params={"feet_cfg": feet, "metric": metric, "foot_width": HOME_FOOT_WIDTH_M},
        )

    # No early attempt tax. Remove inherited velocity curricula: the command
    # slots now encode a clock, not a requested walking speed.
    cfg.curriculum = {
        "action_rate_weight": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={"reward_name": "action_rate_l2", "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.1},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.3},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -0.5},
            ]},
        ),
    }
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.12, 0.125),
        "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_robot_joints"].params.update(
        position_range=(-0.015, 0.015), velocity_range=(-0.05, 0.05),
        asset_cfg=SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
    )
    if ENABLE_VELOCITY_PUSHES and not play:
        cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2)
        }
    else:
        cfg.events.pop("push_robot", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55)
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.rewards["action_rate_l2"].weight = -0.5
    return cfg


MicroduckSideKickReferenceRlCfg = deepcopy(MicroduckRlCfg)
MicroduckSideKickReferenceRlCfg.experiment_name = EXPERIMENT_NAME
MicroduckSideKickReferenceRlCfg.run_name = EXPERIMENT_NAME
MicroduckSideKickReferenceRlCfg.max_iterations = 8000
MicroduckSideKickReferenceRlCfg.save_interval = 250
MicroduckSideKickReferenceRlCfg.logger = "tensorboard"
MicroduckSideKickReferenceRlCfg.algorithm.symmetry_cfg = None
