#!/usr/bin/env python3
"""CPU MuJoCo/BAM dance measurements, without a viewer or graphics context.

Use the same ONNX observation/action adapter as infer_policy.py. This is a
nominal deployment rehearsal, not an evaluation over training randomization.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "disable"

import mujoco
import numpy as np

from infer_policy import (
    BAM_KP_FW, BAM_VIN_MIN, MICRODUCK_XML, PolicyInference,
    load_bam_model, load_mujoco_with_bam,
)


def sole_vertices(model, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    return model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]].copy()


def evaluate(args):
    rng = np.random.default_rng(args.seed)
    model, data, controller, _ = load_mujoco_with_bam(
        MICRODUCK_XML, load_bam_model(BAM_KP_FW, 7.4, None), .005, .1, BAM_VIN_MIN,
    )
    policy = PolicyInference(
        model, data, dance_onnx_path=str(args.onnx_file), bam_ctrl=controller,
        new_cmd_obs=True, use_projected_gravity=True,
    )
    feet = [model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
    sites = [model.site(f"{side}_foot").id for side in ("left", "right")]
    vertices = [sole_vertices(model, geom) for geom in feet]
    floor = model.geom("floor").id
    trunk = model.body("trunk_base").id
    trials = []
    for trial in range(args.trials):
        policy.reset_to_stand()
        data.qpos[policy.joint_qpos_indices] += rng.uniform(-.01, .01, 14)
        controller.reset(data.qpos)
        controller.q_target[:] = policy.default_pose
        mujoco.mj_forward(model, data)
        initial_xy = data.xpos[trunk, :2].copy()
        if not args.idle:
            policy.toggle_dance()
            policy.dance_phase = (trial % 2) * .5
            policy._update_command()
        rows = []
        fell = False
        for step in range(round(args.seconds / .02)):
            phase = policy.dance_phase
            sine = np.sin(2 * np.pi * phase)
            policy.apply_action(policy.infer())
            physics_start_time = data.time
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model, data)
            policy.update_dance_phase(data.time - physics_start_time)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite MuJoCo state")
            contact = np.zeros(2, dtype=bool)
            slip_speeds = [[], []]
            foot_velocities = []
            for geom in feet:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, velocity, 0)
                foot_velocities.append(velocity)
            for item in data.contact:
                pair = {int(item.geom1), int(item.geom2)}
                for i, geom in enumerate(feet):
                    if pair == {floor, geom}:
                        contact[i] = True
                        velocity = foot_velocities[i]
                        point_velocity = velocity[3:] + np.cross(
                            velocity[:3], item.pos - data.geom_xpos[geom],
                        )
                        slip_speeds[i].append(float(np.linalg.norm(point_velocity[:2])))
            heights = [float((v @ data.geom_xmat[g].reshape(3, 3).T
                              + data.geom_xpos[g])[:, 2].min())
                       for g, v in zip(feet, vertices)]
            rotation = data.xmat[trunk].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
            z = float(data.xpos[trunk, 2])
            fallen = tilt > 55 or z < .065
            fell |= fallen
            if step * .02 >= args.burn_in:
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                delta = data.site_xpos[sites[0]] - data.site_xpos[sites[1]]
                width = -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]
                swing = 1 if sine >= 0 else 0
                rows.append(dict(
                    both_contact=bool(contact.all()), both_air=bool((~contact).all()),
                    peak=bool(abs(sine) > .85 and not args.idle), swing=swing,
                    swing_air=bool(not contact[swing]), support_contact=bool(contact[1-swing]),
                    swing_clearance_mm=heights[swing] * 1000,
                    left_clearance_mm=heights[0] * 1000, right_clearance_mm=heights[1] * 1000,
                    foot_width_mm=float(width * 1000), tilt_deg=tilt, trunk_height_m=z,
                    drift_xy_m=float(np.linalg.norm(data.xpos[trunk, :2] - initial_xy)),
                    left_contact=bool(contact[0]), right_contact=bool(contact[1]),
                    left_slip_speed_mps=float(np.mean(slip_speeds[0])) if slip_speeds[0] else None,
                    right_slip_speed_mps=float(np.mean(slip_speeds[1])) if slip_speeds[1] else None,
                    fallen=fallen,
                ))
        if not rows:
            raise ValueError("seconds must exceed burn-in")
        average = lambda key, subset: float(np.mean([row[key] for row in subset])) if subset else None
        report = {"trial": trial, "fell": bool(fell), "samples": len(rows)}
        report["drift_xy_max_m"] = max(row["drift_xy_m"] for row in rows)
        report["drift_xy_final_m"] = rows[-1]["drift_xy_m"]
        for key in ("both_contact", "both_air", "tilt_deg", "trunk_height_m",
                    "left_clearance_mm", "right_clearance_mm", "fallen"):
            report[key + "_mean"] = average(key, rows)
        for side, index in (("left", 0), ("right", 1)):
            grounded = [row for row in rows if row[side + "_contact"]]
            report[side + "_contact_slip_speed_mps"] = average(side + "_slip_speed_mps", grounded)
            subset = [r for r in rows if r["peak"] and r["swing"] == index]
            report[side + "_peak_samples"] = len(subset)
            for key in ("swing_air", "support_contact", "swing_clearance_mm", "foot_width_mm"):
                report[side + "_peak_" + key] = average(key, subset)
        trials.append(report)
    return {"onnx": str(args.onnx_file.resolve()), "seconds_per_trial": args.seconds,
            "burn_in_s": args.burn_in, "seed": args.seed, "idle": args.idle,
            "gravity_observation": "projected_gravity", "trials": trials,
            "note": "Sole clearance uses the lowest collision-mesh vertex; rates are sample fractions. Slip is horizontal contact-point speed, averaged over grounded samples on the static plane. No rendering or domain randomization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-file", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--burn-in", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--idle", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or not 0 <= args.burn_in < args.seconds:
        parser.error("Need trials >= 1 and 0 <= burn-in < seconds")
    result = evaluate(args)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
