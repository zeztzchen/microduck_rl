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
        model, data, walking_onnx_path=str(args.onnx_file), bam_ctrl=controller,
        new_cmd_obs=True, use_projected_gravity=True,
    )
    feet = [model.geom(f"{side}_foot_collision").id for side in ("left", "right")]
    sites = [model.site(f"{side}_foot").id for side in ("left", "right")]
    vertices = [sole_vertices(model, geom) for geom in feet]
    floor = model.geom("floor").id
    trunk = model.body("trunk_base").id
    trials = []
    for trial in range(args.trials):
        mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
        data.qpos[policy.joint_qpos_indices] += rng.uniform(-.01, .01, 14)
        policy.last_action[:] = 0
        controller.q_target[:] = policy.default_pose
        mujoco.mj_forward(model, data)
        rows = []
        fell = False
        for step in range(round(args.seconds / .02)):
            phase = (step * .02 / 1.2 + (trial % 2) * .5) % 1
            sine = np.sin(2 * np.pi * phase)
            policy.command[:] = 0
            if not args.idle:
                policy.command[:2] = [np.cos(2 * np.pi * phase), sine]
            policy.apply_action(policy.infer())
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite MuJoCo state")
            contact = np.zeros(2, dtype=bool)
            for item in data.contact:
                pair = {int(item.geom1), int(item.geom2)}
                for i, geom in enumerate(feet):
                    contact[i] |= pair == {floor, geom}
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
                    fallen=fallen,
                ))
        if not rows:
            raise ValueError("seconds must exceed burn-in")
        average = lambda key, subset: float(np.mean([row[key] for row in subset])) if subset else None
        report = {"trial": trial, "fell": bool(fell), "samples": len(rows)}
        for key in ("both_contact", "both_air", "tilt_deg", "trunk_height_m",
                    "left_clearance_mm", "right_clearance_mm", "fallen"):
            report[key + "_mean"] = average(key, rows)
        for side, index in (("left", 0), ("right", 1)):
            subset = [r for r in rows if r["peak"] and r["swing"] == index]
            report[side + "_peak_samples"] = len(subset)
            for key in ("swing_air", "support_contact", "swing_clearance_mm", "foot_width_mm"):
                report[side + "_peak_" + key] = average(key, subset)
        trials.append(report)
    return {"onnx": str(args.onnx_file.resolve()), "seconds_per_trial": args.seconds,
            "burn_in_s": args.burn_in, "seed": args.seed, "idle": args.idle,
            "gravity_observation": "projected_gravity", "trials": trials,
            "note": "Sole clearance uses the lowest collision-mesh vertex; rates are sample fractions. No rendering or domain randomization."}


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
