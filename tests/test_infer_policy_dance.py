"""CPU-only dance command, timing, and policy-switch regression tests."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGED_MODEL = REPO / "policies/sidekick_reference_v2/model_3000/policy.onnx"


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location("infer_policy_dance", REPO / "scripts/infer_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(ip):
    result = mujoco.MjModel.from_xml_path(str(REPO / ip.MICRODUCK_XML))
    result.opt.timestep = 0.005
    return result


class FakeSession:
    def __init__(self, path, **kwargs):
        self.path = path
        self.input = SimpleNamespace(name=f"{path}_obs", shape=[1, 61], type="tensor(float)")
        self.output = SimpleNamespace(name=f"{path}_actions", shape=[1, 14], type="tensor(float)")
        self.observations = []

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map={})

    def run(self, output_names, feed):
        assert output_names == [self.output.name]
        assert list(feed) == [self.input.name]
        observation = feed[self.input.name]
        assert observation.shape == (1, 61)
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()
        self.observations.append(observation.copy())
        return [np.full((1, 14), 0.01, dtype=np.float32)]


@pytest.fixture
def make_policy(ip, model, monkeypatch):
    monkeypatch.setattr(ip.ort, "InferenceSession", FakeSession)

    def create(**overrides):
        arguments = dict(walking_onnx_path="walking", standing_onnx_path="standing",
                         sitstand_onnx_path="sitstand", roulade_onnx_path="roulade",
                         ground_pick_onnx_path="ground_pick", slope_onnx_path="slope",
                         dance_onnx_path="dance", new_cmd_obs=True, use_projected_gravity=True)
        arguments.update(overrides)
        data = mujoco.MjData(model)
        policy = ip.PolicyInference(model, data, **arguments)
        data.qpos[policy._trunk_qpos_adr + 2] = 0.125
        data.qpos[policy.joint_qpos_indices] = policy.default_pose
        mujoco.mj_forward(model, data)
        return policy

    return create


def test_groundcontact_scene_and_servo_layout(ip, model):
    scene = ET.parse(REPO / ip.MICRODUCK_XML)
    assert "robot_groundcontact.xml" in [element.get("file") for element in scene.findall("include")]
    assert model.nu == 14
    assert [model.joint(joint_id).name for joint_id in model.actuator_trnid[:, 0]] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]


@pytest.mark.parametrize("period", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_period_must_be_finite_and_positive(make_policy, period):
    with pytest.raises(ValueError, match="dance-period"):
        make_policy(dance_period=period)


@pytest.mark.parametrize("overrides,message", [
    ({"new_cmd_obs": False}, "new-cmd-obs"),
    ({"use_projected_gravity": False}, "projected gravity"),
])
def test_dance_requires_compatible_observations(make_policy, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_policy(**overrides)


@pytest.mark.parametrize("tensor,field,value", [
    ("input", "shape", [1, 51]), ("output", "shape", [1, 15]),
    ("input", "type", "tensor(double)"), ("output", "type", "tensor(double)"),
])
def test_dance_rejects_wrong_onnx_contract(make_policy, ip, monkeypatch, tensor, field, value):
    def invalid_session(path, **kwargs):
        session = FakeSession(path)
        if path == "dance":
            setattr(getattr(session, tensor), field, value)
        return session

    monkeypatch.setattr(ip.ort, "InferenceSession", invalid_session)
    with pytest.raises(ValueError, match="float32"):
        make_policy()


@pytest.mark.parametrize("stop_key", ["d", " "])
@pytest.mark.parametrize("with_standing", [True, False])
def test_start_stop_and_command_restoration(make_policy, stop_key, with_standing):
    policy = make_policy(standing_onnx_path="standing" if with_standing else None)
    policy.set_vel_cmd(0.2, 0.1, 0.3)
    policy.head_offset[:] = 0.4
    policy.body_cmd[:] = 0.02
    policy.head_mode = policy.body_pose_mode = True
    policy.last_action[:] = 0.25
    assert policy.handle_dance_key("d")
    observation = policy.get_observations()
    np.testing.assert_array_equal(observation[48:51], [1, 0, 0])
    np.testing.assert_array_equal(observation[51:61], 0)
    np.testing.assert_array_equal(observation[34:48], 0.25)
    np.testing.assert_array_equal(policy.vel_cmd, 0)
    assert not policy.head_mode and not policy.body_pose_mode
    policy.infer()
    assert policy.input_name == "dance_obs" and policy.output_name == "dance_actions"
    policy.update_dance_phase(0.3)
    assert policy.handle_dance_key(stop_key)
    assert not policy.dance_mode
    assert policy.current_policy == "dance"
    assert policy.ort_session is policy.dance_session
    np.testing.assert_array_equal(policy.get_observations()[48:61], 0)
    policy.update_dance_phase(1.0)
    assert policy.dance_phase == 0
    policy._update_policy_session()
    assert policy.current_policy == "dance"
    policy.infer()
    policy.toggle_dance()
    assert policy.dance_phase == 0
    np.testing.assert_array_equal(policy.command[:3], [1, 0, 0])
    assert policy.handle_dance_key("x")
    assert not policy.dance_mode
    assert policy.current_policy == ("standing" if with_standing else "walking")
    np.testing.assert_array_equal(policy.command, 0)
    policy.infer()


def test_simulation_phase_wrap_pause_and_resume(make_policy):
    policy = make_policy()
    policy.toggle_dance()
    for _ in range(15):
        start_time = policy.data.time
        mujoco.mj_step(policy.model, policy.data, nstep=4)
        policy.update_dance_phase(policy.data.time - start_time)
    assert policy.dance_phase == pytest.approx(0.25)
    np.testing.assert_allclose(policy.command[:3], [0, 1, 0], atol=1e-6)
    frozen_command = policy.command.copy()
    for _ in range(100):
        start_time = policy.data.time
        mujoco.mj_step(policy.model, policy.data, nstep=4)
        policy.update_dance_phase(policy.data.time - start_time, inference_enabled=False)
    np.testing.assert_array_equal(policy.command, frozen_command)
    assert policy.dance_phase == pytest.approx(0.25)
    policy.update_dance_phase(0.02)
    assert policy.dance_phase == pytest.approx(0.25 + 1 / 60)
    policy.update_dance_phase(2.4)
    assert policy.dance_phase == pytest.approx(0.25 + 1 / 60)


@pytest.mark.parametrize("active", [False, True])
def test_dance_locks_commands_and_excludes_other_modes(make_policy, active):
    policy = make_policy()
    policy.toggle_dance()
    if not active:
        policy.stop_dance()
    for key in ("up", "down", "left", "right", "a", "e", "z", "s", "h", "b", "g", "y", "r", "k", "l"):
        assert policy.handle_dance_key(key)
    for key in ("t", "q"):
        assert not policy.handle_dance_key(key)
    policy.set_vel_cmd(0.3, 0.2, 1.5)
    policy.toggle_sit()
    policy.trigger_behavior("roulade")
    policy.trigger_ground_pick()
    policy.toggle_slope_mode()
    policy.toggle_head_mode()
    policy.toggle_body_pose_mode()
    policy.bump_body("z", 0.01)
    policy._update_policy_session()
    assert policy.current_policy == "dance"
    assert not any((policy.sit_mode, policy.ground_pick_mode, policy.slope_mode, policy.head_mode, policy.body_pose_mode))
    assert policy.behavior_mode is None
    np.testing.assert_array_equal(policy.vel_cmd, 0)
    np.testing.assert_array_equal(policy.command, [int(active)] + [0] * 12)


@pytest.mark.parametrize("mode", ["sit", "roulade", "ground_pick", "slope"])
def test_other_modes_block_dance_and_retain_own_session(make_policy, mode):
    policy = make_policy()
    if mode == "sit":
        policy.toggle_sit()
    elif mode == "roulade":
        policy.trigger_behavior(mode)
    elif mode == "ground_pick":
        policy.trigger_ground_pick()
        policy.update_ground_pick_phase(0.02)
    else:
        policy.toggle_slope_mode()
    session = policy.ort_session
    command = policy.command.copy()
    policy.toggle_dance()
    assert not policy.dance_mode
    assert policy.ort_session is session
    np.testing.assert_array_equal(policy.command, command)
    policy.infer()


def test_sitstand_roulade_and_walk_still_switch_with_session_names(make_policy):
    policy = make_policy()
    policy.toggle_dance()
    policy.infer()
    policy.stop_dance()
    policy.leave_dance()
    policy.toggle_sit()
    assert policy.command[0] == 1
    policy.infer()
    policy.toggle_sit()
    assert policy.command[0] == 0
    policy.infer()
    assert len(policy.sit_session.observations) == 2
    policy.trigger_behavior("roulade")
    np.testing.assert_array_equal(policy.command, 0)
    policy.infer()
    policy.update_behavior(2.1)
    assert policy.current_policy == "walking"
    policy.set_vel_cmd(0.1, 0.0, 0.0)
    assert policy.command[0] == pytest.approx(0.1)
    policy.infer()
    policy.set_vel_cmd(0.0, 0.0, 0.0)
    assert policy.current_policy == "standing"
    policy.infer()


def test_viewer_loop_uses_sim_time_and_terminal_pause(ip, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(ip.ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(ip.mujoco.viewer, "_MJPYTHON", object())
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(ip.sys, "argv", ["infer_policy.py", "--walking", "walking", "--dance", "dance", "--new-cmd-obs"])
    captured = []
    original_policy = ip.PolicyInference

    def create_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        captured.append(policy)
        return policy

    monkeypatch.setattr(ip, "PolicyInference", create_policy)
    schedule = [["d"], [], ["t"], ["up", "y", "r", "h"], [], ["t"], [" "], ["d"], ["0"], ["d"], ["x"], ["q"]]
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(ip, "time", SimpleNamespace(time=lambda: clock.now, sleep=lambda seconds: None))

    class Terminal:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_keys(self):
            return schedule.pop(0)

    class Viewer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def is_running(self):
            return bool(schedule)

        def sync(self):
            clock.now += 10.0

    monkeypatch.setattr(ip, "TerminalInput", Terminal)
    monkeypatch.setattr(ip.mujoco.viewer, "launch_passive", lambda *args, **kwargs: Viewer())
    ip.main()
    policy = captured[0]
    observations = policy.dance_session.observations
    assert len(observations) == 7
    for observation, phase in zip(observations, [0, 1 / 60, 2 / 60, None, 0, None, 0]):
        command = [0, 0, 0] if phase is None else [np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase), 0]
        np.testing.assert_allclose(observation[0, 48:51], command, atol=1e-6)
        np.testing.assert_array_equal(observation[0, 51:61], 0)
    np.testing.assert_array_equal(observations[5][0, 34:48], 0)
    assert not policy.dance_mode
    for observation in policy.walking_session.observations:
        np.testing.assert_array_equal(observation[0, 48:61], 0)


@pytest.mark.parametrize("extra,message", [
    ([], "new-cmd-obs"),
    (["--new-cmd-obs", "--roller"], "roller"),
    (["--new-cmd-obs", "--raw-accelerometer"], "projected gravity"),
    (["--new-cmd-obs", "--dance-period", "0"], "dance-period"),
    (["--new-cmd-obs", "--dance-period", "nan"], "dance-period"),
])
def test_cli_rejects_incompatible_dance_arguments(ip, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(ip.sys, "argv", ["infer_policy.py", "--walking", "walk.onnx", "--dance", "dance.onnx", *extra])
    with pytest.raises(SystemExit) as result:
        ip.main()
    assert result.value.code == 2
    assert message in capsys.readouterr().err


def test_local_dance_model_multiple_phases(ip, model):
    dance_path = REPO / "sidekick_dance_model_1000.onnx"
    walking_path = REPO / "alpha_walking.onnx"
    if not dance_path.exists() or not walking_path.exists():
        pytest.skip("Local ONNX artifacts are not tracked in git")
    data = mujoco.MjData(model)
    policy = ip.PolicyInference(model, data, walking_onnx_path=str(walking_path),
                                dance_onnx_path=str(dance_path), new_cmd_obs=True,
                                use_projected_gravity=True)
    data.qpos[policy._trunk_qpos_adr + 2] = 0.125
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    mujoco.mj_forward(model, data)
    metadata = policy.dance_session.get_modelmeta().custom_metadata_map
    assert metadata["joint_names"].split(",") == [model.joint(joint_id).name for joint_id in model.actuator_trnid[:, 0]]
    np.testing.assert_allclose(np.fromstring(metadata["default_joint_pos"], sep=","), policy.default_pose, atol=0.00051)
    assert float(metadata["action_scale"]) == policy.action_scale == 1.0
    policy.toggle_dance()
    for phase in (0, 0.125, 0.25, 0.5, 0.75, 0.999):
        policy.dance_phase = phase
        policy._update_command()
        action = policy.infer()
        assert action.shape == (14,) and np.isfinite(action).all()
    policy.stop_dance()
    assert np.isfinite(policy.infer()).all()


def test_reset_clears_physics_modes_and_action_history(make_policy):
    policy = make_policy(delay_min_lag=1, delay_max_lag=2)
    policy.toggle_dance()
    policy.update_dance_phase(0.3)
    policy.apply_action(policy.infer())
    policy.data.time = 9
    policy.data.qvel[:] = 1
    policy.data.xfrc_applied[:] = 2
    policy.behavior_mode = "roulade"
    policy.behavior_time_left = 2
    policy.sit_mode = policy.ground_pick_mode = policy.slope_mode = True
    policy.reset_to_stand()
    np.testing.assert_array_equal(policy.data.qpos, policy.model.key("STAND").qpos)
    np.testing.assert_array_equal(policy.data.qvel, 0)
    np.testing.assert_array_equal(policy.data.xfrc_applied, 0)
    np.testing.assert_array_equal(policy.last_action, 0)
    np.testing.assert_array_equal(policy.action_buffer, 0)
    np.testing.assert_array_equal(policy.command, 0)
    assert policy.data.time == policy.dance_phase == policy.buffer_index == 0
    assert policy.current_policy == "dance" and not policy.dance_mode
    assert not any((policy.sit_mode, policy.ground_pick_mode, policy.slope_mode))
    assert policy.behavior_mode is None and policy.behavior_time_left == 0


def test_nonfinite_dance_action_is_not_applied(make_policy, monkeypatch):
    policy = make_policy()
    policy.toggle_dance()
    previous = policy.last_action.copy()
    monkeypatch.setattr(policy.dance_session, "run", lambda *args: [np.full((1, 14), np.nan)])
    with pytest.raises(ValueError, match="invalid actions"):
        policy.infer()
    np.testing.assert_array_equal(policy.last_action, previous)


@pytest.fixture
def packaged_policy(ip):
    def create(**overrides):
        actuator = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
        model, data, controller, _ = ip.load_mujoco_with_bam(ip.MICRODUCK_XML, actuator, 0.005, 0.1, ip.BAM_VIN_MIN)
        arguments = dict(bam_ctrl=controller, dance_onnx_path=str(PACKAGED_MODEL),
                         new_cmd_obs=True, use_projected_gravity=True)
        arguments.update(overrides)
        return ip.PolicyInference(model, data, **arguments)
    return create


def test_packaged_model_contract_idle_phases_and_bam_reset(packaged_policy):
    policy = packaged_policy()
    assert policy.dance_metadata["checkpoint"] == "model_3000.pt"
    assert policy.dance_period == 1.2
    policy.reset_to_stand()
    np.testing.assert_array_equal(policy.get_observations()[48:61], 0)
    assert np.isfinite(policy.infer()).all()
    policy.toggle_dance()
    for phase in np.arange(8) / 8:
        policy.dance_phase = phase
        policy._update_command()
        observation = policy.get_observations()
        np.testing.assert_allclose(np.linalg.norm(observation[3:6]), 1, atol=1e-6)
        np.testing.assert_array_equal(observation[51:61], 0)
        action = policy.infer()
        assert action.shape == (14,) and np.isfinite(action).all()
        np.testing.assert_array_equal(policy.get_observations()[34:48], action)
        policy.apply_action(action)
        np.testing.assert_allclose(policy.bam_ctrl.q_target, policy.default_pose + action)
    policy.stop_dance()
    assert np.isfinite(policy.infer()).all()
    policy.leave_dance()
    assert policy.current_policy == "dance" and not policy.dance_mode
    policy.bam_ctrl._prev_motor_torque[:] = 1
    policy.bam_ctrl.last_ts = 10
    policy.reset_to_stand()
    np.testing.assert_array_equal(policy.bam_ctrl._prev_motor_torque, 0)
    np.testing.assert_array_equal(policy.bam_ctrl.q_target, policy.default_pose)
    assert policy.bam_ctrl.last_ts == policy.data.time == 0


@pytest.mark.parametrize("field,value,message", [
    ("onnx_sha256", "incorrect", "SHA256"),
    ("observation_normalizer_baked_in", False, "normalizer"),
    ("input", {"name": "wrong", "shape": [1, 61]}, "input"),
    ("control_hz", 100, "control_hz"),
    ("period_s", 0, "period_s"),
    ("joint_order", [], "joint order"),
    ("home_rad", [0] * 14, "HOME"),
    ("bam", {}, "BAM settings"),
])
def test_packaged_metadata_rejects_mismatches(packaged_policy, tmp_path, field, value, message):
    metadata = json.loads(PACKAGED_MODEL.with_name("metadata.json").read_text())
    target = metadata if field in metadata else metadata["deployment"]
    target[field] = value
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        packaged_policy(dance_metadata_path=str(path))


@pytest.mark.parametrize("overrides,message", [
    ({"dance_period": 2.4}, "period_s"),
    ({"action_scale": 0.5}, "action-scale"),
    ({"bam_ctrl": None}, "requires BAM"),
])
def test_packaged_model_rejects_runtime_overrides(packaged_policy, overrides, message):
    with pytest.raises(ValueError, match=message):
        packaged_policy(**overrides)


def test_macos_python_reports_mjpython_before_loading(ip, monkeypatch, capsys):
    monkeypatch.setattr(ip.sys, "platform", "darwin")
    monkeypatch.setattr(ip.mujoco.viewer, "_MJPYTHON", None)
    monkeypatch.setattr(ip.sys, "argv", ["infer_policy.py", "--walking", "not-loaded.onnx"])
    with pytest.raises(SystemExit) as result:
        ip.main()
    assert result.value.code == 2
    assert "uv run --no-sync mjpython" in capsys.readouterr().err


@pytest.mark.parametrize("idle", [False, True])
def test_headless_evaluator_uses_dance_adapter(tmp_path, idle):
    output = tmp_path / "evaluation.json"
    command = [sys.executable, str(REPO / "scripts/eval_sidekick_reference.py"),
               "--onnx-file", str(PACKAGED_MODEL), "--trials", "2", "--seconds", "1.4",
               "--burn-in", "0.2", "--output", str(output)]
    if idle:
        command.append("--idle")
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dance metadata:" in result.stdout and "Loading walking policy" not in result.stdout
    report = json.loads(output.read_text())
    assert report["idle"] == idle and len(report["trials"]) == 2
    for trial in report["trials"]:
        assert trial["samples"] == 60
        for name, value in trial.items():
            if value is not None:
                assert np.isfinite(value), name
        assert (trial["left_peak_samples"] == 0) == idle
