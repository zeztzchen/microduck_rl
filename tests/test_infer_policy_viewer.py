"""A missing display must fail promptly instead of hanging in launch_passive."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux display handling")
class ViewerStartupTests(unittest.TestCase):
    def test_no_display_exits_before_loading_policy(self):
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env["MUJOCO_GL"] = "glfw"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/infer_policy.py",
                "--walking",
                "unused.onnx",
                "--new-cmd-obs",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("No graphical display configured", result.stderr)
        self.assertIn("DISPLAY", result.stderr)
        self.assertNotIn("Loading MuJoCo model", result.stdout)


if __name__ == "__main__":
    unittest.main()
