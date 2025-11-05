"""
Lightly cleaned runner script for GA quadruped.
- Tidier imports & constants
- CLI flags for simulation, controller type, XML/ONNX paths
- Safer shutdown on exceptions/KeyboardInterrupt
- Optional torque/temperature printing
- Minimal comments for future readers

Assumes project-local modules are importable.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Callable, Tuple

import numpy as np
from blessed import Terminal
from scipy.spatial.transform import Rotation as R

from ga_can.core.ga_logging import get_logger
from tqdm import tqdm
from ga_quadruped.controller.accelerate_controller import AccelerateController  # noqa: F401 (left for discoverability)
from ga_quadruped.controller.jump_controller import JumpController
from ga_quadruped.controller.sbus_controller import SbusVelocityController
from ga_quadruped.controller.velocity_controller import VelocityController
from ga_quadruped.param.param import Param
from ga_quadruped.plot.publisher import SimpleZmqPublisher
from ga_quadruped.policy_agent.jump_policy_agent import JumpPolicyAgent
from ga_quadruped.policy_agent.recovery_policy_agent import RecoveryPolicyAgent
from ga_quadruped.policy_agent.velocity_policy_agent import VelocityPolicyAgent
from ga_quadruped.robot.quadruped_init import QuadrupedDefaultInitializer
from ga_quadruped.sim2sim.robot import SimRobot

from mujoco.viewer import launch_passive

# =====================
# Constants & Defaults
# =====================

VEL_STEP = 0.1           # m/s per key press
DEFAULT_DT = 0.02        # s
JUMP_STEPS = 120         # placeholder if jump is re-enabled

# Default home pose (Go1-like ordering)
_THETA0 = 0.0
_THETA1 = 0.4
_THETA2 = 1.2
HOME_POSE = [
    _THETA0, -_THETA1, _THETA2,
    -_THETA0, _THETA1, -_THETA2,
    _THETA0, -_THETA1, _THETA2,
    -_THETA0, _THETA1, -_THETA2,
]

term = Terminal()
pub = SimpleZmqPublisher()

# =====================
# Utilities
# =====================
def save_obs_to_csv(obs: np.ndarray, filename: str = "obs_mujoco.csv") -> None:
    """Save 3D observation array to a CSV file (flattened per row)."""
    obs_np = np.array(obs)
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    np.savetxt(filename, obs_flat, delimiter=",")


# =====================
# Core Runner
# =====================

def run(args: argparse.Namespace) -> None:
    logger = get_logger("runner")

    # Controller selection
    if args.jump:
        controller = JumpController()
    elif args.controller == "velocity":
        controller = VelocityController(vel_step=VEL_STEP, max_lin_x=args.max_vel, max_lin_y=0.5, max_ang=1.0)
    elif args.controller == "remote":
        controller = SbusVelocityController(
            vmax_lin_x=args.max_vel,
            vmax_lin_y=0.5,
            vmax_ang=1.0,   # rad/s
            deadzone=0.05,
            invert_left_vertical=False,
            invert_right_vertical=False,
            invert_left_left_right=True,
            invert_right_left_right=True,
        )
    else:
        raise ValueError(f"Unknown controller: {args.controller}")

    # Robot initialization
    if args.sim:
        robot = SimRobot(args.xml)
    else:
        robot = Param()
        robot.start()
        time.sleep(1)

    viewer = None
    if args.sim:
        viewer = launch_passive(robot.model, robot.data)

    quadraped_init = QuadrupedDefaultInitializer(
        robot=robot,
        stand_gait=np.array(HOME_POSE),
        sit_gait=np.zeros(12),
        viewer=viewer
    )

    if not args.recovery:
        quadraped_init.sit()
        time.sleep(0.5)

        print("Standing Up!")
        quadraped_init.stand()

        for _ in tqdm(range(2), desc="Preparing", unit="s"):
            time.sleep(1)

    if args.recovery and args.sim:
        robot.fall_start(58)
        viewer.sync()



    def run_loop(viewer=None) -> None:
        with term.cbreak(), term.hidden_cursor():
            for step_idx in range(int(50 * 24 * 60 * 60)):
                t_start = time.time()

                # Keyboard → controller (still supports runner-level quit)
                key = term.inkey(timeout=0.001)
                key_str = str(key) if key else None
                if key_str in ("q", "Q"):
                    break
                if key_str:
                    try:
                        controller.handle_event(key_str)
                    except (AttributeError, NotImplementedError):
                        pass

                # One policy tick: control ingest + obs + act + apply
                out, obs, ctrl = policy.tick(control_timeout_ms=1)

                # Honor controller-level quit (e.g., SBUS button)
                # if getattr(out, "events", None) and out.events.get("quit", False):
                #     break

                # Telemetry
                qpos = policy.last_signals["qpos"]
                qvel = policy.last_signals["qvel"]
                imu_quat = policy.last_signals["imu_quat"]
                gyro = policy.last_signals["gyro"]
                rpy = R.from_quat(imu_quat, scalar_first=True).as_euler('xyz', degrees=True)

                motor_torques = robot.get_motor_torques()
                motor_temps = robot.get_motor_temps()
                #for k, v in motor_torques.items():
                #    print(f"{k}: {v:.2f}")
                #print()
                

                pub.send({
                    **motor_torques,
                    **motor_temps,
                    "GYRO_X": float(gyro[0]),
                    "GYRO_Y": float(gyro[1]),
                    "GYRO_Z": float(gyro[2]),
                    "Roll": float(rpy[0]),
                    "Pitch": float(rpy[1]),
                    "Yaw": float(rpy[2]),
                })

                # Logging
                logger.log({
                    "qpos": qpos,
                    "qvel": qvel,
                    "imu_quat": imu_quat,
                    "gyro": gyro,
                    "ctrl": ctrl,
                    "command": np.array([
                        float(out.axes.get("vx", 0.0)) if getattr(out, "axes", None) else 0.0,
                        float(out.axes.get("vy", 0.0)) if getattr(out, "axes", None) else 0.0,
                        float(out.axes.get("w",  0.0)) if getattr(out, "axes", None) else 0.0,
                    ], dtype=np.float32),
                })

                # Sim advance
                if viewer is not None:
                    robot.step(nsteps=4)
                    viewer.sync()

                # Timing
                elapsed = time.time() - t_start
                delay = DEFAULT_DT - elapsed
                if delay > 0:
                    time.sleep(delay)
                else:
                    print(f"Step time {elapsed:.4f}s exceeded {DEFAULT_DT}s")

    try:
        if args.recovery:
            policy_path = f"{sys.path[0]}/policy/{args.recovery_policy}"
            policy = RecoveryPolicyAgent(
                controller=None,
                robot=robot,
                onnx_path=policy_path,
                default_qpos=HOME_POSE,
            )
        else:
            if args.jump:
                policy_path = f"{sys.path[0]}/policy/{args.jump_policy}"
                policy = JumpPolicyAgent(
                    controller=controller,
                    robot=robot,
                    onnx_path=policy_path,
                    default_qpos=HOME_POSE,
                )
            else:
                policy_path = f"{sys.path[0]}/policy/{args.policy}"
                policy = VelocityPolicyAgent(
                    controller=controller,
                    robot=robot,
                    onnx_path=policy_path, 
                    default_qpos=HOME_POSE,
                    gait_freq=args.freq
                )

        obs_buffer = []
    
        if args.sim:
            run_loop(viewer)
        else:
            run_loop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Sitting Down!")
        quadraped_init.sit()
        time.sleep(1)
        raise
    except Exception as e:
        print("Exception occurred:", e)
        print("Sitting Down!")
        quadraped_init.sit()
        time.sleep(1)
        raise
    finally:
        quadraped_init.sit()
        time.sleep(1)
        # Example: persist observations if desired
        if args.save_obs:
            try:
                save_obs_to_csv(np.array(obs_buffer), filename=args.save_obs)
                print(f"Saved observations to {args.save_obs}")
            except Exception as e:
                print("Failed to save observations:", e)


# =====================
# Entrypoint
# =====================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA Quadruped runner")
    parser.add_argument("--sim", action="store_true", help="Run in simulation (MuJoCo)")
    parser.add_argument("--controller", choices=["velocity", "remote"], default="velocity", help="Controller type")
    parser.add_argument("--xml", type=str, default="/home/radon12/Documents/workspace/mjx_rl/src/mjx_rl/assets/param/param_scene_full_collisions.xml", help="MuJoCo XML path (sim only)")
    parser.add_argument("--policy", type=str, default=f"low_height_2k.onnx", help="Policy ONNX path")
    parser.add_argument("--save-obs", dest="save_obs", type=str, default="", help="Optional CSV file to save observations")
    parser.add_argument("--recovery", action="store_true", help="Enable recovery behavior")
    parser.add_argument("--recovery_policy", type=str, default="recovery_1k.onnx", help="Recovery policy ONNX path")
    parser.add_argument("--jump", action="store_true", help="Use jump controller (overrides --controller)")
    parser.add_argument("--jump_policy", type=str, default="jump_command_5000.onnx", help="Jump policy ONNX path")
    parser.add_argument("--freq",type=float,default=1.25,help="Gait frequency")
    parser.add_argument("--max_vel",type=float,default=1.0,help="Max linear velocity")
    return parser.parse_args(argv)


if __name__ == "__main__":
    from ga_can.core.ga_logging import logging_session
    with logging_session("param") as _session:
        run(parse_args())
