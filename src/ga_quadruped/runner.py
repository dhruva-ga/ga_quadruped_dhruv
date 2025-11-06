"""
Finite State Machine (FSM) runner for GA quadruped.

Key changes vs. the original script:
- Clear state machine with runtime policy switching (Velocity ⟷ Jump ⟷ Recovery)
- Safer lifecycle (on_enter/on_exit), unified telemetry, and clean shutdown
- Same CLI surface with small additions to key bindings
- Keeps support for simulation (MuJoCo) and hardware Param robot

Default key bindings (terminal):
  q      : quit & sit
  1 / v  : Velocity policy
  2 / j  : Jump policy
  3 / r  : Recovery policy

Automatic transition:
  If roll/pitch magnitude exceeds SWAY_THRESHOLD_DEG, transition → Recovery

Assumes project-local modules are importable.
"""
from __future__ import annotations

import argparse
import sys
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from blessed import Terminal
from scipy.spatial.transform import Rotation as R

from ga_can.core.ga_logging import get_logger
from ga_quadruped.controller.accelerate_controller import (  # noqa: F401 (discoverability)
    AccelerateController,
)
from ga_quadruped.controller.jump_controller import JumpController
from ga_quadruped.controller.push_controller import PushController
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
VEL_STEP = 0.1
DEFAULT_DT = 0.02
SWAY_THRESHOLD_DEG = 45.0

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
    obs_np = np.array(obs)
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    np.savetxt(filename, obs_flat, delimiter=",")


def policy_path(filename: str) -> str:
    return f"{sys.path[0]}/policy/{filename}"


# =====================
# FSM Plumbing
# =====================
class Event(Enum):
    TICK = auto()
    KEY = auto()
    QUIT = auto()
    SWAY = auto()


@dataclass
class Ctx:
    args: argparse.Namespace
    robot: Param | SimRobot
    viewer: Optional[object]
    quad_init: QuadrupedDefaultInitializer
    push_controller: Optional[PushController]
    logger: object
    obs_buffer: list = field(default_factory=list)


class State:
    name: str = "State"

    def on_enter(self, fsm: "FSM") -> None:
        pass

    def on_exit(self, fsm: "FSM") -> None:
        pass

    def handle(self, fsm: "FSM", ev: Event, data: Optional[object] = None) -> None:
        pass


class FSM:
    def __init__(self, ctx: Ctx, initial: State) -> None:
        self.ctx = ctx
        self.state: State = initial
        self._running = True
        self.prev_non_recovery: Optional[State] = None
        self.state.on_enter(self)

    def transition(self, new_state: State) -> None:
        if isinstance(self.state, RecoveryState) and not isinstance(new_state, RecoveryState):
            # leaving recovery → restore last non-recovery as the prev
            self.prev_non_recovery = new_state
        if not isinstance(self.state, RecoveryState) and not isinstance(new_state, RecoveryState):
            self.prev_non_recovery = new_state
        self.state.on_exit(self)
        self.state = new_state
        self.state.on_enter(self)
        logging.info({"transition": self.state.name})

    def stop(self) -> None:
        self._running = False

    def running(self) -> bool:
        return self._running

    # convenience for sway detection
    def maybe_sway(self, rpy_deg: Tuple[float, float, float]) -> None:
        roll, pitch, _ = rpy_deg
        if abs(roll) > SWAY_THRESHOLD_DEG or abs(pitch) > SWAY_THRESHOLD_DEG:
            logging.info({"event": "SWAY", "roll": float(roll), "pitch": float(pitch)})
            # Only auto-enter recovery if not already there
            if not isinstance(self.state, RecoveryState):
                self.transition(RecoveryState())


# =====================
# Concrete States
# =====================
class BootState(State):
    name = "Boot"

    def on_enter(self, fsm: FSM) -> None:
        args = fsm.ctx.args
        # If recovery+sim, optionally start in a fallen pose
        if args.recovery and args.sim:
            try:
                fsm.ctx.robot.fall_start(58)
                if fsm.ctx.viewer is not None:
                    fsm.ctx.viewer.sync()
            except Exception as e:
                logging.warning({"boot_fall_start_failed": str(e)})

        if args.recovery:
            fsm.transition(RecoveryState())
        else:
            fsm.transition(StandState())


class StandState(State):
    name = "Stand"

    def on_enter(self, fsm: FSM) -> None:
        qi = fsm.ctx.quad_init
        # Sit then stand like original pre-roll
        qi.sit()
        time.sleep(0.5)
        print("Standing Up!")
        qi.stand()
        # small prep delay
        for _ in range(2):
            time.sleep(1)
        # default to velocity policy unless user asked for jump via CLI
        if fsm.ctx.args.jump:
            fsm.transition(JumpState())
        else:
            fsm.transition(VelocityState())


# class SafetyStopState(State):
#     """
#     Safety interlock state reached when sway exceeds threshold.
#     Robot is commanded to sit and await a human decision:
#       - 's' → StandState (will perform sit→stand pre-roll)
#       - 'r' → RecoveryState (policy-led self-righting)
#       - 'q' → ShutdownState
#     """
#     name = "SafetyStop"

#     def on_enter(self, fsm: FSM) -> None:
#         print("SAFETY STOP: Excessive sway detected. Daming the motors and awaiting input (s=Stand, r=Recovery, q=Quit)…")
#         try:
#             fsm.ctx.robot.stop()
#         except Exception as e:
#             logging.warning({"safety_stop_sit_failed": str(e)})

#     def handle(self, fsm: FSM, ev: Event, data: Optional[object] = None) -> None:
#         key = str(data)
#         if ev == Event.KEY:
#             if key in ("q", "Q"):
#                 fsm.transition(ShutdownState())
#                 return
#             if key in ("s", "S"):
#                 fsm.transition(StandState())
#                 return
#             if key in ("r", "R", "3"):
#                 fsm.transition(RecoveryState())
#                 return
#             # Ignore other keys in safety stop
#         # No control outputs on tick; remain passive and safe
#         if ev == Event.TICK:
#             pass


def _key_common(fsm: FSM, key) -> None:
    # Simulation-specific push injection
    if fsm.ctx.push_controller is not None:
        print("Handling push controller event", key)
        fsm.ctx.push_controller.handle_event(key)
        out_push = fsm.ctx.push_controller.step()
        mag = out_push.axes.get("mag")
        if out_push.events.get("push", True):
            fsm.ctx.robot.push(mag=mag)


def _tick_common(fsm: FSM, agent) -> None:
    out, obs, ctrl = agent.tick(control_timeout_ms=1)
    # Gather telemetry + sway detection
    qpos = agent.last_signals["qpos"]
    qvel = agent.last_signals["qvel"]
    imu_quat = agent.last_signals["imu_quat"]
    gyro = agent.last_signals["gyro"]
    rpy = R.from_quat(imu_quat, scalar_first=True).as_euler("xyz", degrees=True)

    # Optionally honor controller-level quit (e.g. SBUS button)
    # try:
    #     if getattr(out, "events", None) and out.events.get("quit", False):
    #         fsm.transition(ShutdownState())
    #         return
    # except Exception:
    #     pass

    # Publish motor & IMU
    motor_torques = fsm.ctx.robot.get_motor_torques()
    motor_temps = fsm.ctx.robot.get_motor_temps()
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
    fsm.ctx.logger.log({
        "qpos": qpos,
        "qvel": qvel,
        "imu_quat": imu_quat,
        "gyro": gyro,
        "ctrl": ctrl,
    })

    # Auto recovery if swayed
    fsm.maybe_sway(tuple(map(float, rpy)))

class VelocityState(State):
    name = "Velocity"

    def __init__(self) -> None:
        self.controller = None
        self.agent = None

    def on_enter(self, fsm: FSM) -> None:
        args = fsm.ctx.args
        # Controller selection
        if args.controller == "velocity":
            self.controller = VelocityController(
                vel_step=VEL_STEP,
                max_lin_x=args.max_vel,
                max_lin_y=0.5,
                max_ang=1.0,
            )
        elif args.controller == "remote":
            self.controller = SbusVelocityController(
                vmax_lin_x=args.max_vel,
                vmax_lin_y=0.5,
                vmax_ang=1.0,
                deadzone=0.05,
                invert_left_vertical=False,
                invert_right_vertical=False,
                invert_left_left_right=True,
                invert_right_left_right=True,
            )
        else:
            raise ValueError(f"Unknown controller: {args.controller}")

        self.agent = VelocityPolicyAgent(
            controller=self.controller,
            robot=fsm.ctx.robot,
            onnx_path=policy_path(args.policy),
            default_qpos=HOME_POSE,
            gait_freq=args.freq,
        )

    def on_exit(self, fsm: FSM) -> None:
        # Nothing persistent to tear down
        pass

    def handle(self, fsm: FSM, ev: Event, data: Optional[object] = None) -> None:
        key = str(data)
        if ev == Event.KEY:
            if key in ("q", "Q"):
                fsm.transition(ShutdownState())
                return
            if key in ("2", "j", "J"):
                fsm.transition(JumpState())
                return
            if key in ("3", "r", "R"):
                fsm.transition(RecoveryState())
                return
            
            # forward other keys to the controller if it supports it
            try:
                self.controller.handle_event(key)
            except (AttributeError, NotImplementedError):
                pass
            
            print("Calling common key handler")
            _key_common(fsm, key)
        if ev == Event.TICK:
            _tick_common(fsm, self.agent)

    


class JumpState(State):
    name = "Jump"

    def __init__(self) -> None:
        self.controller = JumpController()
        self.agent = None

    def on_enter(self, fsm: FSM) -> None:
        args = fsm.ctx.args
        self.agent = JumpPolicyAgent(
            controller=self.controller,
            robot=fsm.ctx.robot,
            onnx_path=policy_path(args.jump_policy),
            default_qpos=HOME_POSE,
        )

    def handle(self, fsm: FSM, ev: Event, data: Optional[object] = None) -> None:
        key = str(data)
        if ev == Event.KEY:
            if key in ("q", "Q"):
                fsm.transition(ShutdownState())
                return
            if key in ("1", "v", "V"):
                fsm.transition(VelocityState())
                return
            if key in ("3", "r", "R"):
                fsm.transition(RecoveryState())
                return
            try:
                self.controller.handle_event(key)
            except (AttributeError, NotImplementedError):
                pass

            _key_common(fsm, key)
        if ev == Event.TICK:
            _tick_common(fsm, self.agent)


class RecoveryState(State):
    name = "Recovery"

    def __init__(self) -> None:
        self.agent = None

    def on_enter(self, fsm: FSM) -> None:
        args = fsm.ctx.args
        self.agent = RecoveryPolicyAgent(
            controller=None,
            robot=fsm.ctx.robot,
            onnx_path=policy_path(args.recovery_policy),
            default_qpos=HOME_POSE,
        )
        print("Entering recovery mode…")

    def handle(self, fsm: FSM, ev: Event, data: Optional[object] = None) -> None:
        key = str(data)
        if ev == Event.KEY:
            if key in ("q", "Q"):
                fsm.transition(ShutdownState())
                return
            if key in ("1", "v", "V"):
                fsm.transition(VelocityState())
                return
            if key in ("2", "j", "J"):
                fsm.transition(JumpState())
                return
            
            _key_common(fsm, key)

        if ev == Event.TICK:
            # _tick_common(fsm, self.agent)
            out, obs, ctrl = self.agent.tick(control_timeout_ms=1)
            imu_quat = self.agent.last_signals["imu_quat"]
            gyro = self.agent.last_signals["gyro"]
            rpy = R.from_quat(imu_quat, scalar_first=True).as_euler("xyz", degrees=True)

            # Publish minimal telemetry; avoid controller commands in recovery
            pub.send({
                "GYRO_X": float(gyro[0]),
                "GYRO_Y": float(gyro[1]),
                "GYRO_Z": float(gyro[2]),
                "Roll": float(rpy[0]),
                "Pitch": float(rpy[1]),
                "Yaw": float(rpy[2]),
            })

            # If we returned upright, bounce to last non-recovery or velocity as default
            if abs(rpy[0]) < SWAY_THRESHOLD_DEG * 0.5 and abs(rpy[1]) < SWAY_THRESHOLD_DEG * 0.5:
                next_state = StandState()
                fsm.transition(type(next_state)())


class ShutdownState(State):
    name = "Shutdown"

    def on_enter(self, fsm: FSM) -> None:
        print("Sitting down & stopping…")
        try:
            fsm.ctx.quad_init.sit()
            time.sleep(1)
        finally:
            fsm.stop()


# =====================
# Runner
# =====================

def run(args: argparse.Namespace) -> None:
    # Robot init
    if args.sim:
        robot = SimRobot(args.xml)
    else:
        robot = Param()
        robot.start()
        time.sleep(1)

    viewer = launch_passive(robot.model, robot.data) if args.sim else None
    quad_init = QuadrupedDefaultInitializer(
        robot=robot,
        stand_gait=np.array(HOME_POSE),
        sit_gait=np.zeros(12),
        viewer=viewer,
    )

    push_controller = PushController() if args.sim else None

    logger = get_logger("runner")

    ctx = Ctx(
        args=args,
        robot=robot,
        viewer=viewer,
        quad_init=quad_init,
        push_controller=push_controller,
        logger=logger,
        obs_buffer=[],
    )

    fsm = FSM(ctx, BootState())

    # Main loop
    with term.cbreak(), term.hidden_cursor():
        while fsm.running():
            t0 = time.time()

            key = term.inkey(timeout=0.001)
            if key:
                fsm.state.handle(fsm, Event.KEY, str(key))
                # allow immediate transition handling (e.g., quit)
                if not fsm.running():
                    break

            # one control/policy tick
            fsm.state.handle(fsm, Event.TICK)

            # Sim advance
            if ctx.viewer is not None and fsm.running():
                ctx.robot.step(nsteps=4)
                ctx.viewer.sync()

            # Timing control
            elapsed = time.time() - t0
            delay = DEFAULT_DT - elapsed
            if delay > 0:
                time.sleep(delay)
            else:
                print(f"Step time {elapsed:.4f}s exceeded {DEFAULT_DT}s")

    # Finalize / sit in finally-equivalent manner
    try:
        ctx.quad_init.sit()
        time.sleep(1)
    except Exception:
        pass


# =====================
# Entrypoint & CLI
# =====================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA Quadruped FSM runner")
    parser.add_argument("--sim", action="store_true", help="Run in simulation (MuJoCo)")
    parser.add_argument(
        "--controller", choices=["velocity", "remote"], default="velocity", help="Controller type"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="/home/radon12/Documents/workspace/mjx_rl/src/mjx_rl/assets/param/param_scene_full_collisions.xml",
        help="MuJoCo XML path (sim only)",
    )
    parser.add_argument("--policy", type=str, default="low_height_2k.onnx", help="Policy ONNX path")
    parser.add_argument("--recovery", action="store_true", help="Start directly in recovery mode")
    parser.add_argument("--recovery_policy", type=str, default="recovery_1k.onnx", help="Recovery policy ONNX path")
    parser.add_argument("--jump", action="store_true", help="Start directly in jump mode")
    parser.add_argument("--jump_policy", type=str, default="jump_command_5000.onnx", help="Jump policy ONNX path")
    parser.add_argument("--freq", type=float, default=1.25, help="Gait frequency")
    parser.add_argument("--max_vel", type=float, default=1.0, help="Max linear velocity")
    parser.add_argument("--save-obs", dest="save_obs", type=str, default="", help="Optional CSV to save observations")
    return parser.parse_args(argv)


if __name__ == "__main__":
    from ga_can.core.ga_logging import logging_session

    try:
        with logging_session("param") as _session:
            run(parse_args())
    except KeyboardInterrupt:
        print("KeyboardInterrupt: exiting…")
