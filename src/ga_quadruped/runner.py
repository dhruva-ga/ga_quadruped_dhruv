from tqdm import tqdm
# from ga_quadruped.go1.go_one import GoOne
from ga_can.core.logger import log_session
from ga_quadruped.param.param import Param
from ga_quadruped.policy_agent import PolicyAgent
from ga_quadruped.sim2sim.robot import Robot

import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

# import cv2
import sys
# import csv
from blessed import Terminal

# logger = log_session.get_logger("param")

ACTUATOR_NAMES = [
    "hip_abduction_fl","thigh_rotation_fl","calf_fl",
    "hip_abduction_fr","thigh_rotation_fr","calf_fr",
    "hip_abduction_rl","thigh_rotation_rl","calf_rl",
    "hip_abduction_rr","thigh_rotation_rr","calf_rr"
]

def actuator_torque_logger(csv_path, actuator_names, nu, flush_every=100):
    """
    Creates a CSV logger for MuJoCo actuator efforts (data.actuator_force).
    Returns (log_fn, close_fn).
    - log_fn(data, step): write a row for the current step.
    - close_fn(): flush and close the CSV file.
    """
    if nu != len(actuator_names):
        raise ValueError(f"Model has nu={nu} actuators, but {len(actuator_names)} names were provided.")
    
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=["step", "sim_time"] + actuator_names) 
    writer.writeheader()
    count = 0

    def log(data, step):
        nonlocal count
        tau = data.actuator_force  # per-actuator effort
        row = {"step": int(step), "sim_time": float(data.time)}
        row.update({name: float(tau[j]) for j, name in enumerate(actuator_names)})
        writer.writerow(row)
        count += 1
        if flush_every and (count % flush_every == 0):
            f.flush()

    def close():
        f.flush()
        f.close()

    return log, close

def save_obs_to_csv(obs, filename="obs_mujoco.csv"):
    """Save 3D observation array to a CSV file (flattened per row)."""
    obs_np = np.array(obs)
    print(obs_np.shape)
    # Flatten each observation to 1D if needed
    obs_flat = obs_np.reshape(obs_np.shape[0], -1)
    np.savetxt(filename, obs_flat, delimiter=",")

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def limit_effort(effort, effort_limit):
    """Limit the effort to the specified limit."""
    return np.clip(effort, -effort_limit, effort_limit)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help='Sim or real robot')
    args = parser.parse_args()
                        
    time_step = 0.02
    
    # home_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5]
    theta = 0.4
    theta2 = 1.2
    home_pos = [-0.0, -theta, theta2, 0.0, theta, -theta2, 0.0, -theta, theta2, -0.0, theta, -theta2]
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(sys.path[0] + "/output.mp4", fourcc, 50, (1280,720))
    # CAMERA_NAME = "render_cam"  # Name of the camera in the XML file
    if args.sim:
        import mujoco

        XML_PATH = '/home/radon12/Documents/ga_quadruped/assets/param/scene.xml'
        robot = Robot(XML_PATH, randomisation=False, default_joint_pos=home_pos, init_pos=[0, 0, 0.45]) # Go1
    else:
        robot = Param()
        robot.start()
        time.sleep(1)
        robot._sit()
        time.sleep(2)
        print("Standing Up!")
        time.sleep(1)
        robot._stand()

        for _ in tqdm(range(5), desc="Preparing", unit="s"):
            time.sleep(1)
        
    ONNX_PATH = sys.path[0] + '/policy/param_low_com.onnx'
    
    term = Terminal()


    # try:
    #     cam_id = mujoco.mj_name2id(
    #         robot.model,
    #         mujoco.mjtObj.mjOBJ_CAMERA,
    #         CAMERA_NAME
    #     )
    # except Exception as e:
    #     raise RuntimeError(
    #         f"Camera '{CAMERA_NAME}' not found in the XML. "
    #     ) from e
    # ctx = mujoco.GLContext(1280, 720)
    # ctx.make_current()

    # renderer = mujoco.Renderer(robot.model,height=480, width=640)
    policy = PolicyAgent(ONNX_PATH, initial_qpos=home_pos)

    obs_arr = []

    # log_torque, close_torque = actuator_torque_logger(
    # "actuator_torques.csv", ACTUATOR_NAMES, robot.model.nu, flush_every=100
    # )
    

    def run_loop(viewer=None):
        vx, vy,w = 0.0, 0.0,0.0
        VEL_STEP = 0.1 

        gyro_integral = np.zeros(3)

        with term.cbreak(), term.hidden_cursor():
            val = ''
            for ixxxx in range(50*240):
                if val.lower() == 'q':
                    break
                t1 = time.time()
                val = term.inkey(timeout=0.001)
                if val == 'w':
                    vx += VEL_STEP
                elif val == 's':
                    vx -= VEL_STEP
                elif val == 'a':
                    vy += VEL_STEP
                elif val == 'd':
                    vy -= VEL_STEP
                elif val == 'g':
                    w += VEL_STEP
                elif val == 'h':
                    w -= VEL_STEP
                elif val == 't':
                    vx, vy, w = 0.0, 0.0, 0.0

                print("coomand", vx, vy, w)

                command = np.array([vx, vy, w], dtype=np.float32)
                gait_command = np.array([1.5, 0.5, 0.5, 0.5, 0.0])
                phase = np.remainder(time_step * gait_command[0] * ixxxx, 1.0)
                phase = 2 * np.pi * phase
                gait_phase = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
                policy.set_command(command)
                policy.set_gait_command(gait_command)

                if args.sim:
                    qpos = robot.get_position().copy()
                    qvel = robot.get_velocity().copy()
                    imu_quat = robot.get_imu_quat()
                    gyro = robot.get_gyro().copy()

                    # gyro = gyro  + np.random.uniform(-0.3, 0.3, size=3)  # add noise
                else:
                    kinematics_data = robot.get_kinematics_data()
                    imu_data = robot.get_imu_data()

                    qpos = kinematics_data.angles
                    qvel = kinematics_data.velocity
                    imu_quat = imu_data.quat
                    gyro = imu_data.gyro
                    rpy = imu_data.rpy


                obs, z_axis = policy.compute_obs(qpos, qvel, None, imu_quat, None, gyro, gait_phase)
                print("Obs:", obs)
                obs_arr.append(obs)

                gyro_integral += gyro * time_step * 180 / np.pi


                if args.sim:
                    print("current control", robot.data.ctrl)
                else:
                    print("current control", robot.get_ctrl())
                ctrl = policy.act(obs)
                print("setting control", ctrl)
                # robot.set_ctrl(np.array(home_pos))
                robot.set_ctrl(ctrl)
                print("Gyro:", gyro)
                print("Gyro Integral:", gyro_integral)

                # logger.log({
                #     "qpos": qpos,
                #     "qvel": qvel,
                #     "imu_quat": imu_quat,
                #     "gyro": gyro,
                #     "gyro_integral": gyro_integral,
                #     "rpy": rpy * 180 / np.pi,
                #     "imu_quat_rpy": R.from_quat(imu_quat, scalar_first=True).as_euler('xyz') * 180 / np.pi,
                #     "ctrl": ctrl,
                #     "command": command,
                #     "gait_command": gait_command,
                #     "gait_phase": gait_phase,
                # })

                # Need to manually step in sim
                if viewer is not None:
                    robot.step(nsteps=4)
                    viewer.sync()

                t2 = time.time()
                if t2 - t1 < time_step:
                    time.sleep(time_step - (t2 - t1))
                else:
                    print(f"Step time {t2 - t1:.4f} exceeded {time_step}")

    if args.sim:
        from mujoco.viewer import launch_passive
        with launch_passive(robot.model, robot.data) as viewer:
            run_loop(viewer)
    else:
        run_loop()

    if not args.sim:
        print("Sitting Down!")
        robot._sit()
        time.sleep(1)
    
    # close_torque()
    # save_obs_to_csv(obs_arr)
# writer.release()