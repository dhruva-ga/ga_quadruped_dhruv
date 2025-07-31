import mujoco
from mujoco.viewer import launch_passive
import numpy as np
import csv

from force_logger import init_combined_logger, log_combined


model = mujoco.MjModel.from_xml_path("mjcf/reformatted_quadruped_mjcf.xml")
data  = mujoco.MjData(model)

# Get the integer ID of each joint# List all of your actuator names once
ACTUATOR_NAMES = [
    "hip_rr","hip_rl","hip_fr","hip_fl",
    "thigh_rr","thigh_rl","thigh_fr","thigh_fl",
    "calf_rr","calf_rl","calf_fr","calf_fl",
]

# Build a dict name → id
act_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    for name in ACTUATOR_NAMES
}


POSES = {
    "stand": np.zeros(len(ACTUATOR_NAMES)),
    "sit":   np.array([ 
        0.0,0.0,0.0,0.0,   # hips
        -90,-90,-90,-90,   # thighs
        105,105,105,105 # calves
    ])*np.pi/180.0,
    "trot1": np.array([ 
        0.0,0.0,0.0,0.0,   # hips
        -60,0,0,-60,   # thighs
        85,-50,-50,85 # calves
    ])*np.pi/180.0,
    "trot2":  np.array([    
        0.0,0.0,0.0,0.0,   # hips
        0,-60,-60,0,   # thighs
        -50,85,85,-50 # calves
    ])*np.pi/180.0,
    "stance":  np.array([    
        0.0,0.0,0.0,0.0,   # hips
        0,-60,-60,0,   # thighs
        -50,85,85,-50 # calves
    ])*np.pi/180.0,    
    "gallop1": np.array([ 
        0.0,0.0,0.0,0.0,   # hips
        -30,-30,-30,-30,   # thighs
        90,90,90,90 # calves
    ])*np.pi/180.0,
    "gallop2":  np.array([ 
        0.0,0.0,0.0,0.0,   # hips
        90,90,90,90,   # thighs
        -30,-30,-30,-30 # calves
    ])*np.pi/180.0,
    # add more poses here
}

def apply_pose(name: str):
    targets = POSES[name]
    for nm, tgt in zip(ACTUATOR_NAMES, targets):
        data.ctrl[act_ids[nm]] = tgt

# initialize logger once
log_file, csv_writer = init_combined_logger("logged_data.csv", ACTUATOR_NAMES)


hold_steps  = int( 1/ model.opt.timestep)
step_counter = 0
curr_idx = 0

POSE_SEQUENCE = ["stand","sit"] #["trot1","trot2"]   
sequence_len  = len(POSE_SEQUENCE)

curr_pose = "stand"

desired_rpm = 30.0
# convert to rad/sec
omega = desired_rpm * 2 * np.pi / 60.0
# per-step increment (rad per physics step)
delta = omega * model.opt.timestep  # simulation timestep

# Address into data.qpos for each actuator’s joint
qpos_addrs = [
    model.jnt_qposadr[ mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) ]
    for name in ACTUATOR_NAMES
]

virtual_targets = np.array([ data.qpos[addr] for addr in qpos_addrs ])


with launch_passive(model, data) as viewer:
    while viewer.is_running():

        if step_counter and step_counter % hold_steps == 0:
            curr_idx = (curr_idx + 1) % sequence_len


        curr_pose = POSE_SEQUENCE[curr_idx]
        desired_angles = POSES[curr_pose]

        for i in range(len(virtual_targets)):
            if virtual_targets[i] < desired_angles[i]:
                virtual_targets[i] = min(virtual_targets[i] + delta, desired_angles[i])
            else:
                virtual_targets[i] = max(virtual_targets[i] - delta, desired_angles[i])
                # handles the case desired < start for negative motion

        # 2) Write those targets into your position actuators
        for name, vt in zip(ACTUATOR_NAMES, virtual_targets):
            data.ctrl[ act_ids[name] ] = vt
        
        mujoco.mj_step(model, data)
        viewer.sync()
        log_combined(csv_writer, data, ACTUATOR_NAMES, act_ids,joint_ids=act_ids)

        step_counter += 1

# close when done
log_file.close()