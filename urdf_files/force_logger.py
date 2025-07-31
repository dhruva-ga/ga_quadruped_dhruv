import csv
import mujoco

def init_force_logger(filename: str, actuator_names: list[str]):
    """
    Opens (and returns) a CSV writer that’s already written the header for forces:
      time, <joint>_force, ...
    """
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    # build header: time + force columns
    header = ["time"] + [f"{name}_force" for name in actuator_names]
    writer.writerow(header)
    return f, writer


def init_angle_logger(filename: str, actuator_names: list[str]):
    """
    Opens (and returns) a CSV writer that’s already written the header for angles:
      time, <joint>_cmd, <joint>_act, ...
    """
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    # build header: time + cmd/act columns
    header = ["time"]
    for name in actuator_names:
        header += [f"{name}_cmd", f"{name}_act"]
    writer.writerow(header)
    return f, writer


def init_combined_logger(filename: str, actuator_names: list[str]):
    """
    Opens (and returns) a CSV writer that’s already written the header for combined data:
      time, <joint>_force, <joint>_cmd, <joint>_act, ...
    """
    f = open(filename, "w", newline="")
    writer = csv.writer(f)
    # build header: time + for each joint force, cmd and act
    header = ["time"]
    for name in actuator_names:
        header += [f"{name}_force", f"{name}_cmd", f"{name}_act"]
    writer.writerow(header)
    return f, writer


def log_joint_forces(writer, data: mujoco.MjData, actuator_names: list[str], joint_ids: dict[str,int]):
    """
    Append one row: [time, qfrc_actuator[jid] for each joint].
    """
    row = [f"{data.time:.4f}"]
    for name in actuator_names:
        jid = joint_ids[name]
        row.append(f"{data.qfrc_actuator[jid]:.6f}")
    writer.writerow(row)


def log_joint_angles(writer, data: mujoco.MjData,
                     actuator_names: list[str],
                     act_ids: dict[str,int],
                     joint_ids: dict[str,int]):
    """
    Append one row: [time, ctrl[aid], qpos[jid] for each joint].
    """
    row = [f"{data.time:.4f}"]
    for name in actuator_names:
        aid = act_ids[name]
        cmd = data.ctrl[aid]
        jid = joint_ids[name]
        act = data.qpos[jid]
        row += [f"{cmd:.6f}", f"{act:.6f}"]
    writer.writerow(row)


def log_combined(writer, data: mujoco.MjData,
                 actuator_names: list[str],
                 act_ids: dict[str,int],
                 joint_ids: dict[str,int]):
    """
    Append one row: [time, force, cmd, act] per joint.
    """
    row = [f"{data.time:.4f}"]
    for name in actuator_names:
        # force
        jid = joint_ids[name]
        force = data.qfrc_actuator[jid]
        # commanded
        aid = act_ids[name]
        cmd = data.ctrl[aid]
        # actual
        act = data.qpos[jid]
        row += [f"{force:.6f}", f"{cmd:.6f}", f"{act:.6f}"]
    writer.writerow(row)
