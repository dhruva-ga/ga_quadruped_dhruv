import mujoco
from mujoco.viewer import launch_passive

model = mujoco.MjModel.from_xml_path("task_flat.xml")
data  = mujoco.MjData(model)

with launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
