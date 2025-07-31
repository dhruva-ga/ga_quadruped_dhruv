from urdf2mjcf import run

# Convert URDF → MJCF
run(
    urdf_path="urdf/quadruped.urdf",
    mjcf_path="mjcf/quadruped_mjcf.xml",
    copy_meshes=True,      # copy all mesh files alongside
)

# Then load in MuJoCo
import mujoco
model = mujoco.MjModel.from_xml_path("mjcf/quadruped_mjcf.xml")
data  = mujoco.MjData(model)
