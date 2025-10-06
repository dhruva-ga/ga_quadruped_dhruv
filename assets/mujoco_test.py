import mujoco
import mujoco.viewer
import numpy as np
import time

def print_joint_and_actuator_order(model: mujoco.MjModel):
    # --- Joint order (as stored in model) ---
    jtype_names = {
        mujoco.mjtJoint.mjJNT_FREE:  "free",
        mujoco.mjtJoint.mjJNT_BALL:  "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }
    qpos_dims = {  # how many qpos each joint contributes
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    dof_dims = {   # how many dof (qvel entries) each joint contributes
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }

    print("\n=== JOINT ORDER (njnt = {}, nq = {}, nv = {}) ===".format(model.njnt, model.nq, model.nv))
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jtype = model.jnt_type[j]
        jtype_str = jtype_names.get(jtype, f"type_{int(jtype)}")
        qposadr = model.jnt_qposadr[j]
        dofadr  = model.jnt_dofadr[j]
        qdim    = qpos_dims[jtype]
        ddim    = dof_dims[jtype]
        body_id = model.jnt_bodyid[j]
        body    = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{int(body_id)}"
        axis    = model.jnt_axis[j]  # meaningful for hinge/slide

        print(f"{j:2d}: {name:20s} "
              f"type={jtype_str:5s} "
              f"body={body:15s} "
              f"qpos[{qposadr}:{qposadr+qdim}] "
              f"dof[{dofadr}:{dofadr+ddim}] "
              f"axis={np.array2string(axis, precision=3, suppress_small=True)}")

    # --- Actuator order (as stored in model) ---
    trn_names = {
        mujoco.mjtTrn.mjTRN_JOINT:          "joint",
        mujoco.mjtTrn.mjTRN_JOINTINPARENT:  "joint_in_parent",
        mujoco.mjtTrn.mjTRN_SITE:           "site",
        mujoco.mjtTrn.mjTRN_BODY:           "body",
        mujoco.mjtTrn.mjTRN_TENDON:         "tendon",
    }

    print("\n=== ACTUATOR ORDER (nu = {}) ===".format(model.nu))
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
        trntype = model.actuator_trntype[a]
        trn_str = trn_names.get(trntype, f"trn_{int(trntype)}")
        trnid0, trnid1 = model.actuator_trnid[a]  # usually trnid0 is the primary id

        extra = ""
        # Resolve what this actuator drives
        if trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(trnid0)) or f"joint_{int(trnid0)}"
            dofadr = model.jnt_dofadr[int(trnid0)]
            extra = f"-> joint {jname} (dofadr {dofadr})"
        elif trntype == mujoco.mjtTrn.mjTRN_SITE:
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(trnid0)) or f"site_{int(trnid0)}"
            extra = f"-> site {sname}"
        elif trntype == mujoco.mjtTrn.mjTRN_BODY:
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(trnid0)) or f"body_{int(trnid0)}"
            extra = f"-> body {bname}"
        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, int(trnid0)) or f"tendon_{int(trnid0)}"
            extra = f"-> tendon {tname}"

        # control range (if limited)
        cr = model.actuator_ctrlrange[a]
        limited = bool(model.actuator_ctrllimited[a])
        ctrl_range = f" ctrlrange=({cr[0]:.3g},{cr[1]:.3g})" if limited else ""

        print(f"{a:2d}: {aname:20s} trn={trn_str:14s} {extra}{ctrl_range} gear={np.array2string(model.actuator_gear[a], precision=3)}")

def main():
    model = mujoco.MjModel.from_xml_path("./go2/scene.xml")
    data = mujoco.MjData(model)

    # Print orders once
    print_joint_and_actuator_order(model)

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    hip_idx = [7, 10, 13, 16]
    thigh_idx = [8, 14, 11, 17]
    calf_idx = [9, 15, 18, 12]
    steps = 1000
    settle_steps = 500
    thigh_inter = np.linspace(0, -0.6, steps)
    calf_inter = np.linspace(0, 1.2, steps)

    i = 0
    while viewer.is_running():
        # if i > settle_steps:
        #     if i-settle_steps < steps:
        #         data.qpos[hip_idx] = 0
        #         data.qpos[thigh_idx] = thigh_inter[i-settle_steps]
        #         data.qpos[calf_idx] = calf_inter[i-settle_steps]
        #     else:
        #         data.qpos[hip_idx] = 0
        #         data.qpos[thigh_idx] = -0.6
        #         data.qpos[calf_idx] = 1.2
        # else:
        #     for j in range(7,19):
        #         data.qpos[j] = 0
        # i+=1
        # print(i)
        mujoco.mj_step(model, data)
        viewer.sync()
        # time.sleep(0.01)

if __name__ == "__main__":
    main()
