# Disable JAX prealloc
import os
from pathlib import Path
from functools import partial

# disable JAX preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Standard Library
import argparse
import time

# Third Party
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from jax import Array, lax
from scipy.spatial.transform import Rotation as R


def newton_raphson(f, x0, iters):
    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = lax.scan(update, x0, length=iters)
    return x


def roberts_sequence(num_points, dim, root):
    basis = 1 - (1 / root ** (1 + jnp.arange(dim)))
    n = jnp.arange(num_points)
    x = n[:, None] * basis[None, :]
    x, _ = jnp.modf(x)
    return x


class PyrokiIkBeamHelper:
    def __init__(self, visualize: bool = False):
        # Load URDF and build robot
        urdf_path_r = Path(__file__).parent.parent / "assets" / "quadruped_single_leg.urdf"
        mesh_dir = Path(__file__).parent.parent / "assets" / "meshes"

        urdf = yourdfpy.URDF.load(
            urdf_path_r.expanduser(),
            mesh_dir=mesh_dir.expanduser(),
        )

        self.robot_r = pk.Robot.from_urdf(urdf)

        ee_link = "RR_foot"

        self.target_link_index = jnp.array(self.robot_r.links.names.index(ee_link))

        # find root for quasi-random seeds
        exp = self.robot_r.joints.num_actuated_joints
        self.root = newton_raphson(lambda x: x ** (exp + 1) - x - 1, 1.0, 10_000)

    def solve_ik(
        self,
        robot,
        target_wxyz: jax.Array,
        target_position: jax.Array,
        prev_jnt_angles: jax.Array,
    ) -> jax.Array:
        """
        Solve IK for a single target using previous joint angles as smoothness prior.
        - target_wxyz: (4,) quaternion [w,x,y,z]
        - target_position: (3,)
        - prev_jnt_angles: (n_joints,)
        Returns: (n_joints,)
        """

        n_act = robot.joints.num_actuated_joints
        num_seeds_init, num_seeds_final = 64, 4
        total_steps, init_steps = 16, 6

        @jaxls.Cost.create_factory
        def smoothness_cost(
            vals: jaxls.VarValues,
            curr_joint_var: jaxls.Var[jnp.ndarray],
            past_joint_var: jax.Array,
            weight: Array | float,
        ) -> Array:
            """Computes the residual penalizing joint configuration differences (velocity)."""
            return ((vals[curr_joint_var] - past_joint_var) * weight).flatten()

        def solve_one(init_q, lambda_init, max_iters):
            joint_var = robot.joint_var_cls(0)
            pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz), target_position
            )
            # Set per-joint smoothness weights
            smoothness_weights = jnp.array([0.1, 0.1, 0.1, 0.0001])
            factors = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                pose,
                self.target_link_index,
                pos_weight=50.0,
                ori_weight=10.0,
            ),
            pk.costs.limit_cost(robot, joint_var, weight=100.0),
            # pk.costs.manipulability_cost(
            #     robot, joint_var, self.target_link_index, weight=0.5
            # ),
            smoothness_cost(joint_var, prev_jnt_angles, weight=smoothness_weights),
            ]
            sol, summary = (
                jaxls.LeastSquaresProblem(factors, [joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(init_q)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    termination=jaxls.TerminationConfig(
                        max_iterations=max_iters,
                        early_termination=False,
                    ),
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_init),
                    return_summary=True,
                )
            )
            return sol[joint_var], summary

        # vmap over init_q and lambda, broadcast prev_jnt_angles
        vmapped = jax.vmap(solve_one, in_axes=(0, 0, None))

        # generate quasi-random seeds
        lower, upper = (
            robot.joints.lower_limits,
            robot.joints.upper_limits,
        )
        seeds = lower + roberts_sequence(num_seeds_init, n_act, self.root) * (
            upper - lower
        )

        # initial optimization
        init_sols, sum_init = vmapped(
            seeds,
            jnp.full((num_seeds_init,), 10.0),
            init_steps,
        )
        # select best
        best_idx = jnp.argsort(sum_init.cost_history[:, -1])[:num_seeds_final]
        best_inits = init_sols[best_idx]
        best_lams = sum_init.lambda_history[best_idx, -1]

        # final optimization
        final_sols, sum_final = vmapped(
            best_inits,
            best_lams,
            total_steps - init_steps,
        )
        costs = sum_final.cost_history[
            jnp.arange(num_seeds_final), sum_final.iterations
        ]
        return final_sols[jnp.argmin(costs)]

    def forward_kinematics(self, q: jax.Array | np.ndarray, robot) -> jax.Array:
        return robot.forward_kinematics(jnp.asarray(q))[self.target_link_index]


ik_beam = PyrokiIkBeamHelper(visualize=True)
single_ik_right = jax.jit(partial(ik_beam.solve_ik, robot = ik_beam.robot_r))
single_fk_right = jax.jit(partial(ik_beam.forward_kinematics, robot = ik_beam.robot_r))

# batched_ik = jax.jit(jax.vmap(ik_beam.solve_ik, in_axes=(0, 0, 0, 0)))
# batched_fk = jax.jit(jax.vmap(ik_beam.forward_kinematics, in_axes=(0, 0)))

def batched_ik(quat, pos, prev_jnt_angles):
    # Pre-compiled for each robot

    outs = []
    out = single_ik_right(
        target_wxyz=quat[0],
        target_position=pos[0],
        prev_jnt_angles=prev_jnt_angles[0]
    )
    outs.append(out)
    return jnp.stack(outs)

def batched_fk(angles):
    # Pre-compiled for each robot
    outs = []
    out = single_fk_right(angles[0])
    outs.append(out)
    return jnp.stack(outs)

def _get_poses_and_quats(mats: np.ndarray):
    positions = mats[:, :3, 3]
    rot = mats[:, :3, :3]
    r = R.from_matrix(rot)
    xyzw = r.as_quat()
    quats = np.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], axis=1)
    return positions, quats


if __name__ == "__main__":
    
    np.set_printoptions(precision=3, suppress=True)

    # example batch of two poses
    pose1 = np.eye(4, dtype=np.float32)
    pose2 = pose1.copy()
    pose2[0, 3] += 0.1
    mats = np.stack([pose1, pose2], axis=0)

    positions, quaternions = _get_poses_and_quats(mats)
    positions_jnp = jnp.array(positions)
    quaternions_jnp = jnp.array(quaternions)

    # single previous-joint guess (will be broadcast)
    n_joints = ik_beam.robot.joints.num_actuated_joints
    prev_guess = jnp.zeros((2,n_joints,), dtype=jnp.float32)

    sol_batch = batched_ik(quaternions_jnp, positions_jnp, prev_guess)
    print("Batched IK solutions shape:", sol_batch.shape)
    print(sol_batch)

    # single-pose call (same prev_guess)
    sol_single = ik_beam.solve_ik(
        target_wxyz=quaternions_jnp[0],
        target_position=positions_jnp[0],
        prev_jnt_angles=prev_guess,
    )
    print("Single IK solution:", sol_single)