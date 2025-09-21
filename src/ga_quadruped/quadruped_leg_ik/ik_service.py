#!/usr/bin/env python3
import zmq
import numpy as np
import threading
import time
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

# replace this with your actual IK solver
from ik_utils import batched_fk, batched_ik


class IKSolverService:
    def __init__(self, port: int = 8120):
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"[IKSolverService] Bound to port {port}")

        self._warmup()
    def _warmup(self):
        mats = np.tile(np.eye(4, dtype=np.float32)[None], (5, 1, 1))
        pos, quat = self._get_pos_quat(mats)
        out = batched_ik(
            jnp.array(quat), jnp.array(pos), jnp.zeros((5, 4), dtype=jnp.float32)
        )
        print("[IKSolverService] Warm-up done, output shape:", np.asarray(out).shape)

    def _get_pos_quat(self, mats: np.ndarray):
        positions = mats[..., :3, 3]
        rot = R.from_matrix(mats[..., :3, :3].reshape(-1, 3, 3))
        xyzw = rot.as_quat()  # returns (x,y,z,w)
        quats = np.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], axis=1)
        batch = mats.shape[0]
        return positions.reshape(batch, 3), quats.reshape(batch, 4)
    
    def _get_mat(self, quatpos: np.ndarray):
        mats = np.eye(4, dtype=np.float32)[None].repeat(quatpos.shape[0], axis=0)
        mats[:, :3, 3] = quatpos[:, 4:7]  # set positions
        r = R.from_quat(quatpos[:, :4],scalar_first=True)  # convert from (x,y,z,w) to rotation matrix
        mats[:, :3, :3] = r.as_matrix()
        mats[:, 3, 3] = 1.0
        return mats
    
    def solve_ik(self, prev_jnt_angles: np.ndarray, poses: np.ndarray) -> np.ndarray:
        pos_np, quat_np = self._get_pos_quat(poses)

        # compute IK
        start = time.time()
        sols_jnp = batched_ik(
            jnp.array(quat_np), jnp.array(pos_np), jnp.array(prev_jnt_angles)
        )
        sols = np.asarray(sols_jnp, dtype=np.float32)
        return sols

    def run(self):
        print("[IKSolverService] Running...")
        while True:
            try:
                # Wait for first frame
                packet = self.socket.recv_pyobj()
                method = packet.get("method", "solve")
                start_time = time.time()
                if method == 'solve':
                    sols = self.solve_ik(packet["prev"], packet["matrices"])
                    elapsed = time.time() - start_time
                    print(f"[IKSolverService] Solved IK in {elapsed:.4f}s")
                    self.socket.send_pyobj(sols)
                elif method == 'fk':
                    angles = packet["angles"]
                    sols = batched_fk(jnp.array(angles))
                    sols = self._get_mat(np.asarray(sols, dtype=np.float32))
                    elapsed = time.time() - start_time
                    print(f"[IKSolverService] Solved FK in {elapsed:.4f}s")
                    self.socket.send_pyobj(sols)
                else:
                    print(f"[IKSolverService] Unknown method: {method}")
                    self.socket.send_pyobj(None)
            except Exception as e:
                print("[IKSolverService] ERROR:", e)
                # on error, reply zeros
                self.socket.send_pyobj(None)


if __name__ == "__main__":
    svc = IKSolverService(port=8120)
    svc.run()