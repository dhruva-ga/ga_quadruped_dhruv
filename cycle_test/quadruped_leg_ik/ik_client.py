#!/usr/bin/env python3
from logging import debug
import zmq
import numpy as np
import time


class IKSolverClient:
    def __init__(self, ip: str = "localhost", port: int = 8120):
        self.address = f"tcp://{ip}:{port}"
        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(self.address)
        print(f"[IKSolverClient] Connected to {self.address}")

    def solve(self, prev_jnt_angles: np.ndarray, poses: np.ndarray) -> np.ndarray:
        self.socket.send_pyobj({"method": "solve","matrices": poses, "prev": prev_jnt_angles})
        sols = self.socket.recv_pyobj()
        if sols is None:
            raise RuntimeError("IK solver returned None, possibly an error occurred.")
        return np.asarray(sols, dtype=np.float32)
    
    def fk(self, angles: np.ndarray) -> np.ndarray:
        self.socket.send_pyobj({"method": "fk", "angles": angles})
        mats = self.socket.recv_pyobj()
        if mats is None:
            raise RuntimeError("FK solver returned None, possibly an error occurred.")
        return np.asarray(mats, dtype=np.float32)

if __name__ == "__main__":
    client = IKSolverClient(ip="localhost", port=8120)

    # quick IK test...
    total_elapsed = 0.0
    num_iters = 10
    for i in range(num_iters):
        batch_size, n_joints = 2, 6
        poses = np.array([np.eye(4, dtype=np.float32) for _ in range(batch_size)])
        prevs = np.zeros((batch_size, n_joints), dtype=np.float32)
        start = time.time()
        sols = client.solve(prevs, poses)
        total_elapsed += time.time() - start
        print("IK sols:", sols)
    print(f"Avg IK latency: {total_elapsed/num_iters:.6f}s")

    for i in range(num_iters):
        angles = np.zeros((batch_size, n_joints), dtype=np.float32)
        start = time.time()
        mats = client.fk(angles)
        print(mats.shape)
        total_elapsed += time.time() - start
        print("FK mats:", mats[0,:3,3])
    print(f"Avg FK latency: {total_elapsed/num_iters:.6f}s")
    jnt_angles  = np.array([0.0,0.8,-0.7,0,0,0]).reshape(1, -1).repeat(2,0)
    print(jnt_angles)
    print(jnt_angles.shape)

    mats = client.fk(jnt_angles)
    print(mats)
    init_angles = np.array([0,0,0,0,0,0]).reshape(1, -1).repeat(2,0)
    jnt_angles = client.solve(init_angles,mats)
    print("Sioln",jnt_angles)
