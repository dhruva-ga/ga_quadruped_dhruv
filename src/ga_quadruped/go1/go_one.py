import sys
sys.path.append("/home/unitree/GA/unitree_legged_sdk/lib/python/arm64")

import go1sdk
import numpy as np
import time
from dataclasses import dataclass
from threading import Lock,Thread
import multiprocessing as mp
from logging import debug
np.set_printoptions(precision=3, suppress=True)

from logging import debug

@dataclass
class IMUData:
    rpy: np.ndarray
    quat: np.ndarray
    gyro: np.ndarray

@dataclass
class KinematicsData:
    angles: np.ndarray
    velocity: np.ndarray

class GoOne:

    def __init__(self):
        self._stand_gait = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
        self._sit_gait = np.array([0.3, 1.2, -2.818, -0.3, 1.2, -2.818, 0.3, 1.2, -2.818, -0.3, 1.2, -2.818], dtype=np.float32)

        self.Kp = 25
        self.Kd = 0.5
        self._standing = False
        self._dt = 0.02
        self._step = 0
        self._last_time = time.time()
        self._ly = 0.0
        self._lx = 0.0
        self._rx = 0.0
        self.start_time = time.time()
        self.ctrl_lock = Lock()
        self.ctrl = None

    def set_ctrl(self, ctrl):
        with self.ctrl_lock:
            self.ctrl = ctrl.copy()
            if self.ctrl is not None:
                self._cmd[:] = self.ctrl
                # print(self._cmd[:])

    def read_contact(self):
        with self.contacter._state_lock:
            return self.contacter._st.contacts.copy()

    def get_imu_data(self):
        return IMUData(
            quat=np.array(self._state[34:]),
            gyro=np.array(self._state[3:6]),
	        rpy=np.array(self._state[0:3]),
        )
        
    def get_kinematics_data(self):
        output_angles = np.array(self._state[6:18])
        output_velocity = np.array(self._state[18:30])
        return KinematicsData(
            angles=output_angles,
            velocity=output_velocity
        )

    def start(self):
        # def worker_planner(vcmd):
        #     print("planner start")

        #     context = zmq.Context()
        #     socket = context.socket(zmq.SUB)
        #     socket.setsockopt(zmq.CONFLATE, 1)
        #     socket.connect("tcp://{0}:{1}".format("192.168.0.133","5555"))
        #     socket.setsockopt(zmq.SUBSCRIBE, "".encode())

        #     while True:
        #         msg = socket.recv().decode()
        #         # print(msg)
        #         [vx, yawv] = [float(x) for x in msg.split()]
        #         # print("planner_commands", vx, yawv)
        #         vcmd[:] = [vx, yawv]
        #         time.sleep(0.01)

        def worker_motor(sig, cmd, remote, state):
            print("motor start")
        
            go1 = go1sdk.Go1Controller(self.Kp, self.Kd)
            dt = 0.002
            tau_limits = np.array([23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55])

            step = 0
            st_time = time.time()
            while True:
                if sig.value == 2:
                    break
                
                data = go1.recv()
                debug("recieved data", data)
                remote[:] = data[:6]
                state[:] = data[6:]

                debug("command", cmd)
                go1.send(0x0A if sig.value==1 else 0x00, cmd[:])
                debug("done")

                taus = self.Kp * (np.array(cmd[:]) - np.array(state[6:18])) - self.Kd * np.array(state[18:30])
                high_tau_idx = np.where(np.abs(taus) > tau_limits)[0]
                if sig.value==1 and len(high_tau_idx) > 0:
                    print(f"tau warning idx:{high_tau_idx} tau:{taus[high_tau_idx]}")

                time.sleep(max(time.time()-st_time - dt, 0))
                st_time = time.time()
                step += 1
        
        def worker_run(ctrl_lock, cmd):
            print("run loop start")
            while True:
                t1 = time.time()
                with ctrl_lock:
                    # print("Acquired lock")
                    if self.ctrl is not None:
                        cmd[:] = self.ctrl
                t2 = time.time()
                if t2 - t1 < 0.01:
                    time.sleep(0.01 - (t2 - t1))
                # print("Setting ctrl:", self.ctrl)
                
        self._sig = mp.Value('i', 0)
        self._cmd = mp.Array('f', [0.0]*12)
        self._remote = mp.Array('f', [0]*6)
        self._state = mp.Array('f', [0.0]*38)
        self._vcmd = mp.Array('f', [0.0]*2)

        self._ctrl_worker = mp.Process(target=worker_motor, args=(self._sig, self._cmd, self._remote, self._state))
        self._ctrl_worker.start()
        # self._exec_worker = mp.Process(target=worker_run, args=(self.ctrl_lock, self._cmd))
        # self._exec_worker.start()
        # self._planner_worker = mp.Process(target=worker_planner, args=(self._vcmd,))
        # self._planner_worker.start()
        print("Go1 Interface started")
    
    def _stand(self):
        jnt_pos = np.array(self._state[6:18])
        for i in range(200):
            rate = min(i/200, 1)
            des_gait = jnt_pos * (1 - rate) + self._stand_gait * rate
            self._cmd[:] = des_gait
            if i == 0:
                self._sig.value = 1
            time.sleep(0.01)

    def _sit(self):
        jnt_pos = np.array(self._state[6:18])
        for i in range(200):
            rate = min(i/200, 1)
            des_gait = jnt_pos * (1 - rate) + self._sit_gait * rate
            self._cmd[:] = des_gait
            time.sleep(0.01)
        self._sig.value = 0
    
if __name__ == "__main__":
    # go_one = GoOne()
    # time.sleep(2)
    # go_one.start()
    # go_one._stand()
    # while True:
    #     time.sleep(1)

    go1 = go1sdk.Go1Controller(25, 0.5)
    # home = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    # go1.send(0x0A, home)
    go1.ping()





    
