from quadruped_leg_kinematics.param_helper import ParamLegs
import numpy as np
import time 
from param_logger import ParamLogger
import asyncio

LANDING_POSE = np.array([
    0.0,   
    0.7,   
    -1.3, 
])  # Default pose for the robot

HOME_POSE = np.array([
    0.0,   
    0.2,   
    -0.5, 
])

FLIGHT_POSE = np.array([
    0.0,   
    0.7,   
    -1.3, 
])

class QuadJumper:
    def __init__(self, logger):
        print("Starting Jumping Test...")
        self.logger = logger
        self.param = ParamLegs()
        self.param.start()
        time.sleep(2)        

        self.kinematics_data = self.param.get_kinematics_data()
        self.cycle_count = 0
        # diff = default_pose - qpos
        # for i in range(50): 
        #     ctrl =diff*(i/50) + qpos 

    async def start_cycle(self):
        while True:
            try:
                start_cycle = time.time()
                # At home
                ctrl = HOME_POSE
                self.param.set_ctrl(ctrl)

                await asyncio.sleep(1)

                # Jump
                ctrl = FLIGHT_POSE
                self.param.set_ctrl(ctrl)

                await asyncio.sleep(0.05)

                # Land
                ctrl = LANDING_POSE
                self.param.set_ctrl(ctrl)

                await asyncio.sleep(2)

                self.cycle_count+=1
                self.logger.log([self.cycle_count])
                print(f"Cycle: {self.cycle_count}, Time: {time.time() - start_cycle :.2f}")
            except Exception as e:
                print("Error while executing Jumping: ",e)
                break

    async def log_data(self):
        while True:
            kinematics_data = self.param.get_kinematics_data()
            self.logger.log(None, None, kinematics_data.angles, None, kinematics_data.torque, kinematics_data.motor_current, None, None)
            await asyncio.sleep(0.01)

async def main():
    logger = ParamLogger("jump_logs")
    jumper = QuadJumper(logger=logger)

    await asyncio.gather(
        jumper.start_cycle(),
        jumper.log_data()
    )
    

if __name__ == "__main__":
    asyncio.run(main())