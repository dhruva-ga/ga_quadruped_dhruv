from ga_quadruped.param.param import Param
import time

ga_one = Param()
time.sleep(2)
ga_one.start()
time.sleep(2)


while True:
    angles = ga_one.get_kinematics_data().angles
    print("Angles:", angles)
    time.sleep(0.5)