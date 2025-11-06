import numpy as np
from ga_quadruped.param.param import Param
from ga_quadruped.robot.quadruped_init import QuadrupedDefaultInitializer


if __name__ == "__main__":
    import time

    theta0 = 0.0
    # theta1 = 0.45
    theta1 = 0.4
    theta2 = 1.2
    # theta2 = 1.4
    HOME_POSE = [theta0, -theta1, theta2, -theta0, theta1, -theta2, theta0, -theta1, theta2, -theta0, theta1, -theta2]
    
    robot = Param()

    quadraped_init = QuadrupedDefaultInitializer(
        robot=robot,
        stand_gait=np.array(HOME_POSE),
        sit_gait=np.zeros(12),
    )

    
    robot.start()
    time.sleep(0.1)

    quadraped_init.sit()
    time.sleep(1.0)

    robot.stop()

    time.sleep(1000)
    
    quadraped_init.stand()

