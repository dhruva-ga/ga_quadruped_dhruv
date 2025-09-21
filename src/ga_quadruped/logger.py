import os
import time
import numpy as np


class GaLogger:
    def __init__(self, log_folder):
        # Create unique session folder with timestamp
        self.log_folder = os.path.join("logs", log_folder, str(time.time()))
        self.artifact_folder = os.path.join(self.log_folder, "artifacts")
        os.makedirs(self.artifact_folder, exist_ok=True)
        self.timestamp_file = open(os.path.join(self.log_folder, "timestamp.txt"), "w")

    def log(self, data: dict, timestamp=None):
        """Log dictionary of name -> list or np.ndarray. Saves each as .npy using timestamp as filename."""
        if timestamp is None:
            timestamp = time.time()

        for name, values in data.items():
            arr = np.array(values)  # Convert lists to np.ndarray if needed
            subfolder = os.path.join(self.artifact_folder, name)
            os.makedirs(subfolder, exist_ok=True)
            filename = os.path.join(subfolder, f"{timestamp:.6f}.npy")
            np.save(filename, arr)
            os.sync()  # Ensures durability

        self.timestamp_file.write(f"{timestamp:.6f}\n")
        self.timestamp_file.flush()

    def close(self):
        self.timestamp_file.close()


if __name__ == "__main__":
    import numpy as np
    logger = GaLogger("param")
    for i in range(1000):
        ctrl = np.zeros(12)
        imu_rpy = [0.0, 0.0, 0.0]
        gyro = [0.0, 0.0, 0.0]
        accel = [0.0, 0.0, 9.81]
        position = [i, i, i]
        velocity = [i * 0.1, i * 0.1, i * 0.1]
        torque = [i * 0.01, i * 0.01, i * 0.01]
        t1 = time.time()
        logger.log({
            "ctrl": ctrl,
            "imu_rpy": imu_rpy,
            "gyro": gyro,
            "accel": accel,
            "position": position,
            "velocity": velocity,
            "torque": torque
        })
        t2 = time.time()
        print(f"Logged data {i+1} at {t1:.4f}, took {t2 - t1:.4f} seconds")
    logger.close()
