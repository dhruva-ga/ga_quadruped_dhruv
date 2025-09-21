import os
import time

class ParamLogger:
    def __init__(self, log_folder):
        self.log_folder = log_folder
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.log_folder = os.path.join(self.log_folder, timestamp_str)
        os.makedirs(self.log_folder, exist_ok=True)
        self.files = {}
        self.series_names = ['cycle_count', 'ctrl', 'position', 'velocity', 'torque', 'current', 'temperature']
        for name in self.series_names:
            self.files[name] = open(os.path.join(self.log_folder, f"{name}.txt"), "w")
        # Open timestamp.txt for appending
        self.timestamp_file = open(os.path.join(self.log_folder, "timestamp.txt"), "w")

    def log(self, cycle_count=None, ctrl=None, position=None, velocity=None, torque=None, current=None, temperature=None, timestamp=None):
        data = {
            'cycle_count': cycle_count,
            'ctrl': ctrl,
            'position': position,
            'velocity': velocity,
            'torque': torque,
            'current': current,
            'temperature': temperature
        }
        for name in self.series_names:
            if data[name] is not None:
                line = ",".join(map(str, data[name]))
                self.files[name].write(line + "\n")
        for name in self.series_names:
            self.files[name].flush()
        if timestamp is None:
            timestamp = time.time()
        self.timestamp_file.write(str(timestamp) + "\n")
        self.timestamp_file.flush()

    def close(self):
        for f in self.files.values():
            f.close()

if __name__ == "__main__":
    logger = ParamLogger("logs")
    for i in range(1000):
        ctrl = [10, 10, 10]
        position = [i, i, i]
        velocity = [i * 0.1, i * 0.1, i * 0.1]
        torque = [i * 0.01, i * 0.01, i * 0.01]
        current = [i*234, i*23.4, i*2340]
        temperature = [i * 0.01, i * 0.1, i * 0.05]
        t1 = time.time()
        logger.log(ctrl, position, velocity, torque, current, temperature)
        t2 = time.time()
        print(f"Logged data {i+1} at {t1:.4f}, took {t2 - t1:.4f} seconds")
    logger.close()

