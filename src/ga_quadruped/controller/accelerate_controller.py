from logging import warning

class AccelerateController:
    """
    RC-car style controller that integrates acceleration into speed/yaw-rate.

    Keys this controller handles:
      'w' : forward accel
      's' : backward accel
      'a' : left turn accel (+yaw accel)
      'd' : right turn accel (-yaw accel)
      ' ' : brake (toward zero speed)
      't' : reset

    Notes:
      - Call step(key, dt) each tick. If key is None, it just applies drag/coasting.
      - Unrecognized keys trigger a warnings.warn, unless they’re in passthrough_keys.
      - Output is (vx, vy, w) with vy=0 for car-like motion.
    """

    def __init__(
        self,
        accel=2.0,           # m/s^2
        steer_accel=2.5,     # rad/s^2
        brake=6.0,           # m/s^2
        drag=0.8,            # 1/s
        ang_drag=1.5,        # 1/s
        v_max=1.0,           # m/s
        w_max=1.0,           # rad/s
        default_dt=0.02,     # s
        passthrough_keys=("q", "Q"),  # keys your runner uses; won't warn
    ):
        self.accel = float(accel)
        self.steer_accel = float(steer_accel)
        self.brake = float(brake)
        self.drag = float(drag)
        self.ang_drag = float(ang_drag)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.default_dt = float(default_dt)
        self.passthrough_keys = set(passthrough_keys)

        self.v = 0.0  # forward speed
        self.w = 0.0  # yaw rate

    # --- public API ---------------------------------------------------------

    def step(self, key=None, dt=None):
        """
        Integrate one tick and return (vx, vy, w).
        Args:
          key: a single-character string (or None). Only the keys listed above affect motion.
          dt: override timestep for this tick (defaults to self.default_dt).
        """
        dt = self.default_dt if dt is None else float(dt)

        # commands for this frame
        ax = 0.0       # longitudinal acceleration
        alpha = 0.0    # yaw angular acceleration
        braking = False

        if (key is not None) and (key != ''):
            if   key == 'w': ax += self.accel
            elif key == 's': ax -= self.accel
            elif key == 'a': alpha += self.steer_accel
            elif key == 'd': alpha -= self.steer_accel
            elif key == ' ': braking = True
            elif key == 't': self.reset()
            else:
                if key not in self.passthrough_keys:
                    warning(f"AccelerateController: unrecognized key '{key}'")

        # integrate longitudinal speed with drag
        if braking:
            self.v = self._toward_zero(self.v, self.brake * dt)
        else:
            self.v += (ax - self.drag * self.v) * dt

        # integrate yaw rate with angular drag
        self.w += (alpha - self.ang_drag * self.w) * dt

        # clamp
        self.v = self._clip(self.v, self.v_max)
        self.w = self._clip(self.w, self.w_max)

        return self.v, 0.0, self.w

    def reset(self):
        self.v = 0.0
        self.w = 0.0

    def get(self):
        return self.v, 0.0, self.w

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _clip(x, lim):
        if x >  lim: return lim
        if x < -lim: return -lim
        return x

    @staticmethod
    def _toward_zero(x, dx):
        if x > 0.0:   return max(0.0, x - abs(dx))
        if x < 0.0:   return min(0.0, x + abs(dx))
        return 0.0
