import warnings

class VelocityController:
    """
    Keyboard-driven velocity controller (event-in, no polling).

    Keys:
      w/s : +vx / -vx
      a/d : +vy / -vy
      g/h : +w  / -w
      t   : reset (vx=vy=w=0)
    """
    def __init__(self, vel_step=0.05, max_lin=None, max_ang=None, passthrough_keys=("q","Q")):
        """
        Args:
            vel_step: increment per key press
            max_lin: optional abs limit applied to vx, vy (None = no limit)
            max_ang: optional abs limit applied to w (None = no limit)
            passthrough_keys: keys to ignore without warning (runner-level controls)
        """
        self.vel_step = float(vel_step)
        self.max_lin = None if max_lin is None else float(max_lin)
        self.max_ang = None if max_ang is None else float(max_ang)
        self.passthrough_keys = set(passthrough_keys)

        self.vx = 0.0
        self.vy = 0.0
        self.w  = 0.0

        # map keys to (attribute, delta_multiplier)
        self._keymap = {
            'w': ('vx', +1),
            's': ('vx', -1),
            'a': ('vy', +1),
            'd': ('vy', -1),
            'g': ('w',  +1),
            'h': ('w',  -1),
        }

    # --- public API ---------------------------------------------------------

    def step(self, key=None):
        """
        Apply a single key event (if provided) and return (vx, vy, w).
        If key is None, just returns current values.
        """
        if key is not None:
            self.handle_key(key)
        return self.vx, self.vy, self.w

    def handle_key(self, key):
        """Apply a single key event; warn on unrecognized keys (except passthrough)."""
        if key == 't':  # reset
            self.reset()
            return

        target = self._keymap.get(key)
        if target:
            attr, sign = target
            if attr in ('vx', 'vy'):
                new_val = getattr(self, attr) + sign * self.vel_step
                setattr(self, attr, self._clip_lin(new_val))
            elif attr == 'w':
                new_val = self.w + sign * self.vel_step
                self.w = self._clip_ang(new_val)
        else:
            if key not in self.passthrough_keys:
                warnings.warn(f"VelocityController: unrecognized key '{key}'", RuntimeWarning)

    def get(self):
        """Return current (vx, vy, w)."""
        return self.vx, self.vy, self.w

    def reset(self):
        """Zero out all commands."""
        self.vx = self.vy = self.w = 0.0

    def set_step(self, vel_step):
        """Change increment size on the fly."""
        self.vel_step = float(vel_step)

    # --- helpers ------------------------------------------------------------

    def _clip_lin(self, x):
        if self.max_lin is None:
            return x
        if x >  self.max_lin: return self.max_lin
        if x < -self.max_lin: return -self.max_lin
        return x

    def _clip_ang(self, x):
        if self.max_ang is None:
            return x
        if x >  self.max_ang: return self.max_ang
        if x < -self.max_ang: return -self.max_ang
        return x

    def __repr__(self):
        return f"VelocityController(vx={self.vx:.3f}, vy={self.vy:.3f}, w={self.w:.3f}, step={self.vel_step})"
