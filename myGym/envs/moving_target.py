# myGym/envs/moving_target.py
from myGym.envs import distractor as d
import numpy as np

class MovingTargetModule(d.DistractorModule):
    def __init__(self, moveable=1, movement_endpoints=None, constant_speed=1,
                 movement_dims=2, env=None, reach_bounds=None,
                 speed_range=(0.12, 0.30), dt=0.02):
        if movement_endpoints is None:
            movement_endpoints = [-0.58, 0.55, -0.35, 0.35, 0.20, 0.20]
        super().__init__(distractor_moveable=moveable,
                         distractor_movement_endpoints=movement_endpoints,
                         distractor_constant_speed=constant_speed,
                         distractor_movement_dimensions=movement_dims,
                         env=env)
        if reach_bounds is None:
            raise ValueError("reach_bounds=[xL,xH,yL,yH] is required")
        self.reach_bounds = tuple(reach_bounds)
        self.speed_range  = tuple(speed_range)
        self.dt           = float(dt)

        # per-target cached step: uid -> np.array([dx,dy,(dz)])
        self._step_by_uid = {}

        # module-local RNG: use env.np_random if present (respects Gym seeding),
        # otherwise fall back to fresh entropy (not the global np.random).
        self.rng = getattr(env, "np_random", None)
        if self.rng is None or not hasattr(self.rng, "uniform"):
            self.rng = np.random.default_rng()

    # --- lifecycle hook you can call from env.reset() ---
    def reset_velocity_cache(self):
        self._step_by_uid.clear()

    def place_target(self, target_name, p, goal):
        obj = super().place_distractor(target_name, p, goal)
        self._ensure_step_for(obj)  # sample once here
        return obj

    def execute_target_step(self, name):
        for obj in self.env.env_objects.get("moving_target", []):
            if obj.name == name:
                if self.distractor_moveable and not self.distractor_stopped:
                    if obj.uid not in self._step_by_uid:
                        self._ensure_step_for(obj)  # lazy for bind-goal
                    
                    self._move_with_predictive_bounce(obj)
                else:
                    obj.move([0.0, 0.0, 0.0])

    def resample_velocity(self, obj):
        self._step_by_uid.pop(obj.uid, None)
        self._ensure_step_for(obj)

    # ---------- internals ----------
    def get_velocity_for(self, obj):
            """Return world-frame velocity vector (m/s) for this target (or zeros)."""
            step = self.get_step_for(obj)
            if step is None:
                # choose dimensionality consistently with movement_dims
                dims = 3 if self.distractor_movement_dimensions == 3 else 2
                return np.zeros(dims, dtype=float)
            return np.array(step, dtype=float) / max(self.dt, 1e-9)
    
    def get_step_for(self, obj):
        """Return stored per-step displacement vector for this target (or None)."""
        return self._step_by_uid.get(obj.uid)

    def _ensure_step_for(self, obj):
        """
        Sample a constant per-step displacement toward a random point in the reach box.
        Guarantee the FIRST step is heading inward (no initial bounce), even if we
        spawned at or slightly outside a boundary.
        """
        x, y, _ = obj.get_position()
        rxL, rxH, ryL, ryH = self.reach_bounds
        xL, xH, yL, yH, zL, zH = self.distractor_movement_endpoints

        # 1) pick a random target point inside the reachable rectangle
        tx = self.rng.uniform(rxL, rxH)
        ty = self.rng.uniform(ryL, ryH)


        vx, vy = (tx - x), (ty - y)
        nrm = float(np.hypot(vx, vy))
        if nrm < 1e-9:
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            dirx, diry = np.cos(theta), np.sin(theta)
        else:
            dirx, diry = vx / nrm, vy / nrm

        # optional: tiny jitter to avoid axis-aligned directions
        # phi = self.rng.uniform(-0.1, 0.1)  # ± ~5.7°
        # c, s = np.cos(phi), np.sin(phi)
        # dirx, diry = c*dirx - s*diry, s*dirx + c*diry

        vmin, vmax = self.speed_range
        v = self.rng.uniform(vmin, vmax)  # m/s

        step = np.array([v * self.dt * dirx, v * self.dt * diry], dtype=float)
        if self.distractor_movement_dimensions == 3:
            step = np.append(step, 0.0)  # keep Z fixed unless you want Z motion

        # # 2) FIRST-STEP INWARD ENFORCEMENT
        # # If we are at/near/outside a boundary AND the step points outward, flip it.
        # eps = 1e-6

        # # X component
        # if (x <= xL + eps and step[0] < 0) or (x < xL and step[0] < 0):
        #     step[0] = -step[0]  # go inward
        # if (x >= xH - eps and step[0] > 0) or (x > xH and step[0] > 0):
        #     step[0] = -step[0]

        # # Y component
        # if (y <= yL + eps and step[1] < 0) or (y < yL and step[1] < 0):
        #     step[1] = -step[1]
        # if (y >= yH - eps and step[1] > 0) or (y > yH and step[1] > 0):
        #     step[1] = -step[1]

        # # (If you enable 3D Z motion, mirror for step[2] vs zL/zH similarly.)

        self._step_by_uid[obj.uid] = step

    def _move_with_predictive_bounce(self, obj):
        """Flip component if the NEXT position would exit AABB, then move."""
        step = self._step_by_uid[obj.uid]
        xL, xH, yL, yH, zL, zH = self.distractor_movement_endpoints
        x, y, z = obj.get_position()

        nx = x + step[0]
        ny = y + step[1]

        # if nx < xL or nx > xH:
        #     step[0] = -step[0]
        #     nx = x + step[0]
        # if ny < yL or ny > yH:
        #     step[1] = -step[1]
        #     ny = y + step[1]

        if self.distractor_movement_dimensions == 3 and step.shape[0] == 3:
            nz = z + step[2]
            if nz < zL or nz > zH:
                step[2] = -step[2]

        self._step_by_uid[obj.uid] = step

        if self.distractor_movement_dimensions == 1:
            obj.move([step[0], 0.0, 0.0])
        elif self.distractor_movement_dimensions == 2:
            obj.move([step[0], step[1], 0.0])
        else:
            obj.move(step.tolist())

