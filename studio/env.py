"""The practice world: any animal in any world, as a Gymnasium environment for training.

Each attempt the walker starts standing with one leg a little ahead, is asked for a speed,
gets a fresh floor, and earns points every moment (see the world's reward settings) until it
falls or the attempt's time is up. This is the duck environment of learn_to_walk made general:
the motors, feet and senses come from the animal's build.
"""
import gymnasium
import mujoco
import numpy as np

from studio.creatures.kit import Build
from studio.scene import Scene, build_scene, set_stand_pose
from studio.terrain import ground_heights, height_at, highest_near
from studio.worlds import World

# Physics steps per control step (0.005 s each, so the walker is controlled 50 times a second)
FRAME_SKIP = 4

# How far each joint may start away from the standing pose, and the stride stance (learn_to_walk)
START_NOISE = 0.02
START_STRIDE = 2 * START_NOISE

# The walker has fallen when its torso is below this share of its standing height,
# or tilted more than about 50 degrees (its up axis below 0.64)
FALL_HEIGHT_SHARE = 0.6
FALL_UP_Z = 0.64

# The ground ahead, seen as heights at these distances in front of the torso (times the animal's size)
GROUND_SCAN = (-0.1, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6)

# Gusts of wind last this long, in seconds; the first comes after FIRST_EVENT seconds
GUST_SECONDS = 0.5
FIRST_EVENT = 2.0

# Rocks: thrown from this far to the side (times the size), taken away after a second
THROW_DISTANCE = 0.8
ROCK_LIFETIME = 1.0

OBSERVATION_LIMIT = 1e6

# The senses a brain can have switched on
DEFAULT_SENSES = {"feet": True, "ground": False}


# How many numbers the walker senses with these senses
def observation_size(build: Build, senses: dict) -> int:
    size = 12 + 2 * len(build.motors) + 1
    if senses.get("feet", True):
        size += len(build.feet)
    if senses.get("ground", False):
        size += len(GROUND_SCAN)
    return size


class WalkEnv(gymnasium.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, build: Build, world: World, senses: dict | None = None, *, target_speed: float | None = None,
                 terrain_height: float | None = None, scene: Scene | None = None):
        self.build = build
        self.world = world
        self.senses = dict(DEFAULT_SENSES, **(senses or {}))
        self.scene = scene or build_scene(build, world)
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        self.fixed_target_speed = target_speed
        self.fixed_height = terrain_height
        self.target_speed = target_speed if target_speed is not None else float(np.mean(world.speed_range))
        self.difficulty = 1.0           # raised step by step by training when the world is gradually harder
        self.heights = None             # this attempt's ground, None on the flat plane
        self.attempt_height = 0.0
        self.max_steps = round(world.values["attempt_seconds"] / self.control_seconds)
        self.step_count = 0
        self.start_x = 0.0
        self.fall_height = FALL_HEIGHT_SHARE * self.scene.stand_height

        # Where things are in the simulation's lists
        self.torso = self.model.body("torso").id
        self.joint_qpos = np.array([self.model.joint(m.name).qposadr[0] for m in build.motors])
        self.joint_qvel = np.array([self.model.joint(m.name).dofadr[0] for m in build.motors])
        self.joint_ranges = np.array([self.model.joint(m.name).range for m in build.motors])
        self.floor = self.model.geom("floor").id
        self.foot_geoms = [{self.model.geom(name).id for name in foot.geoms} for foot in build.feet]
        self.walker_geoms = {g for g in range(self.model.ngeom) if self._belongs_to_walker(g)}

        # The extras
        self.rock = self._rock_parts() if world.values["rocks"] else None
        self.next_rock = self.next_gust = FIRST_EVENT
        self.rock_thrown_at = None
        self.gust_until = -1.0
        self.gust_force = np.zeros(3)
        self.rock_hits = 0

        motor_count = len(build.motors)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (motor_count,), dtype=np.float32)
        size = observation_size(build, self.senses)
        self.observation_space = gymnasium.spaces.Box(-OBSERVATION_LIMIT, OBSERVATION_LIMIT, (size,), np.float32)

    # ---------- Facts ----------

    @property
    def control_seconds(self) -> float:
        return self.model.opt.timestep * FRAME_SKIP

    def _belongs_to_walker(self, geom_id: int) -> bool:
        body = self.model.geom_bodyid[geom_id]
        return body != 0 and self.model.body_rootid[body] == self.model.body("torso").id

    # The ground height under a point (0 on the flat plane)
    def ground_at(self, x: float, y: float) -> float:
        if self.heights is None:
            return 0.0
        return height_at(self.heights, self.scene.layout, x, y)

    def torso_height(self) -> float:
        x, y, z = self.data.xpos[self.torso]
        return float(z - self.ground_at(x, y))

    def torso_up(self) -> float:
        return float(self.data.xmat[self.torso][8])

    def _finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    # ---------- Starting an attempt ----------

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(min(max(difficulty, 0.0), 1.0))

    def _pick_height(self) -> float:
        if self.fixed_height is not None:
            return self.fixed_height
        tallest = self.world.tallest
        if self.world.values["gradually_harder"]:
            tallest *= self.difficulty
        if self.world.values["mixed_heights"]:
            return float(self.np_random.uniform(0.0, tallest)) if tallest > 0 else 0.0
        return tallest

    def _pick_speed(self) -> float:
        if self.fixed_target_speed is not None:
            return self.fixed_target_speed
        return float(self.np_random.uniform(*self.world.speed_range))

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.target_speed = self._pick_speed()
        set_stand_pose(self.model, self.data, self.build, self.scene.stand_height)
        self.data.qpos[self.joint_qpos] += self.np_random.uniform(-START_NOISE, START_NOISE, len(self.joint_qpos))
        self._take_stride_stance()
        self._lay_ground()
        if self.rock:
            self._park_rock()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.start_x = float(self.data.qpos[0])
        self.next_rock = self.next_gust = FIRST_EVENT
        self.rock_thrown_at = None
        self.gust_until = -1.0
        self.gust_force[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.rock_hits = 0
        return self._observation(), self._info({})

    # A coin flip picks the leading leg of each pair
    def _take_stride_stance(self) -> None:
        lead = 1.0 if self.np_random.random() < 0.5 else -1.0
        for first, second in self.build.stride_pairs:
            self.data.qpos[self.model.joint(first).qposadr[0]] += lead * START_STRIDE
            self.data.qpos[self.model.joint(second).qposadr[0]] -= lead * START_STRIDE

    # A fresh floor for this attempt, and the walker lifted above the ground near its start
    def _lay_ground(self) -> None:
        self.attempt_height = self._pick_height()
        layout = self.scene.layout
        if layout is None:
            self.heights = None
            return
        self.heights = ground_heights(self.world.terrain, self.np_random, layout, self.attempt_height)
        most = max(self.scene.floor_capacity, 1e-4)
        self.model.hfield_data[:] = np.clip(self.heights / most, 0.0, 1.0).ravel()
        radius = 0.3 * self.build.size
        self.data.qpos[2] += highest_near(self.heights, layout, 0.0, 0.0, radius)

    # ---------- Senses ----------

    def _observation(self) -> np.ndarray:
        size = self.observation_space.shape[0]
        if not self._finite():
            return np.zeros(size, dtype=np.float32)
        qpos, qvel = self.data.qpos, self.data.qvel
        parts = [
            [self.torso_height()],
            [qpos[1]],
            qpos[3:7],
            qvel[0:6],
            qpos[self.joint_qpos],
            qvel[self.joint_qvel],
        ]
        if self.senses["feet"]:
            parts.append(self._foot_contacts())
        if self.senses["ground"]:
            parts.append(self._ground_scan())
        parts.append([self.target_speed])
        return np.clip(np.concatenate(parts), -OBSERVATION_LIMIT, OBSERVATION_LIMIT).astype(np.float32)

    # 1 for each foot touching the ground, else 0
    def _foot_contacts(self) -> list[float]:
        touching = [0.0] * len(self.foot_geoms)
        for contact in self.data.contact[:self.data.ncon]:
            pair = (contact.geom1, contact.geom2)
            if self.floor not in pair:
                continue
            other = pair[1] if pair[0] == self.floor else pair[0]
            for index, geoms in enumerate(self.foot_geoms):
                if other in geoms:
                    touching[index] = 1.0
        return touching

    # How far below the torso the ground is at points ahead of it
    def _ground_scan(self) -> list[float]:
        x, y, z = self.data.xpos[self.torso]
        return [float(z - self.ground_at(x + step * self.build.size, y)) for step in GROUND_SCAN]

    def _info(self, reward_parts: dict) -> dict:
        finite = self._finite()
        x = float(self.data.qpos[0]) if finite else 0.0
        return {"x_position": x, "y_position": float(self.data.qpos[1]) if finite else 0.0,
                "forward_distance": x - self.start_x if finite else 0.0, "reward_parts": reward_parts,
                "target_speed": self.target_speed, "rock_hits": self.rock_hits}

    # ---------- Moving ----------

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        return low + (action + 1.0) / 2.0 * (high - low)

    def _simulate(self) -> float:
        energy = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            power = np.abs(self.data.actuator_force * self.data.actuator_velocity)
            energy += float(np.sum(power)) * self.model.opt.timestep
        return energy

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        x_before = float(self.data.qpos[0])
        self.data.ctrl[:] = self._scale_action(action)
        self._extras_before_step()
        energy = self._simulate()
        self._extras_after_step()
        self.step_count += 1
        reward, parts = self._reward(x_before, energy)
        terminated = self._fallen()
        truncated = self.step_count >= self.max_steps
        return self._observation(), reward, terminated, truncated, self._info(parts)

    def _fallen(self) -> bool:
        if not self._finite():
            return True
        return self.torso_height() < self.fall_height or self.torso_up() < FALL_UP_Z

    # ---------- Points ----------

    def _reward(self, x_before: float, energy: float) -> tuple[float, dict]:
        if not self._finite():
            return 0.0, {}
        values = self.world.values
        speed = (float(self.data.qpos[0]) - x_before) / self.control_seconds
        parts = {
            "speed": values["speed_points"] * (self.target_speed - abs(speed - self.target_speed)),
            "alive": values["alive_points"],
            "energy": -values["energy_cost"] * energy,
            "straight": -values["straight_cost"] * abs(float(self.data.qpos[1])),
        }
        if values["end_stop_cost"]:
            parts["end_stops"] = -values["end_stop_cost"] * self._end_stop_closeness()
        return float(sum(parts.values())), parts

    # Each joint counts 0 until it enters the last 5% of its range, rising to 1 at the stop
    def _end_stop_closeness(self) -> float:
        angles = self.data.qpos[self.joint_qpos]
        low, high = self.joint_ranges[:, 0], self.joint_ranges[:, 1]
        room = np.minimum(angles - low, high - angles) / (high - low)
        return float(np.sum(np.clip((0.05 - room) / 0.05, 0.0, 1.0)))

    # ---------- Wind and rocks ----------

    def _extras_before_step(self) -> None:
        values, now = self.world.values, self.data.time
        if values["wind"]:
            if now >= self.next_gust:
                angle = self.np_random.uniform(0, 2 * np.pi)
                strength = values["wind_strength"] / 100.0 * self.scene.total_mass * values["gravity"]
                self.gust_force = strength * np.array([np.cos(angle), np.sin(angle), 0.0])
                self.gust_until = now + GUST_SECONDS
                self.next_gust = now + values["wind_every"] * self.np_random.uniform(0.7, 1.3)
            self.data.xfrc_applied[self.torso, :3] = self.gust_force if now < self.gust_until else 0.0
        if self.rock and now >= self.next_rock:
            self._throw_rock()
            self.next_rock = now + values["rock_every"]

    def _extras_after_step(self) -> None:
        if not self.rock or self.rock_thrown_at is None:
            return
        if not self.rock["hit"] and self._rock_touches_walker():
            self.rock["hit"] = True
            self.rock_hits += 1
        if self.data.time - self.rock_thrown_at >= ROCK_LIFETIME:
            self._park_rock()

    def _rock_parts(self) -> dict:
        joint = self.model.joint("rock")
        return {"qpos": joint.qposadr[0], "qvel": joint.dofadr[0], "geom": self.model.geom("rock").id,
                "parked": self.model.body("rock").pos.copy(), "hit": False}

    def _park_rock(self) -> None:
        start = self.rock["qpos"]
        self.data.qpos[start:start + 7] = [*self.rock["parked"], 1, 0, 0, 0]
        self.data.qvel[self.rock["qvel"]:self.rock["qvel"] + 6] = 0.0
        self.rock_thrown_at = None

    # Throw the rock from one side, aiming where the walker will be and arcing for gravity
    def _throw_rock(self) -> None:
        side = 1.0 if self.np_random.random() < 0.5 else -1.0
        distance = THROW_DISTANCE * self.build.size
        speed = 4.0
        flight = distance / speed
        target = self.data.subtree_com[self.torso] + self.data.qvel[0:3] * flight
        start = target + np.array([0.0, side * distance, 0.0])
        velocity = (target - start) / flight
        velocity[2] += 0.5 * self.world.values["gravity"] * flight
        q, v = self.rock["qpos"], self.rock["qvel"]
        self.data.qpos[q:q + 7] = [*start, 1, 0, 0, 0]
        self.data.qvel[v:v + 6] = [*velocity, 0, 0, 0]
        self.rock_thrown_at = self.data.time
        self.rock["hit"] = False

    def _rock_touches_walker(self) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if self.rock["geom"] in pair and pair & self.walker_geoms:
                return True
        return False
