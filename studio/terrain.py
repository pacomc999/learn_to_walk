"""The shape of the ground: height maps for every kind of terrain.

A height map is a grid of heights in metres (rows along y, columns along x), lowest point 0.
MuJoCo turns it into a solid floor (a height field). The walker starts at x = 0 and walks
toward +x, so every floor reaches far enough ahead for the fastest asked speed.
The cobblestones are the floor of the learn_to_walk project.
"""
from dataclasses import dataclass

import numpy as np

# The cobblestones: rolling bumps with square stones on top (learn_to_walk's FLOOR_SHAPE)
COBBLE_SHAPE = {
    "waves": 8,
    "wavelength_range": (0.2, 0.5),
    "stone_size": 0.06,
    "stone_height_range": (0.35, 1.0),
    "gap_share": 0.07,
    "bump_share": 0.5,
}

# How finely each terrain is sampled, in metres. Small stones need small cells, hills do not
CELL_SIZES = {"cobblestones": 0.005, "blocks": 0.01, "stairs": 0.01, "craters": 0.01, "hills": 0.02, "hurdles": 0.01,
              "slopes": 0.02}

# Slopes: climbs and descents this long (metres); their height is terrain height per metre of slope
SLOPE_LENGTH = 5.0

# The most steps a flight of stairs climbs, so the floor model can hold the highest landing
MOST_STEPS = 6

# The flat run-up before the first step, and the step depth range, in metres
STAIRS_RUN_UP = 1.0
STEP_DEPTH_RANGE = (0.25, 0.4)


# Where a floor lies and how finely it is sampled
@dataclass(frozen=True)
class FloorLayout:
    x_min: float
    x_max: float
    y_half: float
    cell: float

    @property
    def columns(self) -> int:
        return round((self.x_max - self.x_min) / self.cell) + 1

    @property
    def rows(self) -> int:
        return round(2 * self.y_half / self.cell) + 1

    @property
    def column_x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.columns)

    @property
    def row_y(self) -> np.ndarray:
        return np.linspace(-self.y_half, self.y_half, self.rows)

    # The (row, column) of the cell under a point, kept inside the grid
    def index(self, x: float, y: float) -> tuple[int, int]:
        column = round((x - self.x_min) / self.cell)
        row = round((y + self.y_half) / self.cell)
        return min(max(row, 0), self.rows - 1), min(max(column, 0), self.columns - 1)


# The floor for a terrain, long enough for the fastest asked speed over a whole attempt
def floor_layout(terrain: str, fastest_speed: float, seconds: float, size: float) -> FloorLayout:
    reach = max(16.0, fastest_speed * seconds * 1.15 + 3.0)
    return FloorLayout(x_min=-2.0, x_max=round(reach, 1), y_half=max(1.0, size), cell=CELL_SIZES.get(terrain, 0.01))


# The tallest point a terrain can reach when its bumps are `tallest` metres (stairs pile up)
def capacity(terrain: str, tallest: float) -> float:
    if terrain == "stairs":
        return tallest * MOST_STEPS
    if terrain == "slopes":
        return tallest * SLOPE_LENGTH
    return tallest


# Scale values so the lowest becomes 0 and the highest becomes 1
def _to_unit(values: np.ndarray) -> np.ndarray:
    spread = values.max() - values.min()
    return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)


# ---------- Cobblestones (from learn_to_walk) ----------

def _rolling_bumps(rng, layout: FloorLayout, wavelength_range, waves: int, max_angle: float = np.pi):
    wavelength = rng.uniform(*wavelength_range, waves)
    angle = rng.uniform(0, max_angle, waves) - (max_angle / 2 if max_angle < np.pi else 0)
    phase = rng.uniform(0, 2 * np.pi, waves)
    along_x = 2 * np.pi * np.outer(np.cos(angle) / wavelength, layout.column_x) + phase[:, None]
    along_y = 2 * np.pi * np.outer(layout.row_y, np.sin(angle) / wavelength)
    return np.cos(along_y) @ np.cos(along_x) - np.sin(along_y) @ np.sin(along_x)


def _stone_index_and_gap(positions: np.ndarray, start: float, stone: float, gap_share: float):
    place = (positions - start) / stone
    index = np.floor(place).astype(int)
    inside = place - index
    return index, (inside < gap_share) | (inside > 1 - gap_share)


def _square_stones(rng, layout: FloorLayout, stone: float, height_range, gap_share: float):
    column_stone, column_gap = _stone_index_and_gap(layout.column_x, layout.x_min, stone, gap_share)
    row_stone, row_gap = _stone_index_and_gap(layout.row_y, -layout.y_half, stone, gap_share)
    heights = rng.uniform(*height_range, (row_stone.max() + 1, column_stone.max() + 1))
    stones = heights[row_stone[:, None], column_stone[None, :]]
    stones[row_gap[:, None] | column_gap[None, :]] = 0.0
    return stones


def cobblestones(rng, layout: FloorLayout) -> np.ndarray:
    shape = COBBLE_SHAPE
    bumps = _to_unit(_rolling_bumps(rng, layout, shape["wavelength_range"], shape["waves"]))
    stones = _to_unit(_square_stones(rng, layout, shape["stone_size"], shape["stone_height_range"],
                                     shape["gap_share"]))
    return _to_unit(shape["bump_share"] * bumps + (1 - shape["bump_share"]) * stones)


# ---------- The new terrains ----------

# Long smooth hills, mostly across the walking line so the walker goes up and down
def hills(rng, layout: FloorLayout) -> np.ndarray:
    waves = _rolling_bumps(rng, layout, (2.0, 6.0), 4, max_angle=0.7)
    return _to_unit(waves)


# Flights of steps up and down, with flat landings between them, the same all across
def stairs(rng, layout: FloorLayout) -> np.ndarray:
    x = layout.column_x
    profile = np.zeros_like(x)
    position, level = STAIRS_RUN_UP, 0.0
    while position < layout.x_max:
        steps = int(rng.integers(3, MOST_STEPS + 1))
        rises = rng.uniform(0.6, 1.0, steps)
        depth = rng.uniform(*STEP_DEPTH_RANGE)
        for direction in (1, -1):
            for rise in rises:
                level = max(level + direction * rise, 0.0)
                profile[x >= position] = level
                position += depth
            position += rng.uniform(0.6, 1.2)  # landing
    profile /= MOST_STEPS  # a share of what the floor model holds
    return np.tile(profile, (layout.rows, 1))


# A dusty plain with round craters (bowls with a raised rim) and a little roughness
def craters(rng, layout: FloorLayout) -> np.ndarray:
    xs, ys = layout.column_x, layout.row_y
    ground = np.zeros((layout.rows, layout.columns))
    area = (layout.x_max - layout.x_min) * 2 * layout.y_half
    for _ in range(int(area / 2.5)):
        centre_x = rng.uniform(layout.x_min, layout.x_max)
        centre_y = rng.uniform(-layout.y_half, layout.y_half)
        radius = rng.uniform(0.25, 0.8)
        depth = rng.uniform(0.4, 1.0)
        # Only the cells near this crater change, so work on that window alone
        reach = 1.6 * radius
        columns = slice(*np.searchsorted(xs, [centre_x - reach, centre_x + reach]))
        rows = slice(*np.searchsorted(ys, [centre_y - reach, centre_y + reach]))
        distance = np.hypot(xs[columns][None, :] - centre_x, ys[rows][:, None] - centre_y)
        bowl = np.where(distance < radius, depth * (1 - (distance / radius) ** 2), 0.0)
        rim = 0.25 * depth * np.exp(-((distance - radius) / (0.18 * radius)) ** 2)
        ground[rows, columns] += rim - bowl
    dust = _to_unit(_rolling_bumps(rng, layout, (0.3, 0.8), 6))
    return _to_unit(ground + 0.15 * dust)


# Big flat topped rocks of random heights, side by side
def blocks(rng, layout: FloorLayout) -> np.ndarray:
    size = rng.uniform(0.18, 0.32)
    column_block = np.floor((layout.column_x - layout.x_min) / size).astype(int)
    row_block = np.floor((layout.row_y + layout.y_half) / size).astype(int)
    heights = rng.uniform(0.0, 1.0, (row_block.max() + 1, column_block.max() + 1))
    heights[rng.random(heights.shape) < 0.25] = 0.0
    return heights[row_block[:, None], column_block[None, :]]


# Low bars across the path to step over, about a metre apart, after a flat run-up
def hurdles(rng, layout: FloorLayout) -> np.ndarray:
    x = layout.column_x
    profile = np.zeros_like(x)
    position = STAIRS_RUN_UP
    while position < layout.x_max:
        width = rng.uniform(0.03, 0.06)
        profile[(x >= position) & (x < position + width)] = rng.uniform(0.6, 1.0)
        position += rng.uniform(0.8, 1.3)
    return np.tile(profile, (layout.rows, 1))


# Long even climbs and descents with flat tops, like a mountain path. The height setting is the rise per metre
def slopes(rng, layout: FloorLayout) -> np.ndarray:
    x = layout.column_x
    profile = np.zeros_like(x)
    position, level = STAIRS_RUN_UP, 0.0
    while position < layout.x_max:
        length = rng.uniform(0.6, 1.0) * SLOPE_LENGTH
        inside = (x >= position) & (x < position + length)
        profile[inside] = level + (x[inside] - position) / SLOPE_LENGTH
        position += length
        level += length / SLOPE_LENGTH
        flat = rng.uniform(0.8, 1.5)
        profile[(x >= position) & (x < position + flat)] = level
        position += flat
        inside = (x >= position) & (x < position + length)
        profile[inside] = level - (x[inside] - position) / SLOPE_LENGTH
        position += length
        level = max(level - length / SLOPE_LENGTH, 0.0)
        profile[x >= position] = level
        position += rng.uniform(0.8, 1.5)
    return np.tile(np.clip(profile, 0.0, 1.0), (layout.rows, 1))


SHAPES = {"cobblestones": cobblestones, "hills": hills, "stairs": stairs, "craters": craters, "blocks": blocks,
          "hurdles": hurdles, "slopes": slopes}


# Heights in metres for one attempt: the terrain's shape scaled to this attempt's tallest bumps.
# Stairs keep their own scale: each step is a share of `height`, piled up to at most MOST_STEPS steps
def ground_heights(terrain: str, rng, layout: FloorLayout, height: float) -> np.ndarray:
    if height <= 0 or terrain not in SHAPES:
        return np.zeros((layout.rows, layout.columns))
    shape = SHAPES[terrain](rng, layout)
    return shape * capacity(terrain, height)


# The highest ground in the square of half size `radius` around (x, y)
def highest_near(heights: np.ndarray, layout: FloorLayout, x: float, y: float, radius: float) -> float:
    first_row, first_column = layout.index(x - radius, y - radius)
    last_row, last_column = layout.index(x + radius, y + radius)
    return float(heights[first_row:last_row + 1, first_column:last_column + 1].max())


# The ground height under one point
def height_at(heights: np.ndarray, layout: FloorLayout, x: float, y: float) -> float:
    row, column = layout.index(x, y)
    return float(heights[row, column])
