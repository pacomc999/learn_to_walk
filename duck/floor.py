"""A random cobblestone floor: height maps and the MuJoCo height field model."""
import argparse

import mujoco
import numpy as np

from duck.model import DUCK_XML_PATH

# Highest point the floor can ever reach, in metres (also the height field elevation)
MAX_ROUGHNESS = 0.01

# The shape of the floor: rolling bumps with square cobblestones on top
FLOOR_SHAPE = {
    "waves": 8,                       # cosine waves added into the bumps
    "wavelength_range": (0.2, 0.5),   # metres from crest to crest
    "stone_size": 0.06,               # square stone side, metres
    "stone_height_range": (0.35, 1.0),  # flat top height as a share of the tallest stone
    "gap_share": 0.07,                # share of the stone size left as gap on each side
    "bump_share": 0.5,                # weight of the bumps, the stones get the rest
}

# Where the floor lies and how finely it is sampled, in metres
# The rough floor ends at x 16 m and |y| 1 m (the fastest asked speed, 0.7 m/s for 20 s, is 14 m).
# The flat plane (roughness 0 in the exam's flat part) has no edge.
FLOOR_X_MIN, FLOOR_X_MAX = -2.0, 16.0
FLOOR_Y_MIN, FLOOR_Y_MAX = -1.0, 1.0
CELL_SIZE = 0.005
FLOOR_COLS = round((FLOOR_X_MAX - FLOOR_X_MIN) / CELL_SIZE) + 1  # columns run along x
FLOOR_ROWS = round((FLOOR_Y_MAX - FLOOR_Y_MIN) / CELL_SIZE) + 1  # rows run along y

# World position of every column and every row, worked out once
COLUMN_X = np.linspace(FLOOR_X_MIN, FLOOR_X_MAX, FLOOR_COLS)
ROW_Y = np.linspace(FLOOR_Y_MIN, FLOOR_Y_MAX, FLOOR_ROWS)


# Scale values so the lowest becomes 0 and the highest becomes 1
def _scale_to_unit(values: np.ndarray) -> np.ndarray:
    return (values - values.min()) / (values.max() - values.min())


# Rolling bumps: random waves added together
# cos(a + b) = cos(a) cos(b) - sin(a) sin(b), so all waves become one matrix product
def _rolling_bumps(rng: np.random.Generator) -> np.ndarray:
    low, high = FLOOR_SHAPE["wavelength_range"]
    waves = FLOOR_SHAPE["waves"]
    wavelength = rng.uniform(low, high, waves)
    angle = rng.uniform(0, np.pi, waves)
    phase = rng.uniform(0, 2 * np.pi, waves)
    along_x = 2 * np.pi * np.outer(np.cos(angle) / wavelength, COLUMN_X) + phase[:, None]
    along_y = 2 * np.pi * np.outer(ROW_Y, np.sin(angle) / wavelength)
    return np.cos(along_y) @ np.cos(along_x) - np.sin(along_y) @ np.sin(along_x)


# For each point on one axis: which stone it is in, and whether it lies in a gap
def _stone_index_and_gap(positions: np.ndarray, start: float) -> tuple[np.ndarray, np.ndarray]:
    stone_position = (positions - start) / FLOOR_SHAPE["stone_size"]
    stone_index = np.floor(stone_position).astype(int)
    inside_stone = stone_position - stone_index  # 0 to 1 across the stone
    gap_share = FLOOR_SHAPE["gap_share"]
    in_gap = (inside_stone < gap_share) | (inside_stone > 1 - gap_share)
    return stone_index, in_gap


# Square cobblestones: flat tops at random heights with gaps at height 0 between them
def _square_stones(rng: np.random.Generator) -> np.ndarray:
    column_stone, column_gap = _stone_index_and_gap(COLUMN_X, FLOOR_X_MIN)
    row_stone, row_gap = _stone_index_and_gap(ROW_Y, FLOOR_Y_MIN)
    low, high = FLOOR_SHAPE["stone_height_range"]
    stone_heights = rng.uniform(low, high, (row_stone.max() + 1, column_stone.max() + 1))
    stones = stone_heights[row_stone[:, None], column_stone[None, :]]
    stones[row_gap[:, None] | column_gap[None, :]] = 0.0
    return stones


# Check a roughness in metres: from 0 up to MAX_ROUGHNESS, otherwise ValueError
def check_roughness(roughness: float) -> float:
    if roughness < 0 or roughness > MAX_ROUGHNESS:
        raise ValueError(f"Roughness must be from 0 to {MAX_ROUGHNESS} metres, not {roughness}.")
    return roughness


# For argparse: read a roughness from the command line and reject values outside the range
def roughness_argument(text: str) -> float:
    try:
        return check_roughness(float(text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


# Floor heights in metres, lowest point 0 and highest point exactly the roughness
def floor_heights(rng: np.random.Generator, roughness: float) -> np.ndarray:
    check_roughness(roughness)
    if roughness == 0:
        return np.zeros((FLOOR_ROWS, FLOOR_COLS))
    bumps = _scale_to_unit(_rolling_bumps(rng))
    stones = _scale_to_unit(_square_stones(rng))
    bump_share = FLOOR_SHAPE["bump_share"]
    combined = bump_share * bumps + (1 - bump_share) * stones
    return _scale_to_unit(combined) * roughness


# The duck XML with the plane floor swapped for a height field geom named "floor"
def _floor_xml() -> str:
    xml = DUCK_XML_PATH.read_text()
    half_x = (FLOOR_X_MAX - FLOOR_X_MIN) / 2
    half_y = (FLOOR_Y_MAX - FLOOR_Y_MIN) / 2
    centre_x = (FLOOR_X_MAX + FLOOR_X_MIN) / 2
    hfield = (f'<hfield name="ground" nrow="{FLOOR_ROWS}" ncol="{FLOOR_COLS}" '
              f'size="{half_x} {half_y} {MAX_ROUGHNESS} 0.05"/>')
    xml = xml.replace("<asset>", f"<asset>\n    {hfield}", 1)
    start = xml.index('<geom name="floor"')
    end = xml.index("/>", start) + 2
    floor_geom = (f'<geom name="floor" type="hfield" hfield="ground" pos="{centre_x} 0 0" '
                  'material="grass" contype="1" conaffinity="1"/>')
    return xml[:start] + floor_geom + xml[end:]


# The duck model with a flat height field floor, ready for set_floor
def load_floor_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_floor_xml())


# Write floor heights (metres) into the model's height field
def set_floor(model: mujoco.MjModel, heights: np.ndarray) -> None:
    model.hfield_data[:] = (heights / MAX_ROUGHNESS).ravel()


# The (row, column) of the cell under a world point, kept inside the grid
def floor_index(x: float, y: float) -> tuple[int, int]:
    column = round((x - FLOOR_X_MIN) / CELL_SIZE)
    row = round((y - FLOOR_Y_MIN) / CELL_SIZE)
    return min(max(row, 0), FLOOR_ROWS - 1), min(max(column, 0), FLOOR_COLS - 1)


# Highest floor point in the square of half size radius around (x, y)
def highest_floor_near(heights: np.ndarray, x: float, y: float, radius: float) -> float:
    first_row, first_column = floor_index(x - radius, y - radius)
    last_row, last_column = floor_index(x + radius, y + radius)
    return float(heights[first_row:last_row + 1, first_column:last_column + 1].max())
