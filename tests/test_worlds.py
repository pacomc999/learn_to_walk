import pytest

from duck.rewards import RewardWeights
from duck.worlds import Backpack, Rocks, load_world, world_from_dict, world_to_dict


# A world file reads back as the same world, and writing it gives the file layout again
def test_world_file_round_trip():
    world = load_world("stones_up_to_4cm")
    assert world.tallest_stones == 0.04 and world.target_speed_range == (0.2, 0.7)
    assert world.reward == RewardWeights(3.0, 0.2, 1.5, 0.5) and world.rocks is None and world.backpack is None
    assert world_from_dict(world_to_dict(world)) == world


# Extras: rocks take the usual throwing rhythm and speed unless the file says otherwise
def test_world_with_extras():
    data = {**world_to_dict(load_world("flat_walking")),
            "extras": {"rocks": {"mass": 0.1}, "backpack": {"mass": 0.2}}}
    world = world_from_dict(data)
    assert world.rocks == Rocks(0.1, 2.0, 4.0) and world.backpack == Backpack(0.2)


# Mistakes in a world file are refused with a message that names the key
@pytest.mark.parametrize("change, word", [
    ({"colour": "red"}, "colour"),
    ({"extras": {"wind": {}}}, "wind"),
    ({"target_speed_range": [0.7, 0.2]}, "target_speed_range"),
    ({"extras": {"rocks": {"mass": -1}}}, "mass"),
    ({"floor": {"tallest_stones": -0.01}}, "tallest_stones"),
])
def test_bad_world_is_refused_plainly(change, word):
    data = {**world_to_dict(load_world("flat_walking")), **change}
    with pytest.raises(ValueError, match=word):
        world_from_dict(data, source="worlds/test.json")


# The worlds of the existing ducks have the floors and speeds their recipes recorded
def test_the_five_worlds_match_the_old_recipes():
    expected = {"flat_walking": (0.0, (0.2, 0.7)), "stones_up_to_1cm": (0.01, (0.2, 0.7)),
                "stones_up_to_3cm": (0.03, (0.2, 0.7)), "stones_up_to_4cm": (0.04, (0.2, 0.7)),
                "flat_running": (0.0, (0.2, 1.5))}
    for name, (stones, speeds) in expected.items():
        world = load_world(name)
        assert (world.tallest_stones, world.target_speed_range) == (stones, speeds)
