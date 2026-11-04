"""The animals that can learn to walk. Each one is a module with KEY, NAME, DESCRIPTION,
SETTING_GROUPS and build(values)."""
from studio.creatures import duck, puppy

ANIMALS = {module.KEY: module for module in (duck, puppy)}


# The recipe of an animal by its key, with a plain message for an unknown one
def animal(key: str):
    if key not in ANIMALS:
        raise ValueError(f"There is no animal called {key!r} (known: {', '.join(ANIMALS)}).")
    return ANIMALS[key]


# An animal built from its settings
def build_creature(key: str, values: dict | None = None):
    return animal(key).build(values)
