# learn_to_walk

A 3D duck in a physics simulation teaches itself to walk.

The duck has a long body, legs set far back, a bendy neck, a head and beak, folded wings and webbed feet, with 12 motors of limited strength. A small neural network is its brain. It learns by trial and error (reinforcement learning with PPO): it earns points for walking at the speed it is asked for, pays for the energy its motors use, and nobody tells it how to move its legs. Everything runs on a CPU only Windows laptop.

Each training recipe is a duck in the catalogue, and each duck trains in a world (`worlds/`: the floor, the speeds, the reward and extras such as thrown rocks) with a body (`bodies/`: the duck's own numbers, such as motor strength). The newest, Duck 3.2, runs; Duck 3.1 walks on cobblestones up to about 3.5 cm high. See `duck_catalogue/README.md` for every duck and how it does, and the guide's Worlds and bodies section.

## Setup

Do this once, from the project folder:

1. Make a virtual environment with Python 3.12: `py -3.12 -m venv .venv`
2. Install PyTorch from the CPU index: `.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Install the other libraries: `.venv/Scripts/python -m pip install -r requirements.txt`
4. Check that everything works: `.venv/Scripts/python -m pytest` (about 25 minutes)

Commands run in PowerShell or Git Bash from the project folder.

## The commands you use most

| What | Command |
|---|---|
| Start a duck in the catalogue, in a world with a body | `.venv/Scripts/python -m duck.catalogue new duck_catalogue/duck_04 --name "Duck 4" --world stones_up_to_4cm --body standard` |
| Train it, starting from another duck's brain | `.venv/Scripts/python -m duck.train --duck duck_catalogue/duck_04 --run runs/duck_04/from_duck_03_1 --steps 4000000 --start-from duck_catalogue/duck_03_1/brain` |
| Follow training in the browser | `.venv/Scripts/python -m duck.dashboard runs/duck_04/from_duck_03_1 --steps 4000000` |
| Keep its best brain in the catalogue | `.venv/Scripts/python -m duck.catalogue keep duck_catalogue/duck_04 runs/duck_04/from_duck_03_1/best` |
| Drive a brain live with a speed slider (or double-click `duck_catalogue/duck_03_2/Drive Duck 3.2.bat` for the runner) | `.venv/Scripts/python -m duck.drive duck_catalogue/duck_03_2/brain` |
| Watch a brain walk in a window | `.venv/Scripts/python -m duck.view_snapshot duck_catalogue/duck_03_1/brain --roughness 0.03` |

`docs/guide.md` explains all of it: how the duck learns and is scored, the exam, making and training a duck, the viewing tools, the rock test and keeping the project tidy.

## Where things are

- `duck/`: the code, and `tests/` its tests
- `duck_catalogue/`: one folder per duck with its story, recipe, best brain and card
- `runs/`: training output, not kept in git; `runs/early_experiments/README.md` explains the runs from before the catalogue
- `docs/guide.md`: the full guide; `docs/duck_joints.html`: the duck's joints in 3D; `docs/designs/` and `docs/plans/`: the design notes and build plans
- `CLAUDE.md`: notes for working on the code
