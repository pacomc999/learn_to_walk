# Duck 3.1 against thrown rocks

A test of balance, not part of training: stone balls are thrown at the walking duck to see when it falls. Made on 2026-10-30 with `python -m duck.rocks ducks/duck_03_1 --film 150` (code in `duck/rocks.py`).

## How the rocks are thrown

- The duck's kept brain walks on the flat floor at 0.4 m/s, so the rocks are the only difficulty. It never met a rock or a push in training.
- Every 2 s, starting at 2 s, a stone ball (density 2,500 kg per cubic metre, about granite) flies at its body from 0.8 m to the left or right (a coin flip), at 4 m/s, aimed where the body will be and arcing for gravity. After 1 s the rock is taken away, so it never lies on the floor as an obstacle.
- A rock is a real body in the physics, so it pushes the duck by hitting it. The duck's senses read only its own joints, so the rock is not in them.
- 10 attempts of 20 s per rock size (9 throws when the duck stays up). The duck weighs 1.53 kg.

## Results: attempts that walked the full 20 s, out of 10

| Rock | Size | Duck 2.1 (flat ground only) | Duck 3 (stones up to 1 cm) | Duck 3.1 (stones up to 3 cm) |
|---|---|---|---|---|
| 25 g | 1.3 cm | 10 | 10 | 10 |
| 50 g | 1.7 cm | 10 | 10 | 10 |
| 100 g | 2.1 cm | 9 | 5 | 10 |
| 125 g | 2.3 cm | 8 | 0 | 9 |
| 150 g | 2.4 cm | 4 | 0 | 2 |
| 175 g | 2.6 cm | 0 | 0 | 0 |
| 200 g | 2.7 cm | 0 | 0 | 0 |
| 400 g | 3.4 cm | 0 | 0 | 0 |

Every number, with time up, throws and hits, is in `results.csv` here and in `ducks/duck_02_1/rocks/results.csv` and `ducks/duck_03/rocks/results.csv`. Nearly every throw hits (Duck 2.1 is missed a few times, perhaps because it walks faster than asked).

## What it shows

- All three ducks hit a wall at about 175 g, about a ninth of the duck's weight. Below it a duck shrugs off nine hits; above it it falls after one or two.
- Training on stones did not make a better balancer against rocks. Duck 3.1 and Duck 2.1 are about equal (10 attempts per size are too few to separate them), and Duck 3, also trained on stones, is clearly the weakest. A stone disturbs a foot from below as it lands; a rock shoves the whole body sideways in mid-step, which none of the ducks met in training.
- At 150 g Duck 3.1 falls anywhere from the first hit to the eighth: whether a hit tips it over depends on where in its step it lands.

## The films

`films/index.html` plays all 10 attempts with 150 g rocks side by side (2 walk, 8 fall). Films are not kept in git; the command above makes them again, exactly the same.
