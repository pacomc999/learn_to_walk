# Duck 3: challenges

Situations the duck never trained for, met by its kept brain (no wobble). Each attempt lasts up to 20 s; walked means it did not fall. Written by `python -m duck.challenges`; remarks go in `notes.md`.

## Speeds above training (`speed`)

Asked for one speed for the whole attempt, on the flat floor. Training asked for 0.2 to 0.7 m/s.

| asked m/s | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0.4 | 10 | 10 | 20.0 | 0.4 | 9.8 |
| 0.7 | 10 | 10 | 20.0 | 0.69 | 13.1 |
| 0.8 | 3 | 10 | 9.2 | 0.72 | 15.7 |
| 0.9 | 0 | 10 | 1.7 | 0.72 | 15.4 |
| 1.0 | 0 | 10 | 1.0 | 0.66 | 16.7 |
| 1.2 | 0 | 10 | 0.6 | 0.55 | 19.4 |
| 1.5 | 0 | 10 | 0.5 | 0.53 | 19.4 |

## A backpack (`backpack`)

An 8 x 7 x 4 cm box fixed on the back, on the flat floor at 0.4 m/s. The duck weighs 1.53 kg and has no sense for the extra weight.

| backpack g | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0 | 10 | 10 | 20.0 | 0.4 | 9.8 |
| 100 | 10 | 10 | 20.0 | 0.39 | 9.6 |
| 200 | 10 | 10 | 20.0 | 0.38 | 9.6 |
| 300 | 10 | 10 | 20.0 | 0.36 | 9.7 |
| 500 | 10 | 10 | 20.0 | 0.24 | 10.3 |
| 800 | 0 | 10 | 2.7 | 0.13 | 28.4 |

## Taller stones (`stones`)

Cobblestones of one height for the whole attempt, at 0.4 m/s, a new layout each attempt. 30 attempts per height, because on tall stones tiny differences decide falls.

| stones cm | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 1 | 30 | 30 | 20.0 | 0.4 | 10.3 |
| 2 | 26 | 30 | 18.9 | 0.38 | 11.7 |
| 3 | 0 | 30 | 6.3 | 0.36 | 16.1 |
| 4 | 0 | 30 | 3.0 | 0.34 | 19.3 |
| 5 | 0 | 30 | 1.4 | 0.33 | 20.8 |
| 6 | 0 | 30 | 1.2 | 0.32 | 19.5 |
| 8 | 0 | 30 | 1.0 | 0.29 | 18.8 |

## Thrown rocks (`rocks`)

A stone ball thrown at the body every 2 s from a random side at 4 m/s, on the flat floor at 0.4 m/s (see duck/rocks.py). The rock is taken away 1 s after each throw.

| rock g | walked | attempts | average seconds up | actual m/s | J per metre | rock diameter cm | hits |
|---|---|---|---|---|---|---|---|
| 25 | 10 | 10 | 20.0 | 0.4 | 9.8 | 2.7 | 90 |
| 50 | 10 | 10 | 20.0 | 0.4 | 10.0 | 3.4 | 90 |
| 100 | 5 | 10 | 15.6 | 0.39 | 11.6 | 4.2 | 70 |
| 125 | 0 | 10 | 6.8 | 0.39 | 12.9 | 4.6 | 29 |
| 150 | 0 | 10 | 4.7 | 0.38 | 12.4 | 4.9 | 19 |
| 175 | 0 | 10 | 3.9 | 0.38 | 12.8 | 5.1 | 15 |
| 200 | 0 | 10 | 2.8 | 0.38 | 13.2 | 5.3 | 10 |
| 400 | 0 | 10 | 2.5 | 0.37 | 14.2 | 6.7 | 10 |

## Films

Not kept in git; make them again with `--films` (and `--film GRAMS` for a rock round).

- `speed/films/index.html`: attempt 1 of every value
- `backpack/films/index.html`: attempt 1 of every value
- `stones/films/index.html`: attempt 1 of every value
- `rocks/films/index.html`: attempt 1 of every value
