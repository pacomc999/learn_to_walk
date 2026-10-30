# Duck 2.1: challenges

Situations the duck never trained for, met by its kept brain (no wobble). Each attempt lasts up to 20 s; walked means it did not fall. Written by `python -m duck.challenges`; remarks go in `notes.md`.

## Speeds above training (`speed`)

Asked for one speed for the whole attempt, on the flat floor. Training asked for 0.2 to 0.7 m/s.

| asked m/s | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0.4 | 10 | 10 | 20.0 | 0.47 | 12.2 |
| 0.7 | 10 | 10 | 20.0 | 0.73 | 11.3 |
| 0.8 | 10 | 10 | 20.0 | 0.82 | 11.0 |
| 0.9 | 10 | 10 | 20.0 | 0.9 | 10.9 |
| 1.0 | 1 | 10 | 7.4 | 0.97 | 12.9 |
| 1.2 | 0 | 10 | 0.8 | 0.78 | 12.4 |
| 1.5 | 0 | 10 | 0.6 | 0.64 | 11.3 |

## A backpack (`backpack`)

An 8 x 7 x 4 cm box fixed on the back, on the flat floor at 0.4 m/s. The duck weighs 1.53 kg and has no sense for the extra weight.

| backpack g | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0 | 10 | 10 | 20.0 | 0.47 | 12.2 |
| 100 | 10 | 10 | 20.0 | 0.45 | 12.7 |
| 200 | 10 | 10 | 20.0 | 0.44 | 13.2 |
| 300 | 9 | 10 | 18.1 | 0.44 | 14.4 |
| 500 | 9 | 10 | 18.1 | 0.42 | 15.2 |
| 800 | 8 | 10 | 16.1 | 0.39 | 17.3 |

## Taller stones (`stones`)

Cobblestones of one height for the whole attempt, at 0.4 m/s, a new layout each attempt. 30 attempts per height, because on tall stones tiny differences decide falls.

| stones cm | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 1 | 20 | 30 | 17.0 | 0.43 | 14.5 |
| 2 | 13 | 30 | 14.6 | 0.42 | 15.2 |
| 3 | 0 | 30 | 6.9 | 0.41 | 17.2 |
| 4 | 0 | 30 | 3.2 | 0.39 | 19.2 |
| 5 | 0 | 30 | 1.6 | 0.39 | 18.8 |
| 6 | 0 | 30 | 1.2 | 0.35 | 20.4 |
| 8 | 0 | 30 | 1.0 | 0.3 | 17.9 |

## Thrown rocks (`rocks`)

A stone ball thrown at the body every 2 s from a random side at 4 m/s, on the flat floor at 0.4 m/s (see duck/rocks.py). The rock is taken away 1 s after each throw.

| rock g | walked | attempts | average seconds up | actual m/s | J per metre | rock diameter cm | hits |
|---|---|---|---|---|---|---|---|
| 25 | 10 | 10 | 20.0 | 0.46 | 12.3 | 2.7 | 88 |
| 50 | 10 | 10 | 20.0 | 0.46 | 12.3 | 3.4 | 87 |
| 100 | 9 | 10 | 20.0 | 0.46 | 12.6 | 4.2 | 88 |
| 125 | 8 | 10 | 18.3 | 0.46 | 12.9 | 4.6 | 80 |
| 150 | 4 | 10 | 13.7 | 0.47 | 13.1 | 4.9 | 60 |
| 175 | 0 | 10 | 5.8 | 0.47 | 13.1 | 5.1 | 23 |
| 200 | 0 | 10 | 4.3 | 0.46 | 13.1 | 5.3 | 17 |
| 400 | 0 | 10 | 2.6 | 0.43 | 14.1 | 6.7 | 10 |

## Films

Not kept in git; make them again with `--films` (and `--film GRAMS` for a rock round). `index.html` plays every challenge's films on one page.

- `speed/films/index.html`: attempt 1 of every value
- `backpack/films/index.html`: attempt 1 of every value
- `stones/films/index.html`: attempt 1 of every value
- `rocks/films/index.html`: attempt 1 of every value
