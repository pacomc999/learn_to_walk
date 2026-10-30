# Duck 3.1: challenges

Situations the duck never trained for, met by its kept brain (no wobble). Each attempt lasts up to 20 s; walked means it did not fall. Written by `python -m duck.challenges`; remarks go in `notes.md`.

## Speeds above training (`speed`)

Asked for one speed for the whole attempt, on the flat floor. Training asked for 0.2 to 0.7 m/s.

| asked m/s | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0.4 | 10 | 10 | 20.0 | 0.39 | 10.6 |
| 0.7 | 10 | 10 | 20.0 | 0.66 | 13.8 |
| 0.8 | 10 | 10 | 20.0 | 0.74 | 15.8 |
| 0.9 | 0 | 10 | 2.0 | 0.68 | 17.2 |
| 1.0 | 0 | 10 | 0.9 | 0.61 | 17.3 |
| 1.2 | 0 | 10 | 0.6 | 0.54 | 19.8 |
| 1.5 | 0 | 10 | 0.5 | 0.48 | 23.4 |

## A backpack (`backpack`)

An 8 x 7 x 4 cm box fixed on the back, on the flat floor at 0.4 m/s. The duck weighs 1.53 kg and has no sense for the extra weight.

| backpack g | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 0 | 10 | 10 | 20.0 | 0.39 | 10.6 |
| 100 | 10 | 10 | 20.0 | 0.37 | 10.9 |
| 200 | 10 | 10 | 20.0 | 0.34 | 10.8 |
| 300 | 10 | 10 | 20.0 | 0.32 | 10.6 |
| 500 | 9 | 10 | 18.1 | 0.29 | 14.2 |
| 800 | 0 | 10 | 4.3 | 0.27 | 26.7 |

## Taller stones (`stones`)

Cobblestones of one height for the whole attempt, at 0.4 m/s, a new layout each attempt. 30 attempts per height, because on tall stones tiny differences decide falls.

| stones cm | walked | attempts | average seconds up | actual m/s | J per metre |
|---|---|---|---|---|---|
| 1 | 30 | 30 | 20.0 | 0.39 | 10.8 |
| 2 | 29 | 30 | 19.6 | 0.38 | 11.6 |
| 3 | 21 | 30 | 16.6 | 0.36 | 13.5 |
| 4 | 4 | 30 | 9.0 | 0.33 | 17.4 |
| 5 | 0 | 30 | 3.9 | 0.3 | 19.2 |
| 6 | 0 | 30 | 2.2 | 0.28 | 26.6 |
| 8 | 0 | 30 | 1.3 | 0.24 | 26.1 |

## Thrown rocks (`rocks`)

A stone ball thrown at the body every 2 s from a random side at 4 m/s, on the flat floor at 0.4 m/s (see duck/rocks.py). The rock is taken away 1 s after each throw.

| rock g | walked | attempts | average seconds up | actual m/s | J per metre | rock diameter cm | hits |
|---|---|---|---|---|---|---|---|
| 25 | 10 | 10 | 20.0 | 0.39 | 10.7 | 2.7 | 90 |
| 50 | 10 | 10 | 20.0 | 0.39 | 10.8 | 3.4 | 90 |
| 100 | 10 | 10 | 20.0 | 0.39 | 11.6 | 4.2 | 90 |
| 125 | 9 | 10 | 19.1 | 0.39 | 11.9 | 4.6 | 86 |
| 150 | 2 | 10 | 12.3 | 0.38 | 13.3 | 4.9 | 55 |
| 175 | 0 | 10 | 6.7 | 0.37 | 13.4 | 5.1 | 28 |
| 200 | 0 | 10 | 6.2 | 0.36 | 13.4 | 5.3 | 26 |
| 400 | 0 | 10 | 2.6 | 0.35 | 14.8 | 6.7 | 10 |

## Films

Not kept in git; make them again with `--films` (and `--film GRAMS` for a rock round). `index.html` plays every challenge's films on one page.

- `speed/films/index.html`: attempt 1 of every value
- `backpack/films/index.html`: attempt 1 of every value
- `stones/films/index.html`: attempt 1 of every value
- `rocks/films/index.html`: attempt 1 of every value
- `rocks/films/round_150g/index.html`: every attempt of one rock size

## Notes

First run on 2026-10-30, for Duck 2.1, Duck 3 and Duck 3.1 with the same command, so the three can be compared. The films of the 150 g rocks round (`rocks/films/round_150g/index.html`: 2 attempts walk, 8 fall) were made with `--film 150`.

### Walked the full 20 s, three ducks side by side

| Challenge | Duck 2.1 (flat only) | Duck 3 (stones up to 1 cm) | Duck 3.1 (stones up to 3 cm) |
|---|---|---|---|
| Asked for 0.8 m/s | 10 of 10 | 3 of 10 | 10 of 10 |
| Asked for 0.9 m/s | 10 of 10 | 0 of 10 | 0 of 10 |
| Asked for 1.0 m/s | 1 of 10 | 0 of 10 | 0 of 10 |
| 500 g backpack | 9 of 10 | 10 of 10 | 9 of 10 |
| 800 g backpack | 8 of 10 | 0 of 10 | 0 of 10 |
| 2 cm stones | 13 of 30 | 26 of 30 | 29 of 30 |
| 3 cm stones | 0 of 30 | 0 of 30 | 21 of 30 |
| 4 cm stones | 0 of 30 | 0 of 30 | 4 of 30 |
| 100 g rocks | 9 of 10 | 5 of 10 | 10 of 10 |
| 150 g rocks | 4 of 10 | 0 of 10 | 2 of 10 |

### What it shows

- Each duck is good at what it trained for and not much beyond. Only Duck 3.1 walks on 3 cm stones (21 of 30), and it still manages 4 of 30 at 4 cm; its limit lies between 3 and 4 cm.
- Duck 2.1 is the strongest at speed and weight, even though it only ever trained on flat ground. It never learned the overshoot rule, so it always walks faster than asked (0.47 m/s when asked for 0.4) with a more forceful gait: it walks 0.9 m/s every time, and carries 800 g in 8 of 10 attempts, where both stone ducks fall every time.
- Duck 3 and Duck 3.1 slow down under a backpack rather than push harder (Duck 3.1: 0.39 m/s with nothing, 0.34 with 200 g, 0.27 with 800 g), and cannot run: asked for more than about 0.8 m/s they lean further forward than a step can catch and dive within a second or two.
- Against rocks no duck learned anything, since none met a push in training. All three hit a wall at about 175 g.
- On tall stones, single attempts are decided by tiny differences: adding a weightless box to the duck in a scratch test moved Duck 3.1 from 7 to 4 walks in 10 at 3 cm. That is why the stones challenge uses 30 attempts per height.
