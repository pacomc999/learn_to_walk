"""How a walker learns: the settings of a training session, in plain words.

Training uses PPO (proximal policy optimisation): the walker practises in several worlds at
once, then studies what happened and nudges its brain toward the moves that earned more
points. These settings decide how long it practises, how big each nudge is, and so on.
"""
from studio.creatures.kit import Setting, SettingGroup, clean_values, default_values, switch


def number(key, label, help, default, minimum, maximum, step, unit) -> Setting:
    return Setting(key, label, "number", default, help, minimum, maximum, step, unit)


def choice(key, label, help, default, choices) -> Setting:
    return Setting(key, label, "choice", default, help, choices=choices)


SETTING_GROUPS = (
    SettingGroup("length", "How long", "More practice usually means better walking. Training stops early by "
                 "itself if it stops getting better.", (
        number("million_steps", "Practice steps", "One step is a fiftieth of a second of practice in one world. "
               "A first wobbly walk often comes after 1 to 2 million.", 3.0, 0.1, 100.0, 0.1, "million"),
        number("parallel_worlds", "Worlds at once", "How many copies of the world practise side by side. Use about "
               "as many as your computer has processor cores.", 8, 1, 32, 1, ""),
        number("patience", "Patience", "Stop when this many exams in a row bring no new best. 0 never stops "
               "early.", 12, 0, 50, 1, "exams"),
        number("exam_every", "Exam every", "How often the walker takes a fixed test without any random wobble.",
               250, 50, 2000, 50, "thousand steps"),
    )),
    SettingGroup("study", "How it studies", "The details of each lesson. The defaults are the ones that taught "
                 "the learn_to_walk ducks.", (
        choice("learning_rate", "Learning speed", "How big each change to the brain is. Too fast and it forgets, "
               "too slow and it takes forever.", 0.0003,
               ((0.0001, "Careful"), (0.0003, "Normal"), (0.001, "Bold"), (0.003, "Very bold"))),
        switch("adaptive", "Adjust learning speed by itself", "Speeds up or slows down each lesson so the brain "
               "changes by a steady amount. Recommended."),
        choice("steps_per_update", "Practice before each lesson", "Steps of practice in each world before the brain "
               "studies them.", 1024, ((256, "Short (256)"), (512, "Medium (512)"), (1024, "Long (1,024)"),
                                      (2048, "Very long (2,048)"), (4096, "Huge (4,096)"))),
        choice("batch_size", "Study batch", "How many moments it looks at together while studying.", 256,
               ((64, "64"), (128, "128"), (256, "256"), (512, "512"), (1024, "1,024"), (2048, "2,048"))),
        number("epochs", "Study rounds", "How many times it goes over each batch of practice.", 10, 1, 30, 1, ""),
        choice("gamma", "Foresight", "How far ahead it plans. Walking needs about two seconds.", 0.99,
               ((0.95, "Short (half a second)"), (0.98, "Medium (one second)"), (0.99, "Long (two seconds)"),
                (0.995, "Very long (four seconds)"))),
        number("curiosity", "Curiosity bonus", "Extra points for trying new things. A little can help a walker that "
               "gets stuck.", 0.0, 0.0, 0.05, 0.001, ""),
        number("seed", "Lucky number", "The seed of all the randomness. The same number repeats the same run.",
               0, 0, 9999, 1, ""),
    )),
)


def clean_learning(values: dict | None) -> dict:
    return clean_values(SETTING_GROUPS, values)


def default_learning() -> dict:
    return default_values(SETTING_GROUPS)


# The PPO settings from the learning values
def ppo_settings(values: dict) -> dict:
    worlds = int(values["parallel_worlds"])
    steps = int(values["steps_per_update"])
    batch = int(values["batch_size"])
    while (worlds * steps) % batch and batch > 32:  # the batch must divide the practice of one lesson
        batch //= 2
    return {"n_steps": steps, "batch_size": batch, "n_epochs": int(values["epochs"]),
            "learning_rate": float(values["learning_rate"]), "gamma": float(values["gamma"]), "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": float(values["curiosity"])}
