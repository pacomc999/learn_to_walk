"""The brain: the neural network that turns what the walker senses into motor commands.

A brain design says how many layers of thinking cells it has, how many cells each layer has,
how a cell responds (its activation), which senses it gets, and whether it is left/right
symmetric. The symmetric brain is learn_to_walk's (the NET method of Abdolhosseini and others,
2019), made to work for any animal: one network reads the situation and its mirror image, its
answers drive the left side, its answers on the mirrored view drive the right side, and the
middle motors (neck, tail) get both answers combined.
"""
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from studio.creatures.kit import Setting, SettingGroup, clean_values, default_values, switch

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU}

# Starting gains of Stable-Baselines3: hidden layers, actor output, critic output
HIDDEN_GAIN = np.sqrt(2)
ACTOR_HEAD_GAIN = 0.01
CRITIC_HEAD_GAIN = 1.0

# The brain's settings as the Brain screen shows them (the layers have their own editor)
SETTING_GROUPS = (
    SettingGroup("cells", "Thinking cells", "How each cell in the network reacts to what it receives.", (
        Setting("activation", "Cell response", "choice", "tanh",
                "Smooth cells (tanh) are calm and reliable for walking. Sharp cells (ReLU) learn fast but can be jumpy.",
                choices=(("tanh", "Smooth (tanh)"), ("relu", "Sharp (ReLU)"), ("elu", "Soft sharp (ELU)"),
                         ("silu", "Swishy (SiLU)"))),
        switch("symmetric", "Left/right symmetric", "The left and right side share one way of thinking, mirrored. "
               "Learns walking gaits much faster. Turn off to let each side think for itself."),
        switch("judge_same", "Judge has the same shape", "Learning uses a second network, the judge, that guesses "
               "how many points are coming. Usually it has the same layers as the brain."),
    )),
    SettingGroup("senses", "Senses", "What the brain gets to feel. Every animal always feels its balance, speed "
                 "and joint angles.", (
        switch("feel_feet", "Feel the feet touching the ground", "Tells the brain which feet are on the ground."),
        switch("see_ground", "See the ground ahead", "Seven heights of the ground just in front of the walker. "
               "Helps a lot on stairs and rocks.", default=False),
    )),
    SettingGroup("habits", "Starting habits", "How the brain behaves before it has learned anything.", (
        Setting("start_wobble", "Starting curiosity", "choice", "calm",
                "How wildly it tries things at the start. Calm is right for most walkers; very stiff muscles need "
                "less, slow learners can use more.",
                choices=(("tiny", "Very calm"), ("calm", "Calm"), ("lively", "Lively"), ("wild", "Wild"))),
    )),
)

# The starting wobble as the log of the spread of random tries per motor
START_WOBBLE = {"tiny": -2.5, "calm": -1.0, "lively": -0.5, "wild": 0.0}

DEFAULT_LAYERS = [256, 256]


# A brain design with every value checked
def clean_design(design: dict | None) -> dict:
    design = design or {}
    values = clean_values(SETTING_GROUPS, design.get("values", {}))
    layers = _clean_layers(design.get("layers"), DEFAULT_LAYERS)
    judge = layers if values["judge_same"] else _clean_layers(design.get("judge_layers"), layers)
    return {"values": values, "layers": layers, "judge_layers": judge}


def _clean_layers(layers, fallback: list[int]) -> list[int]:
    if not isinstance(layers, list) or not layers:
        return list(fallback)
    cleaned = []
    for size in layers[:6]:
        try:
            cleaned.append(int(min(max(int(size), 4), 1024)))
        except (TypeError, ValueError):
            continue
    return cleaned or list(fallback)


def default_design() -> dict:
    return {"values": default_values(SETTING_GROUPS), "layers": list(DEFAULT_LAYERS),
            "judge_layers": list(DEFAULT_LAYERS)}


# The senses switched on by a design, in the environment's words
def design_senses(design: dict) -> dict:
    values = design["values"]
    return {"feet": values["feel_feet"], "ground": values["see_ground"]}


# How many numbers the brain learns (both networks), for the Brain screen
def count_weights(observation_size: int, motor_count: int, design: dict) -> dict:
    values = design["values"]
    symmetric = values["symmetric"]
    inputs = observation_size * (2 if symmetric else 1)

    def stack(sizes, outputs):
        total, previous = 0, inputs
        for size in sizes:
            total += previous * size + size
            previous = size
        return total + previous * outputs + outputs

    actor = stack(design["layers"], motor_count)
    critic = stack(design["judge_layers"], 1)
    return {"brain": actor, "judge": critic, "total": actor + critic}


# ---------- Networks ----------

def start_layer(layer: nn.Module, gain: float) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        layer.bias.data.fill_(0.0)


def build_trunk(input_size: int, sizes: list[int], activation) -> nn.Sequential:
    layers = []
    for size in sizes:
        layers += [nn.Linear(input_size, size), activation()]
        input_size = size
    trunk = nn.Sequential(*layers)
    for layer in trunk:
        start_layer(layer, HIDDEN_GAIN)
    return trunk


def build_head(input_size: int, output_size: int, gain: float) -> nn.Linear:
    head = nn.Linear(input_size, output_size)
    start_layer(head, gain)
    return head


# The symmetric actor and critic, used in place of Stable-Baselines3's MlpExtractor
class MirrorExtractor(nn.Module):
    def __init__(self, features_dim: int, actor_sizes: list[int], critic_sizes: list[int], mirror: dict,
                 activation):
        super().__init__()
        self.mirror = mirror
        self.motor_count = len(mirror["action_order"])
        self.left = list(mirror["left_motors"])
        self.middle = list(mirror["middle_motors"])
        outputs = len(self.left) + len(self.middle)
        self.actor_trunk = build_trunk(2 * features_dim, actor_sizes, activation)
        self.actor_head = build_head(actor_sizes[-1], outputs, ACTOR_HEAD_GAIN)
        self.critic_trunk = build_trunk(2 * features_dim, critic_sizes, activation)
        self.critic_head = build_head(critic_sizes[-1], 1, CRITIC_HEAD_GAIN)
        self.latent_dim_pi = self.motor_count
        self.latent_dim_vf = 1
        self.register_buffer("obs_index", torch.as_tensor(mirror["observation_order"]), persistent=False)
        self.register_buffer("obs_sign", torch.as_tensor(mirror["observation_signs"], dtype=torch.float32),
                             persistent=False)
        self.register_buffer("act_index", torch.as_tensor(mirror["action_order"]), persistent=False)
        self.register_buffer("act_sign", torch.as_tensor(mirror["action_signs"], dtype=torch.float32),
                             persistent=False)

    def _mirror_obs(self, features):
        return features[..., self.obs_index] * self.obs_sign

    def _mirror_act(self, actions):
        return actions[..., self.act_index] * self.act_sign

    # Both views stacked on the batch axis: real [s, Ms] first, then mirrored [Ms, s]
    def _views(self, features):
        mirrored = self._mirror_obs(features)
        real_view = torch.cat([features, mirrored], dim=-1)
        mirrored_view = torch.cat([mirrored, features], dim=-1)
        return torch.cat([real_view, mirrored_view], dim=0)

    def _place(self, outputs):
        commands = outputs.new_zeros((*outputs.shape[:-1], self.motor_count))
        commands[..., self.left] = outputs[..., :len(self.left)]
        commands[..., self.middle] = outputs[..., len(self.left):]
        return commands

    def _actions(self, views):
        real, mirrored = self.actor_head(self.actor_trunk(views)).chunk(2, dim=0)
        return self._place(real) + self._mirror_act(self._place(mirrored))

    def _score(self, views):
        real, mirrored = self.critic_head(self.critic_trunk(views)).chunk(2, dim=0)
        return real + mirrored

    def forward(self, features):
        views = self._views(features)
        return self._actions(views), self._score(views)

    def forward_actor(self, features):
        return self._actions(self._views(features))

    def forward_critic(self, features):
        return self._score(self._views(features))


# PPO policy that uses MirrorExtractor and wobbles each motor like its mirror partner
class SymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, mirror: dict | None = None, activation_name: str = "tanh", **kwargs):
        object.__setattr__(self, "mirror", mirror)
        object.__setattr__(self, "activation_name", activation_name)
        kwargs["ortho_init"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        arch = self.net_arch
        actor = list(arch["pi"]) if isinstance(arch, dict) else list(arch)
        critic = list(arch["vf"]) if isinstance(arch, dict) else list(arch)
        self.mlp_extractor = MirrorExtractor(self.features_dim, actor, critic, self.mirror,
                                             ACTIVATIONS[self.activation_name])
        self.mlp_extractor.to(self.device)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(mirror=self.mirror, activation_name=self.activation_name)
        return data

    def symmetric_log_std(self):
        order = torch.as_tensor(self.mirror["action_order"], device=self.log_std.device)
        return (self.log_std + self.log_std[order]) / 2

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.symmetric_log_std())


# The policy class and its settings for PPO
def policy_for(design: dict, mirror: dict):
    values = design["values"]
    kwargs = {"net_arch": {"pi": list(design["layers"]), "vf": list(design["judge_layers"])},
              "log_std_init": START_WOBBLE[values["start_wobble"]]}
    if values["symmetric"]:
        kwargs.update(mirror=mirror, activation_name=values["activation"])
        return SymmetricPolicy, kwargs
    kwargs["activation_fn"] = ACTIVATIONS[values["activation"]]
    return "MlpPolicy", kwargs
