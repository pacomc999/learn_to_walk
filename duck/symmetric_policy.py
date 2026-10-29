"""A brain that is exactly left/right symmetric (the NET method of Abdolhosseini and others, 2019).

One actor network F reads the situation s together with its mirror image Ms.
Its answers on the real view [s, Ms] drive the left leg; its answers on the mirrored
view [Ms, s], mirrored back, drive the right leg. The neck answers of both views are
added. The judge (critic) G scores both views and adds the two scores.
"""
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from duck.mirror import ACTION_ORDER, LEFT_MOTORS, NECK_MOTORS, mirror_action, mirror_observation

# How many motor commands the duck has, and how many one pass of F gives (5 side, 2 neck)
MOTOR_COUNT = 12
SIDE_COUNT = len(LEFT_MOTORS)
HEAD_COUNT = SIDE_COUNT + len(NECK_MOTORS)

# Stable-Baselines3 default starting gains: hidden layers, actor output, critic output
HIDDEN_GAIN = np.sqrt(2)
ACTOR_HEAD_GAIN = 0.01
CRITIC_HEAD_GAIN = 1.0


# A stack of Linear and Tanh layers with the given sizes, started the Stable-Baselines3 way
def build_trunk(input_size: int, hidden_sizes: list[int]) -> nn.Sequential:
    layers = []
    for size in hidden_sizes:
        layers += [nn.Linear(input_size, size), nn.Tanh()]
        input_size = size
    trunk = nn.Sequential(*layers)
    for layer in trunk:
        start_layer(layer, HIDDEN_GAIN)
    return trunk


# Give a Linear layer orthogonal weights with this gain and zero biases
def start_layer(layer: nn.Module, gain: float) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        layer.bias.data.fill_(0.0)


# A Linear output layer, started with the given gain
def build_head(input_size: int, output_size: int, gain: float) -> nn.Linear:
    head = nn.Linear(input_size, output_size)
    start_layer(head, gain)
    return head


# The two views of a situation: real [s, Ms] and mirrored [Ms, s]
def both_views(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mirrored = mirror_observation(features)
    return torch.cat([features, mirrored], dim=-1), torch.cat([mirrored, features], dim=-1)


# Both views stacked on the batch axis: the real views first, then the mirrored views
def stack_views(features: torch.Tensor) -> torch.Tensor:
    real_view, mirrored_view = both_views(features)
    return torch.cat([real_view, mirrored_view], dim=0)


# Place one pass of F (5 side outputs then 2 neck outputs) into 12 left side motor slots
def place_on_left(outputs: torch.Tensor) -> torch.Tensor:
    commands = outputs.new_zeros((*outputs.shape[:-1], MOTOR_COUNT))
    commands[..., LEFT_MOTORS] = outputs[..., :SIDE_COUNT]
    commands[..., NECK_MOTORS] = outputs[..., SIDE_COUNT:]
    return commands


# The symmetric actor and critic, used in place of Stable-Baselines3's MlpExtractor
class MirrorExtractor(nn.Module):
    # Build F (actor) and G (critic); each reads 2 x 39 numbers
    def __init__(self, features_dim: int, actor_sizes: list[int], critic_sizes: list[int]):
        super().__init__()
        self.actor_trunk = build_trunk(2 * features_dim, actor_sizes)
        self.actor_head = build_head(actor_sizes[-1], HEAD_COUNT, ACTOR_HEAD_GAIN)
        self.critic_trunk = build_trunk(2 * features_dim, critic_sizes)
        self.critic_head = build_head(critic_sizes[-1], 1, CRITIC_HEAD_GAIN)
        # The parent policy reads these sizes: 12 motor commands and 1 score
        self.latent_dim_pi = MOTOR_COUNT
        self.latent_dim_vf = 1

    # Both answers at once, as the parent policy expects; the views are built once and shared
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        views = stack_views(features)
        return self.actions_from_views(views)[0], self.score_from_views(views)

    # The 12 mean motor commands
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_cells(features)[0]

    # The 12 commands plus the two hidden layers of the pass on the real view
    def actor_cells(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actions_from_views(stack_views(features))

    # The judge's score: the sum of its scores for both views
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_from_views(stack_views(features))

    # Run F once on both stacked views, then combine the two halves into 12 commands
    def actions_from_views(self, views: torch.Tensor):
        layer1 = self.actor_trunk[1](self.actor_trunk[0](views))
        layer2 = self.actor_trunk[3](self.actor_trunk[2](layer1))
        real_outputs, mirrored_outputs = self.actor_head(layer2).chunk(2, dim=0)
        left_commands = place_on_left(real_outputs)
        right_commands = place_on_left(mirrored_outputs)
        real_count = views.shape[0] // 2
        return left_commands + mirror_action(right_commands), layer1[:real_count], layer2[:real_count]

    # Run G once on both stacked views and add the two scores
    def score_from_views(self, views: torch.Tensor) -> torch.Tensor:
        real_score, mirrored_score = self.critic_head(self.critic_trunk(views)).chunk(2, dim=0)
        return real_score + mirrored_score


# Read the actor and critic hidden sizes from net_arch (a dict, or one list for both)
def hidden_sizes(net_arch) -> tuple[list[int], list[int]]:
    if isinstance(net_arch, dict):
        return list(net_arch["pi"]), list(net_arch["vf"])
    return list(net_arch), list(net_arch)


# PPO policy that uses MirrorExtractor and wobbles both legs by the same amount
class SymmetricPolicy(ActorCriticPolicy):
    # Turn off the parent's own starting weights: MirrorExtractor starts itself
    def __init__(self, *args, **kwargs):
        kwargs["ortho_init"] = False
        super().__init__(*args, **kwargs)

    # Build the symmetric actor and critic in place of the usual MlpExtractor
    def _build_mlp_extractor(self) -> None:
        actor_sizes, critic_sizes = hidden_sizes(self.net_arch)
        self.mlp_extractor = MirrorExtractor(self.features_dim, actor_sizes, critic_sizes)
        self.mlp_extractor.to(self.device)

    # After the parent builds its layers, switch off its output layers and
    # remake the optimizer so it trains exactly the weights that remain
    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    # The wobble of each motor, averaged with its mirror partner so both legs match
    def symmetric_log_std(self) -> torch.Tensor:
        return (self.log_std + self.log_std[ACTION_ORDER]) / 2

    # The action distribution, built from the symmetric wobble
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.symmetric_log_std())
