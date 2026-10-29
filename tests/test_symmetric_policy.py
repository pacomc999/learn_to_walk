import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from duck.duck_env import DuckWalkEnv
from duck.mirror import mirror_action, mirror_observation
from duck.symmetric_policy import SymmetricPolicy

# How closely mirrored answers must agree
TOLERANCE = 1e-5

# Motor numbers of the leg partners whose wobble must match
LEFT_HIP_SIDE, LEFT_KNEE = 3, 4
RIGHT_HIP_SIDE, RIGHT_KNEE = 8, 9


# A small PPO model that uses the symmetric brain
def make_model():
    return PPO(
        SymmetricPolicy,
        DummyVecEnv([DuckWalkEnv]),
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -1.0},
        n_steps=64,
        batch_size=32,
        device="cpu",
        seed=0,
    )


# Shake every weight so the symmetry checks cannot pass by accident (zero biases)
def perturb(policy):
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)


# A random batch of 16 situations
def random_situations():
    torch.manual_seed(1)
    return torch.randn(16, 39)


# A model whose weights have been shaken
@pytest.fixture
def shaken_policy():
    model = make_model()
    perturb(model.policy)
    return model.policy


# Check that the actor gives the mirror image commands for the mirror image situation
def assert_actor_is_symmetric(policy):
    situations = random_situations()
    with torch.no_grad():
        mirrored = policy.mlp_extractor.forward_actor(mirror_observation(situations))
        expected = mirror_action(policy.mlp_extractor.forward_actor(situations))
    assert torch.allclose(mirrored, expected, atol=TOLERANCE)


# The mirror image situation gives the mirror image commands
def test_mirrored_situation_gives_mirrored_commands(shaken_policy):
    assert_actor_is_symmetric(shaken_policy)


# The judge gives the same score to a situation and its mirror image
def test_judge_rates_mirror_images_equally(shaken_policy):
    situations = random_situations()
    with torch.no_grad():
        mirrored = shaken_policy.mlp_extractor.forward_critic(mirror_observation(situations))
        original = shaken_policy.mlp_extractor.forward_critic(situations)
    assert torch.allclose(mirrored, original, atol=TOLERANCE)


# The left and right legs wobble by the same amount
def test_left_and_right_wobble_match(shaken_policy):
    with torch.no_grad():
        shaken_policy.log_std.copy_(torch.randn(12))
        distribution = shaken_policy.get_distribution(random_situations())
    std = distribution.distribution.stddev
    assert torch.allclose(std[:, LEFT_KNEE], std[:, RIGHT_KNEE])
    assert torch.allclose(std[:, LEFT_HIP_SIDE], std[:, RIGHT_HIP_SIDE])


# The optimizer trains every weight, and the parent's output layers are switched off
def test_every_weight_is_trained():
    policy = make_model().policy
    trained = {p for group in policy.optimizer.param_groups for p in group["params"]}
    assert trained == set(policy.parameters())
    assert isinstance(policy.action_net, nn.Identity)
    assert isinstance(policy.value_net, nn.Identity)


# A new brain gives small commands
def test_new_brain_starts_calm():
    policy = make_model().policy
    with torch.no_grad():
        mean_actions = policy.mlp_extractor.forward_actor(random_situations())
    assert mean_actions.abs().max() < 0.1


# PPO can train the symmetric brain, and it stays symmetric
def test_ppo_learns_with_the_symmetric_brain():
    model = make_model()
    perturb(model.policy)
    before = [p.detach().clone() for p in model.policy.parameters()]
    model.learn(128)
    after = list(model.policy.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    assert_actor_is_symmetric(model.policy)


# The cell readout gives the same commands, and two layers of 256 cells
def test_brain_cells_match_the_commands(shaken_policy):
    situations = random_situations()
    with torch.no_grad():
        actions, layer1, layer2 = shaken_policy.mlp_extractor.actor_cells(situations)
        commands = shaken_policy.mlp_extractor.forward_actor(situations)
    assert torch.allclose(actions, commands)
    for layer in (layer1, layer2):
        assert layer.shape == (16, 256)
        assert layer.abs().max() <= 1.0
