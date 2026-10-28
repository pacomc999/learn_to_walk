import numpy as np
from gymnasium.utils.env_checker import check_env
import duck.duck_env as duck_env_module
from duck.duck_env import DuckWalkEnv

def test_passes_gymnasium_checks():
    check_env(DuckWalkEnv(), skip_render_check=True)

def test_spaces():
    env = DuckWalkEnv()
    assert env.observation_space.shape == (37,)
    assert env.action_space.shape == (12,)
    assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1

def test_same_seed_same_start():
    env = DuckWalkEnv()
    first, _ = env.reset(seed=0)
    second, _ = env.reset(seed=0)
    assert np.array_equal(first, second)

def test_attempt_stops_after_20_seconds(monkeypatch):
    monkeypatch.setattr(duck_env_module, "has_fallen", lambda *args, **kwargs: False)
    env = DuckWalkEnv()
    env.reset(seed=0)
    for step in range(1, 1001):
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        assert not terminated
        assert truncated == (step == 1000)

def test_lying_down_ends_the_attempt():
    env = DuckWalkEnv()
    env.reset(seed=0)
    env.data.qpos[2] = 0.05
    _, _, terminated, _, _ = env.step(np.zeros(12))
    assert terminated

def test_extreme_actions_stay_finite():
    env = DuckWalkEnv()
    obs, _ = env.reset(seed=0)
    for step in range(1000):
        action = np.ones(12) if step % 2 else -np.ones(12)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()

def test_render_gives_an_image():
    env = DuckWalkEnv(render_mode="rgb_array", width=160, height=120)
    env.reset(seed=0)
    assert env.render().shape == (120, 160, 3)
