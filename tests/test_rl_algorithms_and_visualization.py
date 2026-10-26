import gymnasium as gym
import matplotlib
import numpy as np
import pytest

from qvis.rl import QLearning, RLVisualizer, Sarsa


matplotlib.use("Agg")


class OneStepEnv(gym.Env):
    """Tiny deterministic env used for fast unit tests."""

    metadata = {}

    def __init__(self, reward: float = 1.0):
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.np_random = np.random.default_rng(0)
        self._reward = float(reward)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        return 0, {}

    def step(self, action):
        del action
        return 1, self._reward, True, False, {}


def _trained_agent(agent_cls, episodes=6, snapshot_interval=2):
    env = OneStepEnv(reward=1.0)
    agent = agent_cls(
        env=env,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.0,
        epsilon_decay=0.5,
        epsilon_min=0.01,
    )
    agent.train(
        episodes=episodes,
        max_steps=3,
        snapshot_interval=snapshot_interval,
    )
    return agent


@pytest.mark.parametrize("agent_cls", [QLearning, Sarsa])
def test_train_populates_rewards_history_and_resets_epsilon(agent_cls):
    agent = _trained_agent(agent_cls, episodes=6, snapshot_interval=2)

    assert len(agent.rewards) == 6
    assert all(reward == 1.0 for reward in agent.rewards)
    assert len(agent.history) == 3
    assert [item["episode"] for item in agent.history] == [2, 4, 6]
    assert agent._last_epsilon == pytest.approx(0.01)
    assert agent.epsilon == pytest.approx(agent._base_epsilon)


@pytest.mark.parametrize("agent_cls", [QLearning, Sarsa])
def test_train_rejects_non_positive_snapshot_interval(agent_cls):
    env = OneStepEnv()
    agent = agent_cls(env=env)

    with pytest.raises(ValueError, match="snapshot_interval must be positive"):
        agent.train(snapshot_interval=0)


def test_qlearning_update_uses_max_next_state_value():
    env = OneStepEnv(reward=0.0)
    agent = QLearning(env=env, alpha=0.1, gamma=0.9, epsilon=0.0)

    agent.q_table[1] = np.array([2.0, 5.0])
    agent.update(
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        terminated=False,
        truncated=False,
    )

    expected = 0.0 + 0.1 * (1.0 + 0.9 * 5.0 - 0.0)
    assert agent.q_table[0, 1] == pytest.approx(expected)


def test_sarsa_update_uses_policy_next_action_value(monkeypatch):
    env = OneStepEnv(reward=0.0)
    agent = Sarsa(env=env, alpha=0.2, gamma=0.9, epsilon=0.0)

    agent.q_table[1] = np.array([3.0, 10.0])

    # Force the next action to 0 so SARSA target differs from max-based target.
    monkeypatch.setattr(agent, "select_action", lambda _: 0)

    agent.update(
        state=0,
        action=0,
        reward=1.0,
        next_state=1,
        terminated=False,
        truncated=False,
    )

    expected = 0.0 + 0.2 * (1.0 + 0.9 * 3.0 - 0.0)
    assert agent.q_table[0, 0] == pytest.approx(expected)


def test_visualizer_plot_uses_final_state_and_snapshot():
    agent = _trained_agent(QLearning, episodes=4, snapshot_interval=2)
    visualizer = RLVisualizer(agent)

    fig_final, _, ax_q_final = visualizer.plot()
    assert "Final Q-Table" in ax_q_final.get_title()

    fig_snap, _, ax_q_snap = visualizer.plot(episode=2)
    assert "Q-Table at Episode 2" in ax_q_snap.get_title()

    matplotlib.pyplot.close(fig_final)
    matplotlib.pyplot.close(fig_snap)


def test_visualizer_plot_rejects_missing_snapshot():
    agent = _trained_agent(QLearning, episodes=4, snapshot_interval=2)
    visualizer = RLVisualizer(agent)

    with pytest.raises(ValueError, match="No snapshot for episode 3"):
        visualizer.plot(episode=3)


def test_visualizer_timelapse_validates_inputs():
    env = OneStepEnv()
    agent = QLearning(env=env)
    visualizer = RLVisualizer(agent)

    with pytest.raises(ValueError, match="No Q-table snapshots available"):
        visualizer.timelapse()

    trained = _trained_agent(QLearning, episodes=2, snapshot_interval=1)
    visualizer = RLVisualizer(trained)

    with pytest.raises(ValueError, match="reward_window must be positive"):
        visualizer.timelapse(reward_window=0)


def test_visualizer_timelapse_creates_animation_and_stores_reference():
    agent = _trained_agent(Sarsa, episodes=5, snapshot_interval=1)
    visualizer = RLVisualizer(agent)

    fig, animation = visualizer.timelapse(interval=5, reward_window=2)

    assert "SARSA" in fig._suptitle.get_text()
    assert animation is visualizer.animation
    assert animation is not None
    assert len(fig.axes) >= 2

    matplotlib.pyplot.close(fig)
