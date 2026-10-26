from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

class AbstractRLAgent(ABC):
    """Base class for tabular reinforcement learning agents."""

    def __init__(
            self, 
            env: gym.Env,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon: float = 0.8,
            epsilon_decay=0.999,
            epsilon_min=0.01
    ):
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("Expected a discrete observation space.")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("Expected a discrete action space.")

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._base_epsilon = epsilon
        self._last_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((
            env.observation_space.n,
            env.action_space.n
        ))

        self.rewards = []
        self.history = []

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Choose an action from the current state."""


    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Update the learned value estimates."""

    def predict(self, state: int) -> int:
        """Return the greedy action"""
        return int(np.argmax(self.q_table[state]))

    

    def train(
            self,
            episodes: int = 1000,
            max_steps: int = 1000,
            snapshot_interval: int = 10,
    ):
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")

        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = (
                    self.env.step(action)
                )

                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    truncated,
                )

                total_reward += float(reward)
                state = next_state

                if terminated or truncated:
                    break

            self.rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if (episode + 1) % snapshot_interval == 0:
                self.history.append({
                    "episode": episode + 1,
                    "q_table": self.q_table.copy(),
                    "rewards": self.rewards.copy(),
                    "epsilon": self.epsilon
                })

        self._last_epsilon = self.epsilon
        self.epsilon = self._base_epsilon

        return self