
import numpy as np

from .base import AbstractRLAgent

class Sarsa(AbstractRLAgent):
    """Tabular Sarsa with epsilon-greedy exploration."""

    def select_action(self, state: int) -> int:
        if self.env.np_random.random() < self.epsilon:
            return int(self.env.action_space.sample())


        return self.predict(state)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool
    ) -> None:
        old_value = self.q_table[state, action]

        future_value = (
            0.0 if terminated
            else self.q_table[next_state, self.select_action(next_state)]
        )

        target = reward + self.gamma * future_value

        self.q_table[state, action] = (
            old_value + self.alpha * (target - old_value)
        )