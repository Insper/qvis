# Q-Vis: Visualization of Tabular Reinforcement Learning Algorithms

Q-Vis is a Python library for visualizing the training process of tabular reinforcement learning algorithms such as Q-Learning and Sarsa. It provides an easy way to observe how the agent's policy evolves over time in various environments.

## Features
- Visualize the evolution of Q-values and policies over time.
- Support for popular tabular RL algorithms like Q-Learning and Sarsa.
- Compatible with OpenAI Gym environments.

## Installation
```bash
pip install qvis
```

## Usage
```python
import gymnasium as gym
from qvis.rl import QLearning, RLVisualizer
from matplotlib import pyplot as plt

env = gym.make("Taxi-v4")
agent = QLearning(env=env, alpha=0.1, gamma=0.99, epsilon=0.8, epsilon_decay=0.999)
agent.train(episodes=10000, max_steps=200, snapshot_interval=100)
visualizer = RLVisualizer(agent)
visualizer.timelapse()
plt.show()
```

## Examples

You can find example scripts for different environments in the `tests` directory, such as:
- `CliffWalkingQLearning.py` for the CliffWalking environment using Q-Learning.
- `CliffWalkingSarsa.py` for the CliffWalking environment using Sarsa.
- `FrozenLake.py` for the FrozenLake environment using Q-Learning.

From the root directory of the project, you can run these example scripts to see Q-Vis in action. For instance:
```bash
python -m tests.CliffWalkingQLearning
python -m tests.CliffWalkingSarsa
python -m tests.FrozenLake
```

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions to Q-Vis are welcome! If you find a bug or have an idea for a new feature, feel free to open an issue or submit a pull request.

When contributing, please ensure that your code follows the existing style and includes appropriate tests for any new functionality.

Please also update the documentation as needed to reflect any changes or additions to the library.
