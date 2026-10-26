import gymnasium as gym
from qvis.rl import QLearning, RLVisualizer, Sarsa
from matplotlib import pyplot as plt

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)

    if (not isinstance(env, gym.Env)):
        raise f"It was not possible to get enviroment {env}"

    agent = Sarsa(
        env=env,
        alpha=0.1,
        gamma=0.99,
        epsilon_decay=0.999,
        epsilon=0.8
    )

    agent.train(
        episodes=10000,
        max_steps=200,
        snapshot_interval=100,
    )

    visualizer = RLVisualizer(agent)

    visualizer.timelapse()

    plt.show()

if __name__ == "__main__":
    main()