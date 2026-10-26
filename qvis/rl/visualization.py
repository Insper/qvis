

import matplotlib.pyplot as plt
import numpy as np
from .base import AbstractRLAgent
from matplotlib.animation import FuncAnimation

class RLVisualizer:
    """Visualize learning progress for tabular RL agents."""

    def __init__(self, agent: AbstractRLAgent):
        self.agent = agent
        self.animation = None

    def plot(self, episode=None, window=100):
        fig, (ax_q, ax_r) = plt.subplots(
                    1, 2,
                    figsize=(14, 6),
                    gridspec_kw={"width_ratios": [1, 1.4]},
                )
        
        if episode is None:
            q_table = self.agent.q_table
            rewards = self.agent.rewards
            title = "Final Q-Table"
        else:
            snapshot = next(
                (
                    item for item in self.agent.history
                    if item["episode"] == episode
                ),
                None,
            )

            if snapshot is None:
                raise ValueError(
                    f"No snapshot for episode {episode}"
                )

            q_table = snapshot["q_table"]
            rewards = snapshot["rewards"]
            title = f"Q-Table at Episode {episode}"

        title += f" - epsilon {self.agent._last_epsilon}"

        image = ax_q.imshow(q_table, aspect="auto", cmap="viridis")

        # Print Q-value inside each cell
        for state in range(q_table.shape[0]):
            for action in range(q_table.shape[1]):
                value = q_table[state, action]

                ax_q.text(
                    action,
                    state,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if image.norm(value) < 0.5 else "black",
                    fontsize=9,
                )

        ax_q.set_title(title)
        ax_q.set_xlabel("Actions")
        ax_q.set_ylabel("States")

        ax_r.plot(rewards, alpha=0.3, label="Episode reward")

        if (len(rewards) >= window):
            moving_avg = np.convolve(
                rewards, 
                np.ones(window) / window,
                mode="valid"
            )

            ax_r.plot(
                np.arange(window - 1, len(rewards)),
                moving_avg,
                label="Moving avarage",
            )

        ax_r.set_xlabel("Episode")
        ax_r.set_ylabel("Total reward")
        ax_r.legend()
        fig.tight_layout()
        return fig, ax_r, ax_q

    def plot_rewards(self, ):
        
        return fig, ax

    def timelapse(
        self,
        interval: int = 10,
        reward_window: int = 100,
        repeat: bool = False,
        save_path: str | None = None,
    ):
        """Animate Q-table evolution and rewards together."""

        history = self.agent.history
        rewards = np.asarray(self.agent.rewards, dtype=float)

        if not history:
            raise ValueError("No Q-table snapshots available.")

        if reward_window <= 0:
            raise ValueError("reward_window must be positive.")

        episodes = np.asarray([
            item["episode"] for item in history
        ])

        tables = np.stack([
            item["q_table"] for item in history
        ])

        epsilons = np.asarray([
            item["epsilon"] for item in history
        ])

        if episodes[-1] > len(rewards):
            raise ValueError("Snapshots exceed recorded rewards.")

        # Moving average aligned with episode numbers.
        cumulative = np.cumsum(
            np.concatenate(([0.0], rewards))
        )

        counts = np.minimum(
            np.arange(1, len(rewards) + 1),
            reward_window,
        )

        indices = np.arange(1, len(rewards) + 1)
        starts = np.maximum(0, indices - reward_window)

        moving_avg = (
            cumulative[indices] - cumulative[starts]
        ) / counts

        fig, (ax_q, ax_r) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [1, 1.4]},
        )

        algorithm = type(self.agent).__name__
        algorithm = {"QLearning": "Q-Learning", "Sarsa": "SARSA"}.get(
            algorithm, algorithm
        )
        fig.suptitle(f"{algorithm} Training Progress")

        # ---- Q-table heatmap ----

        vmin = float(tables.min())
        vmax = float(tables.max())

        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        image = ax_q.imshow(
            tables[0],
            cmap="viridis",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax_q.set_title("Q-Table")
        ax_q.set_xlabel("Actions")
        ax_q.set_ylabel("States")

        ax_q.set_xticks(np.arange(tables.shape[2]))
        ax_q.set_yticks(np.arange(tables.shape[1]))

        fig.colorbar(image, ax=ax_q, label="Q-value")

        texts = []

        for state in range(tables.shape[1]):
            row = []

            for action in range(tables.shape[2]):
                text = ax_q.text(
                    action,
                    state,
                    "",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                row.append(text)

            texts.append(row)

        # ---- Reward chart ----

        x = np.arange(1, len(rewards) + 1)

        reward_line, = ax_r.plot(
            [],
            [],
            alpha=0.3,
            label="Episode reward",
        )

        average_line, = ax_r.plot(
            [],
            [],
            linewidth=2,
            label=f"Moving average ({reward_window})",
        )

        cursor = ax_r.axvline(
            episodes[0],
            color="gray",
            linestyle="--",
            label="Current episode",
        )

        ax_r.set_xlim(1, max(2, len(rewards)))

        ymin = min(0.0, float(rewards.min()))
        ymax = max(1.0, float(rewards.max()))
        margin = max(0.1, (ymax - ymin) * 0.1)

        ax_r.set_ylim(ymin - margin, ymax + margin)

        ax_r.set_xlabel("Episode")
        ax_r.set_ylabel("Reward")
        ax_r.set_title("Training Rewards")
        ax_r.legend(loc="upper left")
        ax_r.grid(alpha=0.2)

        episode_title = fig.text(
            0.5,
            0.94,
            "",
            ha="center",
            fontsize=11,
        )

        # ---- Animation update ----

        def update(frame):
            episode = int(episodes[frame])
            q_table = tables[frame]
            epsilon = float(epsilons[frame])

            image.set_data(q_table)

            for state in range(q_table.shape[0]):
                for action in range(q_table.shape[1]):
                    value = q_table[state, action]

                    text = texts[state][action]
                    text.set_text(f"{value:.3f}")

                    r, g, b, _ = image.cmap(
                        image.norm(value)
                    )

                    luminance = (
                        0.2126 * r
                        + 0.7152 * g
                        + 0.0722 * b
                    )

                    text.set_color(
                        "black" if luminance > 0.5
                        else "white"
                    )

            reward_line.set_data(
                x[:episode],
                rewards[:episode],
            )

            average_line.set_data(
                x[:episode],
                moving_avg[:episode],
            )

            cursor.set_xdata([episode, episode])

            episode_title.set_text(
                f"Episode {episode} / {len(rewards)} | Epsilon {epsilon:.4f}"
            )

            return (
                image,
                reward_line,
                average_line,
                cursor,
                episode_title,
                *[text for row in texts for text in row],
            )

        update(0)

        animation = FuncAnimation(
            fig,
            update,
            frames=len(history),
            interval=interval,
            repeat=repeat,
            blit=False,
        )

        # Retain a reference while the window is open.
        self.animation = animation

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if save_path:
            animation.save(
                save_path,
                fps=max(1, round(1000 / interval)),
            )

        return fig, animation

        


