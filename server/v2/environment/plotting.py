import os

import matplotlib.pyplot as plt
import numpy as np

from environment.state_handling import get_storage_path


def plot_average_results(rewards, num_steps, description):
    ema_rewards = __exp_moving_average(np.array(rewards), max(10 / num_steps, 1 / 1000))
    return __plot_combined_results(rewards, ema_rewards, num_steps, description)


def __exp_moving_average(data, alpha):
    # https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def __plot_combined_results(rewards, ema_rewards, num_steps, description):
    fig, ax1 = plt.subplots(1, 1)  # 1 rows for subplots, 1 column
    fig.set_size_inches(10, 3)  # width/height in inches
    fig.set_tight_layout(tight=True)

    # ==============================
    # PLOT REWARDS
    # ==============================

    ax1.scatter(range(1, num_steps + 1), rewards, s=5, color="blue")
    ax1.plot(range(1, num_steps + 1), ema_rewards, color="red")
    ax1.set_ylabel("Rewards")

    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.legend(["Abs", "EMA"])

    # ==============================
    # FINALIZE AND SAVE FIGURE
    # ==============================

    ax1.set_xlabel("Steps")
    fig.align_ylabels()

    fig_file = os.path.join(get_storage_path(), "results-fig={}.png".format(description))
    plt.savefig(fig_file)
    return fig_file
