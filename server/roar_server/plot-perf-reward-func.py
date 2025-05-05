import math
import os

from matplotlib import use as use_plot_backend
import matplotlib.pyplot as plt
import numpy as np

from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path


def pos(r):
    h = +0
    return list(map(lambda v: 10 * math.log10(v + 1) + abs(h), r))  # log 10
    # return list(map(lambda v: 10 * math.log(v + 1) + abs(h), r))  # log = log_e = ln


def neg(r):
    d = -20
    return list(map(lambda v: (d/max(v, 1)) - abs(d), r))


try:
    initialize_storage()
    use_plot_backend("template")

    x_lim = 500
    description = "pos-{}--neg-{}--h{}--d{}--x-{}".format("10log", "1overX", "+0", "-20", x_lim)

    x = np.linspace(0, x_lim)

    plt.plot(x, pos(x))
    plt.plot(x, neg(x))

    plt.ylabel("Reward")
    plt.xlabel("Encryption Rate")
    plt.legend(["Hidden", "Detected"])

    fig_file = os.path.join(get_storage_path(), "reward-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()
