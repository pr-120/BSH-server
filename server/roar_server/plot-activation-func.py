import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as use_plot_backend

from environment.state_handling import initialize_storage, cleanup_storage, get_storage_path


def act1(x):
    return list(map(lambda v: v if v > 0 else 0, x))  # ReLU


def act2(x):
    return 1 / (1 + np.exp(-x))  # Logistic
    # return x / (1 + np.exp(-x))  # SiLU


try:
    initialize_storage()
    use_plot_backend("template")  # required for matplotlib, throws error otherwise

    x_lim = 4
    # description = "a1-{}--a2-{}--x-{}".format("ReLU", "SiLU", x_lim)
    description = "a1-{}--a2-{}--x-{}".format("ReLU", "Logistic", x_lim)

    x = np.linspace(-x_lim, x_lim, num=500)

    plt.plot(x, act1(x))
    plt.plot(x, act2(x))
    # plt.plot(x, act2(x), c="red")

    # plt.legend(["ReLU", "SiLU"])
    plt.legend(["ReLU", "Logistic"])

    plt.xlim(-x_lim, x_lim)
    plt.grid()

    fig_file = os.path.join(get_storage_path(), "activation-func={}.png".format(description))
    plt.savefig(fig_file)
finally:
    cleanup_storage()
