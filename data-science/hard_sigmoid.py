# %%
import matplotlib.pyplot as plt
import numpy as np


def hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# %%
x = np.arange(-10, 10, 0.1)

# %%
plt.plot(x, hard_sigmoid(x))
plt.plot(x, sigmoid(x))
plt.show()
