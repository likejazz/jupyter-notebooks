# %%
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


x = np.arange(-10, 10, 0.1)

plt.plot(x, sigmoid(relu(relu(relu(x)))), label='triple relu with single sigmoid')
plt.plot(x, sigmoid(sigmoid(sigmoid(sigmoid(x)))), label='quadruple sigmoids')
plt.legend()

plt.show()
