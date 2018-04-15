# %%
import numpy as np

from cycler import cycler

import matplotlib.pyplot as plt
import seaborn as sns

ax = np.arange(0.1, 0.9, 0.1)
ay = ax / (1 - ax)

by = 3 * ay
bx = by / (1 + by)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                           cycler('marker', ['o', 'o', 'o', 'o']) +
                           cycler('linestyle', ['-', '--', ':', '-.'])
                           ))

plt.title("Connection with Odds Ratio(OR) 1:3")
plt.xlabel("Probability")
plt.ylabel("Odds")

for i in range(len(ax)):
    plt.plot([ax[i], bx[i]], [ay[i], by[i]])

plt.show()

# Revert to original style.
plt.rcdefaults()
sns.set()
