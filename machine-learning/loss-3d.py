# %%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import collections


def mse(w, b, points):
    sse = 0
    for i in range(len(points)):
        sse += (points[i].y - (w * points[i].x + b)) ** 2

    return sse / len(points)


Point = collections.namedtuple('Point', ['x', 'y'])
points = [Point(1, 1), Point(2, 5), Point(5, 3)]

fig = plt.figure()
ax = Axes3D(fig, elev=30, azim=60)

ws = np.linspace(-3, 3, 100)
bs = np.linspace(-3, 3, 100)

W, B = np.meshgrid(ws, bs)
zs = np.array([mse(wp, bp, points)
               for wp, bp in zip(np.ravel(W), np.ravel(B))])
Z = zs.reshape(W.shape)

ax.plot_surface(W, B, Z, rstride=1, cstride=1, color='b', alpha=0.5)

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('error')

plt.show()

# %% Draw linear graph with the smallest error value.
wm = np.ravel(W)[np.argmin(zs)]
bm = np.ravel(B)[np.argmin(zs)]

for i in range(len(points)):
    plt.scatter(points[i].x, points[i].y)
x = np.arange(0, 6, 0.1)
y = wm * x + bm

plt.text(0, 0, "y = %.2f * x + %.2f\nloss: %.4f" % (wm, bm, np.min(zs)))

plt.plot(x, y)
plt.show()
