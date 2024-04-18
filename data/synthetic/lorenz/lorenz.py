import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=8/3, R=1):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    # x_dot = s*(y - x)
    # y_dot = r*x - y - x*z
    # z_dot = x*y - b*z
    x_dot = -1 * x - y
    y_dot = - x*z
    z_dot = R + x*y
    return x_dot, y_dot, z_dot

def sample(x, window = 10):
    xlist = x.ravel().tolist()
    ts = []
    for i in range(len(xlist)):
        if i % window == 0 :
            ts.append(xlist[i])

    return ts

dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty((num_steps + 1,))
ys = np.empty((num_steps + 1,))
zs = np.empty((num_steps + 1,))

# Set initial values
# xs[0], ys[0], zs[0] = (0., 1., 1.05)
xs[0] = 1
ys[0] = -1.1
zs[0] = 0.01

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

np.save('data/synthetic/lorenz/lorenz', xs)
# Plot
# fig = plt.figure(figsize=(20,6))
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')

# # ax.plot(xs, ys, zs, lw=0.5)
# # ax.set_xlabel("X Axis")
# # ax.set_ylabel("Y Axis")
# # ax.set_zlabel("Z Axis")
# # ax.set_title("Lorenz Attractor")

# xs = sample(xs)
# ys = sample(ys)
# zs = sample(zs)
# plt.subplot(311)
# plt.plot(xs)
# plt.subplot(312)
# plt.plot(ys)
# plt.subplot(313)
# plt.plot(zs)
# plt.show()