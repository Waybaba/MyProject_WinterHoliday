from math import sin, cos
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

g = 9.8
leng = 1.0
b_const = 0.2

t = np.arange(0, 20, 0.1)

#track = odeint(pendulum_equations2, (1.0, 0), t, args=(leng, b_const))
xdata = [1,2,3,4,5]
ydata = [1,2,3,4,5]

fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# fig = plt.figure()
fig,ax = fig.add_subplot(111, projection='3d')
# ax.plot([-3,3], [-3,3], [-3,3])
# ax.grid()
line,= ax.plot([3,3], [-3,3], [-3,3])


def init():
    #设置上下限
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_ylim(-2, 2)

    return line

def update(i):
    xdata = [1, 2, 3, 4, 5,6,8,45,5,5,2,4,6,2,5,5,6,6,6,5,0]
    ydata = [1, 2, 3, 4, 5,5,5,2,3,6,5,4,5,5,5,63,6,5,4,1,2,2,2,3,3]
    newx = [0, xdata[i]]
    newy = [0, ydata[i]]
    newz = [0,0]

    return line

ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, interval=20,blit=False)
#ani.save('single_pendulum_decay.gif', writer='imagemagick', fps=100)
# ani.save('single_pendulum_nodecay.gif', writer='imagemagick', fps=100)
plt.show()