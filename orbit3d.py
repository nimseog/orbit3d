from re import A
from timeit import repeat
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def gravity_eq_derivs(t, state, M):
    state = np.asarray(state)
    pos = state[:3]
    vel = state[3:]
    dist = np.linalg.norm(pos)
    acc_magnitude = gravitational_constant * M / dist**2
    acc_dir = -pos / dist
    acc = acc_magnitude * acc_dir
    return np.hstack((vel, acc))

def animate(i_frame, ax, xvals, yvals, zvals, xlims, ylims, zlims):
    ax.clear()
    r_earth = 6371e3
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_zlim(zlims[0], zlims[1])
    # ax.set_xlim(-3*r_earth, 3*r_earth)
    # ax.set_ylim(-3*r_earth, 3*r_earth)
    # ax.set_zlim(-3*r_earth, 3*r_earth)
    ax.plot(xvals[:i_frame+1], yvals[:i_frame+1], zvals[:i_frame+1], label='Trajectory')
    # ax.scatter([0], [0], [0], color='r', label='Earth')
    draw_earth_wireframe(ax, r_earth)
    ax.legend()

def draw_earth_wireframe(ax, r):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u)*np.sin(v)
    y = r * np.sin(u)*np.sin(v)
    z = r * np.cos(v)
    ax.plot_wireframe(x, y, z, color='r', label='Earth')

M = 5.972e24
AU = 1.495978707e11
atol = 1e-4*AU

init_pos = [9500e3, 1200e3, 0]
init_vel = [0, 6500, 3500]
init_state = init_pos + init_vel

t_end = 3600. * 24
dt = 1.
t = np.arange(0, t_end, dt)

solution = solve_ivp(gravity_eq_derivs, [0, t_end], init_state, t_eval=t,
    args=[M])
print(solution.message)

state = solution.y
x, y, z = state[0], state[1], state[2]
vx, vy, vz = state[2], state[3], state[4]

n_frames = 300
t_anim = np.linspace(0, t_end, n_frames)
x_anim = np.interp(t_anim, t, x)
y_anim = np.interp(t_anim, t, y)
z_anim = np.interp(t_anim, t, z)

axismax = max([max(abs(x_anim)), max(abs(y_anim)), max(abs(z_anim))])
axislims = [-axismax, axismax]

fig = plt.figure()
ax = plt.axes(projection='3d')
anim = FuncAnimation(fig, animate, frames=n_frames, interval=1., repeat=False,
    fargs=[ax, x_anim, y_anim, z_anim, axislims, axislims, axislims])
plt.show()


# ax.plot(x, y, z, label='Trajectory')
# # ax.scatter([0], [0], [0], label='Earth', color='r')

# # draw sphere
# r_earth = 6371e3
# # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# # xs = r_earth * np.cos(u)*np.sin(v)
# # ys = r_earth * np.sin(u)*np.sin(v)
# # zs = r_earth * np.cos(v)
# # ax.plot_wireframe(xs, ys, zs, color="r")
# ax.set_xlim(-3*r_earth, 3*r_earth)
# ax.set_ylim(-3*r_earth, 3*r_earth)
# ax.set_zlim(-3*r_earth, 3*r_earth)



# # ax.legend()
# plt.show()
