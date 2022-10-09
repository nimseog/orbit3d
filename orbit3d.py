import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import gravitational_constant
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def gravity_eq_derivs(t, state, m_earth):
    state = np.asarray(state)
    pos = state[:3]
    vel = state[3:]
    acc = -pos * gravitational_constant * m_earth / np.linalg.norm(pos)**3
    return np.hstack((vel, acc))

def animate(i_frame, ax, xvals, yvals, zvals, xlims, ylims, zlims, r_earth):
    ax.clear()
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_zlim(zlims[0], zlims[1])
    ax.plot(xvals[:i_frame+1], yvals[:i_frame+1], zvals[:i_frame+1], label='Trajectory')
    draw_earth_wireframe(ax, r_earth)
    ax_text = '[1000 km]'
    ax.set_xlabel(ax_text)
    ax.set_ylabel(ax_text)
    ax.set_zlabel(ax_text)
    ax.legend()

def draw_earth_wireframe(ax, radius):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u)*np.sin(v)
    y = radius * np.sin(u)*np.sin(v)
    z = radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color='r', label='Earth')

m_earth = 5.972e24
r_earth = 6371e3

init_pos = [9500e3, 1200e3, 0]
init_vel = [0, 6500, 3500]
init_state = init_pos + init_vel

t_end = 3600. * 24
dt = 1.
t = np.arange(0, t_end, dt)

solution = solve_ivp(gravity_eq_derivs, [0, t_end], init_state, t_eval=t,
    args=[m_earth])
print(solution.message)

state = solution.y
x, y, z = state[0], state[1], state[2]

length_scaling = 1e-6
n_frames = 300
t_anim = np.linspace(0, t_end, n_frames)
x_anim = length_scaling * np.interp(t_anim, t, x)
y_anim = length_scaling * np.interp(t_anim, t, y)
z_anim = length_scaling * np.interp(t_anim, t, z)
r_earth_anim = length_scaling * r_earth

axismax = max([max(abs(x_anim)), max(abs(y_anim)), max(abs(z_anim))])
axislims = [-axismax, axismax]

fig = plt.figure()
ax = plt.axes(projection='3d')
anim = FuncAnimation(fig, animate, frames=n_frames, interval=1., repeat=False,
    fargs=[ax, x_anim, y_anim, z_anim, axislims, axislims, axislims, r_earth_anim])
plt.show()
