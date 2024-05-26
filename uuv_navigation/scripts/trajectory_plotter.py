#!/usr/bin/env python3

import rospkg
import numpy as np
from yaml import safe_load
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

"""
READING WAYPOINTS
"""

with open(rospkg.RosPack().get_path("uuv_navigation") + f"/config/semi_olympic_waypoints.yaml", 'r') as file:
    cfg = safe_load(file)

waypoints = np.array(cfg["waypoints"])

# Sample data points
x = waypoints[:, 0]
y = waypoints[:, 1]
z = waypoints[:, 2]
theta = np.arange(0, len(x))

"""
CREATING INTERPOLATORS
"""

# Create the PCHIP interpolator using the given data points
x_interpolator = PchipInterpolator(theta, x)
y_interpolator = PchipInterpolator(theta, y)
z_interpolator = PchipInterpolator(theta, z)

x_dot_interpolator = x_interpolator.derivative()
y_dot_interpolator = y_interpolator.derivative()
z_dot_interpolator = z_interpolator.derivative()

"""
PATH GENERATION
"""

# Generate a series of 100 x-values from 0 to 9 for a smooth plot
t_fine = np.linspace(0, len(x)-1, 100)

# Compute the interpolated y-values
x_fine = x_interpolator(t_fine)
y_fine = y_interpolator(t_fine)
z_fine = z_interpolator(t_fine)

x_dot_fine = x_dot_interpolator(t_fine)
y_dot_fine = y_dot_interpolator(t_fine)
z_dot_fine = z_dot_interpolator(t_fine)

"""
TRAJECTORY GENERATION
"""

time_constant = 1  # speed time constant
desired_speed = cfg["desired_speed"]  # desired speed waypoint 1

initial_speed = 0     # initial speed waypoint 0
th = 0.01    # initial th-value waypoint 0

h = cfg["dt"]  # sampling time
i = 0     # counter

x_th = []
y_th = []
z_th = []
th_list = []
speed_list = []

while th < len(x)-1:
    # cubic polynominal between waypoints 0 and 1
    x_th.append(x_dot_interpolator(th))
    y_th.append(y_dot_interpolator(th))
    z_th.append(z_dot_interpolator(th))

    th += h * (initial_speed / np.sqrt(x_th[i]**2 + y_th[i]**2 + z_th[i]**2))  # theta dynamics
    initial_speed += h * (-initial_speed + desired_speed) / time_constant  # speed dynamics
    
    speed_list.append(initial_speed)
    th_list.append(th)

    i += 1

"""
PLOTTING
"""

fig1, (ax1, ax2, ax3) = plt.subplots(3)
fig1.suptitle("PCHIP Interpolation")

ax1.plot(theta, x, "o", label="Original data points")
ax1.plot(t_fine, x_fine, "-", label="PCHIP interpolated curve")
ax1.set(ylabel="X")
ax1.grid(True)

ax2.plot(theta, y, "o", label="Original data points")
ax2.plot(t_fine, y_fine, "-", label="PCHIP interpolated curve")
ax2.set(ylabel="Y")
ax2.grid(True)

ax3.plot(theta, z, "o", label="Original data points")
ax3.plot(t_fine, z_fine, "-", label="PCHIP interpolated curve")
ax3.set(xlabel=r"${\Theta}$", ylabel="Z")
ax3.grid(True)
ax3.legend()

fig2, (ax1, ax2, ax3) = plt.subplots(3)
fig2.suptitle("PCHIP Interpolation Dot")

ax1.plot(t_fine, x_dot_fine, "-")
ax1.set(ylabel="X")
ax1.grid(True)

ax2.plot(t_fine, y_dot_fine, "-")
ax2.set(ylabel="Y")
ax2.grid(True)

ax3.plot(t_fine, z_dot_fine, "-")
ax3.set(xlabel=r"${\Theta}$", ylabel="Z")
ax3.grid(True)
ax3.legend()

time = np.arange(0, len(x_th)*h, h)

fig3, (ax1, ax2, ax3) = plt.subplots(3)
fig3.suptitle("PCHIP Interpolation Dot")

ax1.plot(time, x_interpolator(th_list))
ax1.set(ylabel="X")
ax1.grid(True)

ax2.plot(time, y_interpolator(th_list))
ax2.set(ylabel="Y")
ax2.grid(True)

ax3.plot(time, z_interpolator(th_list))
ax3.set(ylabel="Z")
ax3.grid(True)

fig4, (ax1, ax2) = plt.subplots(2)
fig4.suptitle("Speed U_d(t) and Path variable as a function of time t")

ax1.plot(time, np.ones(len(time)) * desired_speed)
ax1.plot(time, speed_list)
ax1.grid(True)

ax2.plot(time, np.ones(len(time)) * (len(x)-1))
ax2.plot(time, th_list)
ax2.grid(True)

plt.show()