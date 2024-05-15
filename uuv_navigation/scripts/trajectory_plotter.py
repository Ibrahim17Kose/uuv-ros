#!/usr/bin/env python3

import rospkg
import numpy as np
from yaml import safe_load
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

"""
READING WAYPOINTS
"""

with open(rospkg.RosPack().get_path("uuv_navigation") + f"/config/demo_waypoints.yaml", 'r') as file:
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

"""
PATH GENERATION
"""

# Generate a series of 100 x-values from 0 to 9 for a smooth plot
t_fine = np.linspace(0, len(x)-1, 100)

# Compute the interpolated y-values
x_fine = x_interpolator(t_fine)
y_fine = y_interpolator(t_fine)
z_fine = z_interpolator(t_fine)

"""
TRAJECTORY GENERATION
"""

time_constant = 1  # speed time constant
desired_speed = cfg["desired_speed"]  # desired speed waypoint 1

initial_speed = 0     # initial speed waypoint 0
th = 0    # initial th-value waypoint 0

h = cfg["dt"]  # sampling time
i = 0     # counter

x_th = []
y_th = []
z_th = []
print(len(x))
while th < len(x)-1:
    # cubic polynominal between waypoints 0 and 1
    x_th.append(x_interpolator(th))
    y_th.append(y_interpolator(th))
    z_th.append(z_interpolator(th))

    th += h * (initial_speed / np.sqrt(x_th[i]**2 + y_th[i]**2 + z_th[i]**2))  # theta dynamics
    initial_speed += h * (-initial_speed + desired_speed) / time_constant  # speed dynamics

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

time = np.arange(0, len(x_th)*h, h)

fig2, (ax1, ax2, ax3) = plt.subplots(3)
fig2.suptitle("Path Time Interpolation")

ax1.plot(time, x_th)
ax1.set(ylabel="X")
ax1.grid(True)

ax2.plot(time, y_th)
ax2.set(ylabel="Y")
ax2.grid(True)

ax3.plot(time, z_th)
ax3.set(xlabel="Time", ylabel="Z")
ax3.grid(True)

plt.show()