'''
Filename: finaldraft.ipynb
Written by: Cricket Bergner
Date: 04/27/26
'''

# import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp as intg
import random as ra
from matplotlib.colors import ListedColormap # for the fractal mapping

# functions

def acceleration(x, y, vx, vy, magnets):
    ax = -restoring * x - damping * vx
    ay = -restoring * y - damping * vy

    for mag in magnets:
        dx = mag[0] - x
        dy = mag[1] - y
        r = np.sqrt(dx**2 + dy**2 + vertical_offset**2)
        ax += magnetic_strength * dx / r**3
        ay += magnetic_strength * dy / r**3

    return ax, ay

def equations(t, state, magnets):
    x, y, vx, vy = state
    ax, ay = acceleration(x, y, vx, vy, magnets)
    return [vx, vy, ax, ay]

def rk45(initial_state, magnets):
    solution = intg(
        fun = lambda t, y: equations(t, y, magnets),
        t_span=t_span,
        y0=initial_state,
        method='RK45',
        rtol=1e-5,
        atol=1e-7,
        events=lambda t, y: stop_event(t, y, magnets)
    )
    return solution

# Computes the distance from a given point to each magnet. Determines which magnet is closest.
# Index of closest magnet is returned to classify a basin of attraction.
def find_closest_magnet(x, y, magnets):
    distances = [np.sqrt((x - m[0])**2 + (y - m[1])**2) for m in magnets]
    return np.argmin(distances)

# Creates grid of inital conditions. Runs through each point, and figures out which magnet the trajectory
# leads to. Result is stored as a 2D array.
def generate_fast_fractal(magnets, res=400, max_iter=400):

    # vectorized grid instead of nested loops
    x = np.linspace(-1.5, 1.5, res)
    y = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(x, y)

    position = np.stack([X, Y], axis=-1)
    vel = np.zeros_like(position)
    dt = 0.05
    settled_at = np.zeros((res, res))

    for i in range(max_iter):
        accel = -restoring * position - damping * vel

        for m in magnets:
            diff = m - position
            r = np.sqrt(np.sum(diff**2, axis=-1) + vertical_offset**2)
            accel += magnetic_strength * diff / r[..., np.newaxis]**3

        # fast integration (Euler instead of RK45)
        vel += accel * dt
        position += vel * dt
        speed = np.sqrt(np.sum(vel**2, axis=-1))
        settled_at[speed < 0.02] = i

    # Determine basins
    distances = []
    for m in magnets:
        dist = np.sqrt(np.sum((position - m)**2, axis=-1))
        distances.append(dist)

    basins = np.argmin(distances, axis=0)

    return basins, settled_at

# stop event
def stop_event(t, state, magnets):
    vx = state[2]
    vy = state[3]
    speed = np.sqrt(vx**2 + vy**2)
    return speed - 0.01

stop_event.terminal = True
stop_event.direction = -1

# initial conditions

damping = 0.2
restoring = 0.6
vertical_offset = 0.3
magnetic_strength = 1.2
t_span = (0,10)

initial_conditions, solutions = [], []

# base code: 3 magnets

# magnets
magnets3 = [np.array([1, 0]),
          np.array([-0.5, np.sqrt(3)/2]),
          np.array([-0.5, -np.sqrt(3)/2])]

# randomly generate initial conditions
for _ in range(3):
    x0 = ra.uniform(-1.5, 1.5)
    y0 = ra.uniform(-1.5, 1.5)
    vx0 = ra.uniform(-0.1, 0.1)
    vy0 = ra.uniform(-0.1, 0.1)
    initial_conditions.append([x0, y0, vx0, vy0])

for i in initial_conditions:
    ans = rk45(i, magnets3)
    solutions.append(ans)

# plot motion
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, j in enumerate(solutions):
    ax = axes[i]
    x = j.y[0]
    y = j.y[1]

    ax.plot(x, y, lw=0.7)

    for m in magnets3:
        ax.scatter(m[0], m[1], color='red', s=50, zorder=3)

    ic = initial_conditions[i]
    ax.set_title(
        f"x0={ic[0]:.2f}, y0={ic[1]:.2f}\n"
        f"vx0={ic[2]:.2f}, vy0={ic[3]:.2f}"
    )

    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

plt.suptitle("Magnetic Pendulum Trajectories (3 Magnets)")
plt.show()

# generate the fractal
print("\nGenerating fractal...\n")


basins, shading = generate_fast_fractal(magnets3, res=500)
colors = ['#582c83', '#FFD100', '#76d7f5'] # I picked these. Wings up!
custom_cmap = ListedColormap(colors)
plt.figure(figsize=(10, 10))

# base colors
plt.imshow(basins, cmap=custom_cmap,
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower')

# shading layer for fractal texture
plt.imshow(shading, cmap='bone',
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower', alpha=0.3)

plt.axis('off')
plt.title("Magnetic Pendulum Fractal (3 Magnets)", color='black')
plt.gcf().set_facecolor('white')

# plot magnets
for m in magnets3:
    plt.scatter(m[0], m[1], color='white', s=60)
print("")
plt.show()


# extension one (4 magnets)

# place four magnets on the ends of a square
magnets4 = [np.array([0.5, 0.5]),
            np.array([0.5, -0.5]),
            np.array([-0.5, -0.5]),
            np.array([-0.5, 0.5])]

# reset initial lists and change a few variables
restoring = 0.1  
magnetic_strength = 2.0
initial_conditions, solutions = [], []


# randomly generate initial conditions
for _ in range(4):
    x0 = ra.uniform(-1.5, 1.5)
    y0 = ra.uniform(-1.5, 1.5)
    vx0 = ra.uniform(-0.1, 0.1)
    vy0 = ra.uniform(-0.1, 0.1)
    initial_conditions.append([x0, y0, vx0, vy0])

for i in initial_conditions:
    ans = rk45(i, magnets4)
    solutions.append(ans)

# plot motion
fig, axes = plt.subplots(1, len(solutions), figsize=(5*len(solutions), 5))
for i, j in enumerate(solutions):
    ax = axes[i]
    x = j.y[0]
    y = j.y[1]

    ax.plot(x, y, lw=0.7)

    for m in magnets4:
        ax.scatter(m[0], m[1], color='red', s=50, zorder=4)

    ic = initial_conditions[i]
    ax.set_title(
        f"x0={ic[0]:.2f}, y0={ic[1]:.2f}\n"
        f"vx0={ic[2]:.2f}, vy0={ic[3]:.2f}"
    )

    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

plt.suptitle("Magnetic Pendulum Trajectories (4 Magnets)")
print("")
plt.show()

# generate the fractal
print("\nGenerating fractal...\n")


basins, shading = generate_fast_fractal(magnets4, res=500)
colors = ['#FF512B', '#CDFF42', '#FC3DB9', '#1EF7BA'] # colors picked by my friend Nolan! 
custom_cmap = ListedColormap(colors)
plt.figure(figsize=(10, 10))

# base colors
plt.imshow(basins, cmap=custom_cmap,
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower')

# shading layer for fractal texture
plt.imshow(shading, cmap='bone',
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower', alpha=0.3)

plt.axis('off')
plt.title("Magnetic Pendulum Fractal (4 Magnets)", color='black')
plt.gcf().set_facecolor('white')

# plot magnets
for m in magnets4:
    plt.scatter(m[0], m[1], color='white', s=60)
print("")
plt.show()

# extension two (5 magnets) ############################################################################

N = 5    # number of sides
R = 1.0  # radius (distance from origin)

magnets5 = [
    np.array([
        R * np.cos(2*np.pi*k/N),
        R * np.sin(2*np.pi*k/N)
    ])
    for k in range(N)
]

# reset initial lists and change a few variables # MAY CHANGE LATER!!!!!!!!!!!!!
restoring = 0.1  
magnetic_strength = 2.0
initial_conditions, solutions = [], []


# randomly generate initial conditions
for _ in range(5):
    x0 = ra.uniform(-1.5, 1.5)
    y0 = ra.uniform(-1.5, 1.5)
    vx0 = ra.uniform(-0.1, 0.1)
    vy0 = ra.uniform(-0.1, 0.1)
    initial_conditions.append([x0, y0, vx0, vy0])

for i in initial_conditions:
    ans = rk45(i, magnets5)
    solutions.append(ans)

# plot motion
fig, axes = plt.subplots(1, len(solutions), figsize=(5*len(solutions), 5))
for i, j in enumerate(solutions):
    ax = axes[i]
    x = j.y[0]
    y = j.y[1]
    ax.plot(x, y, lw=0.7)

    for m in magnets5:
        ax.scatter(m[0], m[1], color='red', s=50, zorder=4)

    ic = initial_conditions[i]
    ax.set_title(
        f"x0={ic[0]:.2f}, y0={ic[1]:.2f}\n"
        f"vx0={ic[2]:.2f}, vy0={ic[3]:.2f}"
    )

    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

plt.suptitle("Magnetic Pendulum Trajectories (5 Magnets)")
print("")
plt.show()

# generate the fractal
print("\nGenerating fractal...\n")


basins, shading = generate_fast_fractal(magnets5, res=500)
colors = ['#0E7526', '#5C2807', '#5D7285', '#DB730B', '#00B3FF'] # colors picked by Eli! 
custom_cmap = ListedColormap(colors)
plt.figure(figsize=(10, 10))

# base colors
plt.imshow(basins, cmap=custom_cmap,
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower')

# shading layer for fractal texture
plt.imshow(shading, cmap='bone',
           extent=[-1.5, 1.5, -1.5, 1.5],
           origin='lower', alpha=0.3)

plt.axis('off')
plt.title("Magnetic Pendulum Fractal (5 Magnets)", color='black')
plt.gcf().set_facecolor('white')

# plot magnets
for m in magnets5:
    plt.scatter(m[0], m[1], color='white', s=60)
plt.show()


# extension 3 (3 magnets, moving magnets)

#################################################################
# END FINAL PROJECT
#################################################################
