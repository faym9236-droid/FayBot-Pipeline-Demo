import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge, FancyArrowPatch
from matplotlib import animation

PIPE_LENGTH = 100.0
PIPE_HEIGHT = 10.0
N_DEFECTS = 8
ROBOT_SPEED = 12.0
DT = 0.05
SIM_TIME = 10.0
SENSOR_RANGE = 10.0
SENSOR_RADIUS = 3.0
SPRAY_RANGE = 6.0
FOV_DEG = 40
np.random.seed(2)

defects_x = np.sort(np.random.uniform(15, PIPE_LENGTH - 10, N_DEFECTS))
defects_y = np.random.uniform(2.0, PIPE_HEIGHT - 2.0, N_DEFECTS)
defects_r = np.random.uniform(0.6, 1.0, N_DEFECTS)
defects_state = np.array(['active'] * N_DEFECTS)

robot_x = 0.0
robot_y = PIPE_HEIGHT / 2.0

fig, ax = plt.subplots(figsize=(12, 2.4))
ax.set_xlim(0, PIPE_LENGTH)
ax.set_ylim(0, PIPE_HEIGHT)
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("In-Pipe Robot: Camera+Sensors + Antibacterial Cold Spray")

pipe_rect = Rectangle((0, 0), PIPE_LENGTH, PIPE_HEIGHT, fill=False, linewidth=2)
ax.add_patch(pipe_rect)

defect_patches = []
for xi, yi, ri in zip(defects_x, defects_y, defects_r):
    p = Circle((xi, yi), ri, color='red', alpha=0.85)
    ax.add_patch(p)
    defect_patches.append(p)

robot_body = Circle((robot_x, robot_y), 1.0, fc='black', ec='black', alpha=0.9)
cam_arrow = FancyArrowPatch(posA=(robot_x, robot_y),
                            posB=(robot_x+1.5, robot_y),
                            arrowstyle='-|>', mutation_scale=10, linewidth=1)
fov = Wedge(center=(robot_x, robot_y), r=4.0, theta1=-FOV_DEG/2, theta2=FOV_DEG/2, alpha=0.15)
spray = Wedge(center=(robot_x+0.8, robot_y), r=SPRAY_RANGE, theta1=-15, theta2=15, alpha=0.25)

ax.add_patch(robot_body)
ax.add_patch(fov)
ax.add_patch(spray)
ax.add_patch(cam_arrow)

status_text = ax.text(1, PIPE_HEIGHT-1.2, "", fontsize=9)

events = []

def detect_and_spray(rx, ry, time_now):
    global defects_state
    ahead = (defects_x >= rx) & (defects_x <= rx + SENSOR_RANGE)
    lateral = (np.abs(defects_y - ry) <= SENSOR_RADIUS)
    candidates = np.where(ahead & lateral & (defects_state == 'active'))[0]
    sprayed = False
    if candidates.size > 0:
        idx = candidates[np.argmin(defects_x[candidates] - rx)]
        if defects_x[idx] - rx <= SPRAY_RANGE:
            defects_state[idx] = 'coated'
            events.append((time_now, int(idx), float(defects_x[idx]), float(defects_y[idx])))
            sprayed = True
    return sprayed

def init():
    robot_body.center = (0.0, robot_y)
    cam_arrow.set_positions((0.0, robot_y), (1.5, robot_y))
    fov.center = (0.0, robot_y)
    spray.set_center((0.8, robot_y))
    spray.set_alpha(0.0)
    status_text.set_text("")
    for i, p in enumerate(defect_patches):
        p.set_color('red')
        p.set_alpha(0.85)
    return [robot_body, cam_arrow, fov, spray, status_text, *defect_patches]

def animate(frame):
    global robot_x
    time_now = frame * DT
    robot_x = min(PIPE_LENGTH, ROBOT_SPEED * time_now)
    robot_body.center = (robot_x, robot_y)
    cam_arrow.set_positions((robot_x, robot_y), (robot_x+1.5, robot_y))
    fov.center = (robot_x, robot_y)
    spray.set_center((robot_x+0.8, robot_y))
    spray.set_alpha(0.0)
    sprayed = detect_and_spray(robot_x, robot_y, time_now)
    for i, (xi, yi, ri) in enumerate(zip(defects_x, defects_y, defects_r)):
        if defects_state[i] == 'coated':
            defect_patches[i].set_color('gray')
            defect_patches[i].set_alpha(0.8)
    if sprayed:
        spray.set_alpha(0.35)
        status_text.set_text(f"t={time_now:4.1f}s  Spray ON")
    else:
        status_text.set_text(f"t={time_now:4.1f}s  Scanning…")
    return [robot_body, cam_arrow, fov, spray, status_text, *defect_patches]

frames = int(SIM_TIME / DT)
ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=DT*1000, blit=True)
ani.save("pipeline_robot_coldspray_sim.gif", writer=animation.PillowWriter(fps=int(1/DT)))
print("Simulation complete. GIF saved as pipeline_robot_coldspray_sim.gif")
