import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dt=1.0                  #seconds between radar measurements
steps=100

# Radar sits at the origin (0,0)
# True target (aircraft): starts 5km away, flies west, then turns
x, y = 5000.0, 5000.0     #metres
speed = 200.0             #m/s
heading=np.radians(180)

true_path=[]
for k in range(steps):
    if 40 <= k < 60:
        heading += np.radians(2)    #gentle turn
    x += speed * np.cos(heading) * dt
    y += speed * np.sin(heading) * dt
    true_path.append((x,y))
true_path = np.array(true_path)

#Radar measures range and bearing, with noise
range_noise = 50.0                  #metres
bearing_noise = np.radians(0.5)     #radians

true_range = np.hypot(true_path[:,0], true_path[:,1])
true_bearing = np.arctan2(true_path[:,1], true_path[:,0])

meas_range = true_range + np.random.normal(0, range_noise, steps)
meas_bearing = true_bearing + np.random.normal(0, bearing_noise, steps)

#Converting measurements back to x/y

meas_x = meas_range * np.cos(meas_bearing)
meas_y = meas_range * np.sin(meas_bearing)

plt.plot(true_path[:,0], true_path[:,1], label="True Path")
plt.scatter(meas_x, meas_y, s=10, color="red", label="Radar Measurements")
plt.scatter(0, 0, marker="^", color="black", label="Radar")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()
                          