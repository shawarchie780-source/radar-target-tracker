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

alpha = 0.25
beta = alpha**2 / (2 - alpha)

R = range_noise**2       # radar noise, in squared metres
Q = 100000.0           # how much extra doubt we add each step
P = R              # start out unsure

est_x = [meas_x[0]]
est_vx = [-200.0]
for k in range(1, steps):
    predicted = est_x[-1] + est_vx[-1] * dt
    P = P + Q
    K = P/(P+R)
    surprise = meas_x[k] - predicted
    new_x = predicted + K * surprise
    new_vx = est_vx[-1] + (K**2 / (2 - K)) * surprise / dt
    P = (1-K)*P
    est_x.append(new_x)
    est_vx.append(new_vx)

print("final K", K)

Py = R
est_y = [meas_y[0]]
est_vy = [0.0]
for k in range(1, steps):
    predicted = est_y[-1] + est_vy[-1] * dt
    Py = Py + Q
    Ky = Py/(Py+R)
    surprise = meas_y[k] - predicted
    new_y = predicted + Ky * surprise
    new_vy = est_vy[-1] + (Ky**2 / (2 - Ky)) * surprise / dt
    Py = (1-Ky)*Py
    est_y.append(new_y)
    est_vy.append(new_vy)

err = np.sqrt((np.array(est_x) - true_path[:, 0])**2 + (np.array(est_y) - true_path[:, 1])**2)
print("alpha", alpha, "beta", beta, "mean error", np.mean(err))

raw_err = np.sqrt((meas_x - true_path[:, 0])**2 + (meas_y - true_path[:, 1])**2)
print("raw measurement error", np.mean(raw_err))

plt.plot(true_path[:,0], true_path[:,1], label="True Path")
plt.scatter(meas_x, meas_y, s=10, color="red", label="Radar Measurements")
plt.scatter(0, 0, marker="^", color="black", label="Radar")
plt.plot(est_x, est_y, color="green", label="Radar")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()

plt.figure()
plt.plot(true_path[:,0], label="True x")
plt.plot(meas_x, ".", label="Measured x")
plt.plot(est_x, label="Filtered x")
plt.xlabel("step")
plt.ylabel("x (m)")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, steps), np.diff(true_path[:, 0]) / dt, label="True vx")
plt.plot(est_vx, label="Estimated vx")
plt.xlabel("step")
plt.ylabel("x velocity (m/s)")
plt.legend()
plt.show()

                          