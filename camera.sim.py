import matplotlib.pyplot as plt

Kp = 0.3
est_vel = 0.5       # pretend filter velocity estimate, in deg/step
pan = 0.0
pans = [pan]
targets = []
for k in range(30):
    target = 30.0 + 0.5 * k
    targets.append(target)
    error = target - pan
    pan = pan + Kp * error + est_vel    # react to the error, plus move with the target
    pans.append(pan)

plt.plot(pans, label="camera angle")
plt.plot(targets, color="red", label="target angle")
plt.xlabel("step")
plt.ylabel("angle (deg)")
plt.legend()
plt.show()
print(targets[-1] - pans[-1])