import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower = np.array([40, 100, 100])
upper = np.array([80, 255, 255])

alpha, beta = 0.5, 0.1
dt = 1.0                    # one video frame per step
x = y = vx = vy = 0.0
started = False

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Find the colour
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = False
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 300:
            (mx, my), _ = cv2.minEnclosingCircle(biggest)
            found = True

    if found and not started:
        x, y = mx, my
        started = True

    # The same filter as tracker.py
    if started:
        pred_x = x + vx * dt
        pred_y = y + vy * dt
        if found:
            surprise_x = mx - pred_x
            surprise_y = my - pred_y
            x = pred_x + alpha * surprise_x
            y = pred_y + alpha * surprise_y
            vx = vx + beta * surprise_x / dt
            vy = vy + beta * surprise_y / dt
            cv2.circle(frame, (int(mx), int(my)), 8, (0, 0, 255), 2)   # raw detection: red ring
        else:
            x, y = pred_x, pred_y                                       # no measurement: just predict
        cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)         # filtered estimate: green dot

    cv2.imshow("Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

