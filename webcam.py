import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower = np.array([40, 100, 100])
upper = np.array([80, 255, 255])

alpha, beta = 0.1, 0.1
dt = 1.0                    # one video frame per step
x = y = vx = vy = 0.0
started = False

FOV_X, FOV_Y = 60.0, 40.0     # camera's field of view in degrees (approximate, adjust later)
Kp = 0.3
PAN_SIGN, TILT_SIGN = 1, -1   # flip either one if the camera moves the wrong way
pan, tilt = 90.0, 90.0        # servo angles, 90 = centred

try:
    import serial
    ser = serial.Serial("COM3", 115200, timeout=0.01)   # change COM3 to your port
except Exception:
    ser = None                #until hardware
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

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        if started:
            err_x = x - cx
            err_y = y - cy
            err_deg_x = err_x / w * FOV_X          # pixels to degrees
            err_deg_y = err_y / h * FOV_Y
            cmd_pan = min(180, max(0, pan + PAN_SIGN * Kp * err_deg_x))
            cmd_tilt = min(180, max(0, tilt + TILT_SIGN * Kp * err_deg_y))
            if ser:
                ser.write(f"{int(cmd_pan)},{int(cmd_tilt)}\n".encode())
                pan, tilt = cmd_pan, cmd_tilt
            cv2.line(frame, (cx, cy), (int(x), int(y)), (255, 255, 0), 1)
            cv2.putText(frame, f"cmd pan {cmd_pan:.0f} tilt {cmd_tilt:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()

