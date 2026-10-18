import time
import collections
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Colour to track (HSV): bright green
lower = np.array([40, 100, 100])
upper = np.array([80, 255, 255])

alpha, beta = 0.5, 0.1
dt = 1.0
x = y = vx = vy = 0.0
started = False

FOV_X, FOV_Y = 60.0, 40.0
Kp = 0.3
PAN_SIGN, TILT_SIGN = 1, -1
pan, tilt = 90.0, 90.0

try:
    import serial
    ser = serial.Serial("COM3", 115200, timeout=0.01)
except Exception:
    ser = None

# Colours are BGR, not RGB
CYAN = (255, 255, 0)
AMBER = (0, 190, 255)
RED = (60, 60, 255)

trail = collections.deque(maxlen=30)   # remembers the last 30 positions
frame_count = 0
prev_time = time.time()


def label(img, text, org, color=CYAN, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def corner_brackets(img, color, size=30, margin=15):
    h, w = img.shape[:2]
    corners = [(margin, margin, 1, 1), (w - margin, margin, -1, 1),
               (margin, h - margin, 1, -1), (w - margin, h - margin, -1, -1)]
    for px, py, dx, dy in corners:
        cv2.line(img, (px, py), (px + dx * size, py), color, 2)
        cv2.line(img, (px, py), (px, py + dy * size), color, 2)


def reticle(img, centre, spin, color):
    cx, cy = int(centre[0]), int(centre[1])
    cv2.circle(img, (cx, cy), 18, color, 1, cv2.LINE_AA)
    for start in (0, 90, 180, 270):
        cv2.ellipse(img, (cx, cy), (32, 32), 0, start + spin, start + spin + 55,
                    color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, -1)


while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Detect the coloured blob
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

    # 2. The filter
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
        else:
            x, y = pred_x, pred_y

    # 3. Camera commands
    if started:
        err_x = x - cx
        err_y = y - cy
        cmd_pan = min(180, max(0, pan + PAN_SIGN * Kp * (err_x / w * FOV_X)))
        cmd_tilt = min(180, max(0, tilt + TILT_SIGN * Kp * (err_y / h * FOV_Y)))
        if ser:
            ser.write(f"{int(cmd_pan)},{int(cmd_tilt)}\n".encode())
            pan, tilt = cmd_pan, cmd_tilt

    # 4. HUD: darken, add scanlines and a blue tint, then draw on top
    frame[::4] = (frame[::4] * 0.75).astype(np.uint8)
    tint = np.full_like(frame, (50, 25, 0))