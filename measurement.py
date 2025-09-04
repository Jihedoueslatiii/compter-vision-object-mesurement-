# measurement.py
import cv2
import numpy as np

KNOWN_WIDTH_CM = 8.5
MIN_CONTOUR_AREA = 1500

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def process_frame(frame, calibrated, pixels_per_cm):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    width_cm, height_cm = None, None

    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = order_points(box)

        (tl, tr, br, bl) = box
        mid_top = midpoint(tl, tr)
        mid_bottom = midpoint(bl, br)
        mid_left = midpoint(tl, bl)
        mid_right = midpoint(tr, br)

        width_px = np.linalg.norm(np.array(mid_top) - np.array(mid_bottom))
        height_px = np.linalg.norm(np.array(mid_left) - np.array(mid_right))

        if not calibrated:
            pixels_per_cm = width_px / KNOWN_WIDTH_CM
            calibrated = True

        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm

        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"W: {width_cm:.2f} cm", (int(tl[0]), int(tl[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"H: {height_cm:.2f} cm", (int(tl[0]), int(tl[1]) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        break  # largest contour only

    return frame, calibrated, pixels_per_cm, width_cm, height_cm
