"""AI Color Memory Game: Color segmentation and detection."""
import cv2
import numpy as np
from config import HSV_RANGES, MIN_COLOR_AREA

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def detect_colors(hsv):
    found = {}
    masks = {}
    for name, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(
                hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)
            ))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
        masks[name] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < MIN_COLOR_AREA:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        found[name] = {"contour": contour, "area": area, "center": center}
    return found, masks
