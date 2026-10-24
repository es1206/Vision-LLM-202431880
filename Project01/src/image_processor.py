"""Lecture: image adjustment, LAB-CLAHE, filtering, BGR to HSV."""
import cv2

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(frame):
    adjusted = cv2.convertScaleAbs(frame, alpha=1.03, beta=3)
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    improved = cv2.merge((CLAHE.apply(light), a, b))
    enhanced = cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv
