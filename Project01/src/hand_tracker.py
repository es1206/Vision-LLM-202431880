"""AI Color Memory Game: Hand landmark detection and tracking."""
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FINGER_TIPS = (4, 8, 12, 16, 20)
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


class HandTracker:
    def __init__(self, model_path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {path}. Copy hand_landmarker.task into Project01/models/."
            )
        base_option = python.BaseOptions(model_asset_buffer=path.read_bytes())
        options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame, draw_on):
        # The same RGB conversion / mp.Image / detector.detect sequence as the lesson.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = __import__('numpy').ascontiguousarray(rgb)
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        fingertip = None
        for hand in result.hand_landmarks:
            h, w = frame.shape[:2]
            points = [(int(p.x * w), int(p.y * h)) for p in hand]
            for edge in CONNECTIONS:
                cv2.line(draw_on, points[edge.start], points[edge.end], (0, 230, 0), 2)
            for i, point in enumerate(points):
                color = (0, 0, 255) if i in FINGER_TIPS else (255, 0, 0)
                cv2.circle(draw_on, point, 6 if i in FINGER_TIPS else 4, color, -1)
            fingertip = points[8]  # Index finger tip: Landmark #8
            cv2.circle(draw_on, fingertip, 11, (255, 255, 255), 2)
            break
        return fingertip, len(result.hand_landmarks)

    def close(self):
        self.detector.close()
