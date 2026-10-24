"""Robust virtual button input using the calibrated, stationary board.

A single MediaPipe dropout cannot erase a nearly completed finger press.
The finger must leave all buttons before another press is accepted.
"""
import cv2
from config import TOUCH_HOLD_SECONDS, TOUCH_RELEASE_SECONDS

# Deliberately small visual tolerance for rough colored-paper contours.
EDGE_TOLERANCE_PX = 3.0
# Pause (do not reset) the hold for a brief lost detection.
TRACKING_GRACE_SECONDS = 0.20
RELEASE_SECONDS = max(TOUCH_RELEASE_SECONDS, 0.22)
LOST_HAND_RELEASE_SECONDS = 0.50  # avoid rearming on brief MediaPipe misses
MAX_FRAME_CREDIT_SECONDS = 0.12


class TouchDetector:
    def __init__(self):
        self.armed = False
        self.was_enabled = False
        self.candidate = None
        self.hold_seconds = 0.0
        self.outside_since = None
        self.last_time = None
        self.progress = 0.0
        self.status = "WAIT FOR YOUR TURN"

    @staticmethod
    def locate(point, regions):
        """Choose a single stable panel ID. Board contours were locked with no hand."""
        if point is None:
            return None
        p = (float(point[0]), float(point[1]))
        best_name, best_distance = None, float("-inf")
        for name, region in regions.items():
            # The outer hull resists imperfect segmentation around paper edges.
            outline = cv2.convexHull(region["contour"])
            distance = cv2.pointPolygonTest(outline, p, True)
            if distance >= -EDGE_TOLERANCE_PX and distance > best_distance:
                best_name, best_distance = name, distance
        return best_name

    def update(self, point, regions, now, enabled):
        if not enabled:
            self.__init__()
            return None, None

        if not self.was_enabled:
            self.was_enabled = True
            self.armed = True  # first selection needs no initial leave gesture
            self.last_time = now
            self.status = "READY - POINT AT A COLOR"

        dt = max(0.0, min(now - self.last_time, MAX_FRAME_CREDIT_SECONDS))
        self.last_time = now
        zone = self.locate(point, regions)

        if zone is None:
            if self.outside_since is None:
                self.outside_since = now
            gap = now - self.outside_since
            if not self.armed:
                self.status = "LIFT FINGER - REARMING"
                release_needed = (LOST_HAND_RELEASE_SECONDS if point is None else RELEASE_SECONDS)
                if gap >= release_needed:
                    self.armed = True
                    self.status = "READY - NEXT COLOR"
            elif self.candidate is not None:
                if gap <= TRACKING_GRACE_SECONDS:
                    # Keep progress, but do not count missing frames as a press.
                    self.status = "BRIEF TRACKING GAP"
                else:
                    self.candidate = None
                    self.hold_seconds = 0.0
                    self.progress = 0.0
                    self.status = "NO PANEL - TRY AGAIN"
            else:
                self.status = "READY - POINT AT A COLOR" if self.armed else self.status
            return None, None

        was_outside = self.outside_since is not None
        self.outside_since = None
        if not self.armed:
            self.progress = 0.0
            self.status = "LIFT FINGER OFF ALL PANELS"
            return None, zone

        if zone != self.candidate:
            self.candidate = zone
            self.hold_seconds = 0.0
            self.progress = 0.0
            self.status = f"HOLD {zone}"
            return None, zone

        # Do not credit an interval in which tracking was missing.
        if not was_outside:
            self.hold_seconds += dt
        self.progress = min(1.0, self.hold_seconds / TOUCH_HOLD_SECONDS)
        self.status = f"HOLD {zone}"
        if self.progress >= 1.0:
            self.armed = False
            self.candidate = None
            self.hold_seconds = 0.0
            self.progress = 0.0
            self.status = f"CONFIRMED {zone} - LIFT FINGER"
            return zone, zone
        return None, zone
