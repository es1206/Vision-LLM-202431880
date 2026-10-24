"""AI Color Memory Game: Color board position tracking."""
from config import COLOR_CACHE_SECONDS


class ColorTracker:
    def __init__(self):
        self.previous = {}

    def update(self, detections, now):
        for name, item in detections.items():
            old = self.previous.get(name)
            if old and now - old["seen"] < COLOR_CACHE_SECONDS:
                # When a hand partially covers a panel, retain its unoccluded shape.
                if item["area"] < old["area"] * 0.65:
                    continue
            self.previous[name] = {**item, "seen": now}
        self.previous = {
            name: item for name, item in self.previous.items()
            if now - item["seen"] <= COLOR_CACHE_SECONDS
        }
        return self.previous
