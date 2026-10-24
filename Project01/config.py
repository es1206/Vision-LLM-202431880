"""AI Color Memory Game: Game settings and constants."""

DEFAULT_CAMERA_MODE = "jetson_usb"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
LAPTOP_CAMERA_INDEX = 0
JETSON_USB_DEVICE = "/dev/video0"
JETSON_USB_FORMAT = "YUY2"  # switch to MJPG only if supported by the webcam
JETSON_CSI_SENSOR_ID = 0
FLIP_CODE = None  # None, 0=vertical, 1=horizontal, -1=both

# Camera BGR color thresholds, in OpenCV HSV units: H=0..179, S/V=0..255.
HSV_RANGES = {
    "RED": [((0, 85, 65), (12, 255, 255)), ((170, 85, 65), (179, 255, 255))],
    "BLUE": [((92, 75, 55), (132, 255, 255))],
    "GREEN": [((36, 65, 55), (88, 255, 255))],
    "YELLOW": [((17, 80, 75), (37, 255, 255))],
}
COLORS = tuple(HSV_RANGES.keys())  # RED, BLUE, GREEN, YELLOW

MIN_COLOR_AREA = 1600  # reduce if the game board looks small in the camera
COLOR_CACHE_SECONDS = 1.2  # survive a brief hand occlusion
TOUCH_HOLD_SECONDS = 0.35
TOUCH_RELEASE_SECONDS = 0.15
TOUCH_INTERIOR_MARGIN = 5.0
# Number of inputs and points awarded for clearing each round.
ROUND_LENGTHS = (3, 5, 8)
ROUND_SCORES = (20, 30, 50)
MAX_ROUND = len(ROUND_LENGTHS)
SHOW_ON_SECONDS = 0.85
SHOW_OFF_SECONDS = 0.25
ROUND_PAUSE_SECONDS = 1.5
COLOR_BGR = {
    "RED": (55, 65, 230),
    "BLUE": (220, 125, 35),
    "GREEN": (70, 180, 70),
    "YELLOW": (40, 215, 240),
}
