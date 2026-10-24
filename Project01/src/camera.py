"""AI Color Memory Game: Webcam initialization and frame capture."""
from pathlib import Path
import cv2
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, JETSON_CSI_SENSOR_ID
)


def open_camera(mode, source="0", device="/dev/video0", usb_format="YUY2"):
    if mode == "laptop":
        idx = int(source)
        # DSHOW is appropriate on Windows; fall back to OpenCV default elsewhere.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if __import__('sys').platform == 'win32' else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    elif mode == "jetson_usb":
        if usb_format == "MJPG":
            start = (f"v4l2src device={device} ! image/jpeg,width={CAMERA_WIDTH},"
                     f"height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! jpegdec ! ")
        else:
            start = (f"v4l2src device={device} ! video/x-raw,format=YUY2,"
                     f"width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! ")
        pipeline = (start + "videoconvert ! video/x-raw,format=BGR ! "
                    "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 "
                    "max-size-time=0 ! appsink drop=true max-buffers=1 sync=false")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif mode == "jetson_csi":
        pipeline = (
            f"nvarguscamerasrc sensor-id={JETSON_CSI_SENSOR_ID} ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! queue leaky=downstream max-size-buffers=1 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif mode == "video":
        p = Path(source)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        cap = cv2.VideoCapture(str(p))
    else:
        raise ValueError("Unsupported camera mode: " + mode)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {mode}. Check webcam index/device/format and GStreamer support.")
    return cap
