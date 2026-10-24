"""Camera acquisition. Jetson USB path reproduces the classroom-tested GStreamer pipeline."""
from pathlib import Path
import sys
import cv2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, JETSON_CSI_SENSOR_ID


def get_jetson_usb_pipeline(device="/dev/video0", usb_format="YUY2"):
    """Classroom verified: v4l2src + YUY2 + 640x480/30 + BGR appsink.

    MJPG is an optional alternative only after confirming camera capabilities.
    """
    if usb_format == "YUY2":
        source = (
            f"v4l2src device={device} ! "
            f"video/x-raw,format=YUY2,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
            f"framerate={CAMERA_FPS}/1 ! "
        )
    elif usb_format == "MJPG":
        source = (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
            f"framerate={CAMERA_FPS}/1 ! jpegdec ! "
        )
    else:
        raise ValueError("Unsupported USB format: " + str(usb_format))
    return (
        source
        + "videoconvert ! video/x-raw,format=BGR ! "
        + "queue leaky=downstream max-size-buffers=1 "
        + "max-size-bytes=0 max-size-time=0 ! "
        + "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(mode, source="0", device="/dev/video0", usb_format="YUY2"):
    if mode == "laptop":
        idx = int(source)
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    elif mode == "jetson_usb":
        pipeline = get_jetson_usb_pipeline(device, usb_format)
        print("[CAMERA] Jetson USB GStreamer:", pipeline, flush=True)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif mode == "jetson_csi":
        # Kept for optional CSI camera. The final classroom camera was USB, not CSI.
        pipeline = (
            f"nvarguscamerasrc sensor-id={JETSON_CSI_SENSOR_ID} ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! queue leaky=downstream max-size-buffers=1 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif mode == "video":
        path = Path(source)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        cap = cv2.VideoCapture(str(path))
    else:
        raise ValueError("Unsupported camera mode: " + str(mode))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open {mode}. Check /dev/video0, USB YUY2 640x480 30fps, "
            "and OpenCV GStreamer support."
        )
    return cap
