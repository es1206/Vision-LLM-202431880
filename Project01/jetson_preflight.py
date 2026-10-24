"""Quick camera test: run BEFORE the full game, no MediaPipe dependency.

Examples:
  python jetson_preflight.py --show
  python jetson_preflight.py --show --flip 0
"""
import argparse
from pathlib import Path
import cv2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, JETSON_USB_DEVICE
from src.camera import open_camera, get_jetson_usb_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=JETSON_USB_DEVICE)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--flip", choices=("none", "0", "1", "-1"), default="none")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    line = [s.strip() for s in cv2.getBuildInformation().splitlines() if "GStreamer:" in s]
    print("[PREFLIGHT] OpenCV", cv2.__version__)
    print("[PREFLIGHT]", " ".join(line) if line else "GStreamer build information unavailable")
    print("[PREFLIGHT] USB pipeline:", get_jetson_usb_pipeline(args.device))
    cap = None
    good = 0
    last = None
    try:
        cap = open_camera("jetson_usb", device=args.device)
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                print("[PREFLIGHT] Frame read failed")
                break
            if args.flip != "none":
                frame = cv2.flip(frame, int(args.flip))
            last = frame
            good += 1
            if args.show:
                cv2.imshow("Jetson USB camera preflight", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
        if last is not None:
            target = Path(__file__).resolve().parent / "screenshots" / "jetson_camera_check.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", last)
            if ok:
                data.tofile(str(target))
                print("[PREFLIGHT] Frame saved:", target)
        print(f"[PREFLIGHT] Frames: {good}/{args.frames}; frame shape: {last.shape if last is not None else None}")
        return good > 0
    except (RuntimeError, ValueError) as exc:
        print("[PREFLIGHT]", exc)
        return False
    finally:
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
