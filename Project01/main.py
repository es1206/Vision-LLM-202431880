"""AI Color Memory Game: Main application entry point."""
import argparse
from pathlib import Path
import time
import cv2
from config import (
    DEFAULT_CAMERA_MODE, LAPTOP_CAMERA_INDEX, JETSON_USB_DEVICE,
    JETSON_USB_FORMAT, FLIP_CODE, COLORS,
)
from src.camera import open_camera
from src.image_processor import preprocess
from src.color_detector import detect_colors
from src.object_tracker import ColorTracker
from src.hand_tracker import HandTracker
from src.touch_detector import TouchDetector
from src.game import SimonGame
from src.sound_manager import SoundManager
from src.visualizer import draw_screen, masks_preview

PROJECT_DIR = Path(__file__).resolve().parent


def find_model():
    locations = (
        PROJECT_DIR / "models" / "hand_landmarker.task",
        PROJECT_DIR.parent / "src" / "models" / "MediaPipe" / "hand_landmarker.task",
    )
    for location in locations:
        if location.is_file():
            return location
    raise FileNotFoundError(
        "Missing hand_landmarker.task. Copy it into Project01/models/."
    )


def submit_choice(game, chosen, now, sounds):
    """Play only for a NEW confirmed answer, never on every video frame.

    A completed round queues Correct -> Next, and the final completion queues
    Correct -> Victory. The SoundManager plays its queue serially, never together.
    """
    game.update(now, chosen)
    if game.last_correct:
        sounds.play("correct")
        if game.state == "ROUND_CLEAR":
            sounds.play("next")
        elif game.state == "GAME_CLEAR":
            sounds.play("victory")
    elif game.state == "GAME_OVER":
        sounds.play("wrong")


def main():
    parser = argparse.ArgumentParser(description="AI Color Memory Game game")
    parser.add_argument("--mode", choices=("laptop", "jetson_usb", "jetson_csi", "video"),
                        default=DEFAULT_CAMERA_MODE)
    parser.add_argument("--source", default=str(LAPTOP_CAMERA_INDEX))
    parser.add_argument("--device", default=JETSON_USB_DEVICE)
    parser.add_argument("--usb-format", choices=("YUY2", "MJPG"), default=JETSON_USB_FORMAT)
    parser.add_argument("--debug-input", action="store_true", help="Laptop diagnostic: keys 1/2/3/4 test scoring without webcam touch")
    parser.add_argument("--verbose", action="store_true", help="Print changes in touch state")
    parser.add_argument("--mute", action="store_true", help="Run without sound effects")
    args = parser.parse_args()

    cap = None
    hand = None
    sounds = None
    masks_visible = False
    screenshot_dir = PROJECT_DIR / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)
    try:
        hand = HandTracker(find_model())
        cap = open_camera(args.mode, args.source, args.device, args.usb_format)
        colors = ColorTracker()
        touch = TouchDetector()
        game = SimonGame()
        sounds = SoundManager(PROJECT_DIR / "sounds", enabled=not args.mute)
        print("AI Color Memory Game: live panel tracking, 4 visible panels -> SPACE=start")
        print("R=reset / S=screenshot / H=masks / Q=quit")
        if args.debug_input:
            print("DEBUG INPUT enabled: 1=RED 2=BLUE 3=GREEN 4=YELLOW")
        last_status = None
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Video/camera frame ended.")
                break
            now = time.monotonic()
            if FLIP_CODE in (-1, 0, 1):
                frame = cv2.flip(frame, FLIP_CODE)
            frame = cv2.resize(frame, (640, 480))
            hsv = preprocess(frame)
            detections, masks = detect_colors(hsv)
            live_regions = colors.update(detections, now)
            visual = frame.copy()
            fingertip, hand_count = hand.process(frame, visual)

            # Original behavior: always track live colors each frame.
            # ColorTracker keeps the last good contour across brief hand occlusion.
            regions = live_regions
            game.update(now)
            chosen, hover = touch.update(fingertip, regions, now, game.state == "INPUT")
            if args.verbose:
                status = (game.state, hover, touch.armed, touch.status)
                if status != last_status:
                    print(f"[STATE] {status} | Input {game.index}/{len(game.sequence)} | Score {game.score}")
                    last_status = status
            if chosen is not None:
                submit_choice(game, chosen, now, sounds)
                print(f"[CAMERA INPUT] {chosen} | {game.message} | Score: {game.score}")
            result = draw_screen(
                visual, regions, fingertip, hand_count, game, hover,
                touch.progress, now,
                touch_armed=touch.armed, touch_status=touch.status,
            )
            cv2.imshow("AI Color Memory Game", result)
            if masks_visible:
                cv2.imshow("AI Color Memory Game - HSV Masks", masks_preview(masks))
            key = cv2.waitKey(1) & 0xFF
            # Diagnostic option is disabled unless --debug-input is supplied.
            # It lets us distinguish score calculation from hand detection issues.
            debug_keys = {ord("1"): "RED", ord("2"): "BLUE", ord("3"): "GREEN", ord("4"): "YELLOW"}
            if args.debug_input and game.state == "INPUT" and key in debug_keys:
                debug_color = debug_keys[key]
                submit_choice(game, debug_color, now, sounds)
                print(f"[DEBUG KEY INPUT] {debug_color} | {game.message} | Score: {game.score}")
            if key == ord("q") or key == 27:
                break
            elif key == ord(" ") and game.state in ("READY", "GAME_OVER", "GAME_CLEAR"):
                if len(regions) == len(COLORS):
                    game.start(now)
                    sounds.play("start")
                    print("Game started. Panels are tracked continuously.")
                else:
                    print(f"All 4 panels must be visible: {len(regions)}/4 currently tracked.")
            elif key == ord("r"):
                sounds.clear_pending()
                game.reset()
                touch.update(None, {}, now, False)
                colors = ColorTracker()
                print("Reset. Show all 4 colors and press SPACE.")
            elif key == ord("s"):
                target = screenshot_dir / "ai_simon_game.png"
                ok_enc, png = cv2.imencode(".png", result)
                if ok_enc:
                    png.tofile(str(target))
                    print("Saved:", target)
            elif key == ord("h"):
                masks_visible = not masks_visible
                if not masks_visible:
                    cv2.destroyWindow("AI Color Memory Game - HSV Masks")
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print("ERROR:", e)
    finally:
        if sounds is not None:
            sounds.close()
        if hand is not None:
            hand.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
