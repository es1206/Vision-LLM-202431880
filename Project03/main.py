import cv2
import time
import csv
import subprocess
from datetime import datetime
from ultralytics import YOLO
from llama_cpp import Llama

YOLO_MODEL = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
AUDIO_FILE = "src/audio/response.wav"
LOG_FILE = "focus_log.csv"
CONFIDENCE = 0.4
STABLE_FRAMES = 3

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def speak(text):
    print("[TTS]", text)
    subprocess.run([
        PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL,
        "-f", AUDIO_FILE, "--", text
    ], check=True)
    subprocess.run(["aplay", AUDIO_FILE], check=True)


def feedback(llm, event, seconds):
    prompt = (
        "당신은 공부 집중을 도와주는 한국어 AI 코치입니다. "
        "YOLO 카메라 감지 이벤트에 맞는 짧고 친절한 한 문장을 만드세요. "
        "휴대폰이 보이면 집중 시작, 사라지면 집중 일시정지입니다. "
        "실제로 스마트폰을 사용했다고 단정하지 마세요. "
        f"이벤트: {event}. 누적 집중 시간: {int(seconds)}초. "
        "한국어 한 문장만 출력하세요."
    )
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"].strip()


def save_log(seconds):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(seconds, 1),
        ])


def main():
    yolo = YOLO(YOLO_MODEL)
    llm = Llama(
        model_path=GEMMA_MODEL,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    focused = False
    present_frames = 0
    absent_frames = 0
    total_seconds = 0.0
    started_at = 0.0

    print("휴대폰을 카메라에 놓으면 시작 / 치우면 일시정지 / q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("카메라 프레임을 읽지 못했습니다.")

        result = yolo.predict(
            source=frame, conf=CONFIDENCE, iou=0.5,
            classes=[67], verbose=False
        )[0]
        phone_visible = len(result.boxes) > 0
        present_frames = present_frames + 1 if phone_visible else 0
        absent_frames = absent_frames + 1 if not phone_visible else 0

        if not focused and present_frames >= STABLE_FRAMES:
            focused = True
            started_at = time.monotonic()
            speak(feedback(llm, "집중 시작", total_seconds))

        elif focused and absent_frames >= STABLE_FRAMES:
            total_seconds += time.monotonic() - started_at
            focused = False
            save_log(total_seconds)
            speak(feedback(llm, "집중 일시정지", total_seconds))

        elapsed = total_seconds + (time.monotonic() - started_at if focused else 0)
        status = "FOCUS" if focused else "PAUSED"
        screen = result.plot()
        cv2.putText(screen, f"{status} | {int(elapsed)} sec", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if focused else (0, 0, 255), 2)
        cv2.imshow("AI Focus Assistant", screen)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if focused:
        total_seconds += time.monotonic() - started_at
    save_log(total_seconds)
    cap.release()
    cv2.destroyAllWindows()
    print(f"총 집중 시간: {total_seconds:.1f}초")


if __name__ == "__main__":
    main()
