# AI Color Memory Game
### 컴퓨터 비전 기반 손동작 인식 색상 기억력 게임

**학번 및 이름: 202431880 최은수**

**프로젝트 구분:** Section Project 01  
**프로젝트 기간:** 2026.10.19 ~ 2026.10.24  
**과정:** GachonUniv On-Device AI

---

## 1. 프로젝트 소개

AI Color Memory Game은 기존의 Simon 기억력 게임을 컴퓨터 비전 기술과 결합하여 제작한 실시간 손동작 인식 게임입니다.

기존 Simon 게임은 전용 전자 게임기에 표시되는 색상 순서를 기억하고 해당 버튼을 누르는 방식으로 진행됩니다.

본 프로젝트에서는 전자 게임기 대신 빨강, 노랑, 초록, 파랑의 색종이로 직접 제작한 게임판을 사용합니다.

OpenCV를 통해 색종이의 색상과 위치를 실시간으로 인식하고, MediaPipe를 통해 사용자의 검지 손가락 위치를 추적합니다. 이를 바탕으로 사용자가 제시된 색상 순서를 정확하게 선택했는지 자동으로 판별하고 점수를 계산합니다.

최종적으로 NVIDIA Jetson과 USB 웹캠을 사용하여 별도의 웹 브라우저나 전자 입력 장치 없이 실행할 수 있도록 구성하였습니다.

### 프로젝트 목표

- 수업에서 배운 이미지 처리 및 컴퓨터 비전 기술 활용
- 실시간 색상 객체 인식 및 추적 구현
- MediaPipe 기반 손가락 위치 인식
- 객체와 신체 사이의 위치 관계를 이용한 가상 버튼 구현
- 자동 정답 판정 및 점수 계산
- NVIDIA Jetson 기반 독립 실행 프로그램 구현

## 2. 기획 의도 및 차별성

기존 기억력 게임은 별도의 전자 게임기가 필요하지만, 본 프로젝트는 색종이와 웹캠만으로 게임을 진행할 수 있도록 확장하였습니다.

실제 버튼을 누르는 동작을 손가락과 색상 영역의 위치 관계로 대체하여 물리적인 입력 장치 없이도 상호작용이 가능합니다.

또한 단순히 객체를 인식하는 것에서 끝나지 않고, 객체 인식과 신체 인식을 결합하여 사용자의 행동에 따라 게임이 진행되도록 설계하였습니다.

**주요 차별성**

1. 일상적인 색종이를 게임의 입력 장치로 활용
2. 색상 영역의 위치를 실시간으로 인식하고 추적
3. 손가락을 이용한 비접촉식 가상 버튼 구현
4. 자동 정답 판정 및 실시간 점수 계산
5. 라운드별 난이도 증가 및 효과음 제공
6. 별도의 웹 브라우저 없이 Jetson에서 로컬 실행 가능

---

## 3. 게임 방법

### 3.1 게임 준비

검은색 배경판 위에 4가지 색종이를 배치합니다.

| 위치 | 색상 |
|---|---|
| 왼쪽 상단 | RED |
| 오른쪽 상단 | YELLOW |
| 왼쪽 하단 | GREEN |
| 오른쪽 하단 | BLUE |

웹캠 화면에 색종이 4개가 모두 들어오도록 게임판을 배치합니다.

프로그램이 모든 색상을 정상적으로 인식하면 화면에 `Panels: 4/4`가 표시됩니다.

### 3.2 게임 진행

**STEP 1. 게임 시작**

키보드의 SPACE를 누르면 게임이 시작되고 Start 효과음이 재생됩니다.

**STEP 2. 색상 순서 기억**

프로그램이 무작위로 생성한 색상 순서를 화면에 하나씩 표시합니다.

사용자는 제시된 순서를 기억해야 합니다.

**STEP 3. 색상 선택**

화면에 `YOUR TURN`이 나타나면 사용자는 실제 색종이 판 위에 검지 손가락을 가져갑니다.

MediaPipe가 검지 끝의 위치를 추적하고, 해당 위치가 어떤 색상 영역에 포함되는지 판별합니다.

검지를 색상 영역 내부에 약 0.35초 동안 유지하면 하나의 입력으로 인정됩니다.

정확한 중복 입력 방지를 위해 다음 색상을 선택하기 전에는 손을 모든 색상 패널 밖으로 완전히 뺐다가 다시 넣어야 합니다.

**STEP 4. 정답 판정**

입력한 색상이 정답 순서와 일치하면 Correct 효과음과 함께 점수가 즉시 증가합니다.

오답을 선택하면 Wrong 효과음이 재생되고 게임이 종료됩니다.

**STEP 5. 다음 라운드**

한 라운드의 모든 색상을 맞히면 Next 효과음이 재생되고 다음 라운드로 진행합니다.

이전 라운드의 색상 순서는 유지되며 새로운 색상이 추가됩니다.

3라운드를 모두 성공하면 Victory 효과음과 함께 최종 점수 100점이 표시됩니다.

### 3.3 라운드 및 점수 구성

| 라운드 | 기억할 색상 수 | 라운드 총점 | 성공 시 누적 점수 |
|---|---:|---:|---:|
| ROUND 1 | 3개 | 20점 | 20점 |
| ROUND 2 | 5개 | 30점 | 50점 |
| ROUND 3 | 8개 | 50점 | 100점 |

정답을 입력할 때마다 점수가 즉시 반영되도록 구현하였습니다.

각 라운드의 총점을 입력 개수에 맞게 정수로 배분하여 최종 점수가 정확히 100점이 되도록 구성하였습니다.

---

## 4. 사용 기술 및 구현 내용

본 프로젝트는 수업에서 학습한 이미지 처리, 객체 인식, 객체 추적 및 신체 인식 기술을 활용하였습니다.

| 기술 | 적용 내용 |
|---|---|
| Image Adjustment | 영상의 밝기와 대비 조정 |
| Color Space Conversion | BGR → LAB → BGR → HSV 변환 |
| Image Enhancement | CLAHE를 이용한 명암 개선 |
| Filtering | Gaussian Blur를 이용한 노이즈 감소 |
| Color Segmentation | HSV 범위에 따른 4가지 색종이 분리 |
| Morphological Operation | OPEN/CLOSE 연산을 이용한 마스크 보정 |
| Contour Detection | 색상 영역의 외곽선 추출 |
| Object Tracking | 프레임별 색상 영역 위치 갱신 및 추적 |
| Body Recognition | MediaPipe HandLandmarker를 이용한 손 인식 |
| Relationship Analysis | 검지 끝 좌표와 색상 영역의 위치 비교 |
| Result Visualization | 색상, 라운드, 점수 및 정답 여부를 영상 위에 표시 |

### 4.1 이미지 전처리

입력 영상의 밝기와 대비를 조정하고 LAB 색공간의 L 채널에 CLAHE를 적용하여 명암을 개선합니다.

이후 Gaussian Blur로 영상의 노이즈를 줄인 뒤 HSV 색공간으로 변환합니다.

### 4.2 색상 객체 인식

`cv2.inRange()`를 이용하여 빨강, 파랑, 초록, 노랑의 색상 영역을 분리합니다.

Morphological OPEN/CLOSE 연산으로 마스크를 보정한 뒤 `cv2.findContours()`를 이용하여 각 색상 영역의 외곽선을 추출합니다.

`cv2.moments()`를 활용하여 색종이 영역의 중심 좌표를 계산합니다.

### 4.3 객체 추적

각 프레임에서 검출된 색상 영역의 위치를 지속적으로 갱신합니다.

손에 의해 색종이가 일시적으로 가려지는 상황에서는 직전에 정상적으로 검출된 정보를 짧은 시간 유지하도록 처리하였습니다.

게임판의 위치를 별도로 LOCK하지 않고 실시간으로 색상을 추적하는 방식입니다.

### 4.4 MediaPipe 손 인식

수업에서 학습한 MediaPipe Tasks API의 HandLandmarker를 사용합니다.

BGR 영상을 RGB로 변환한 뒤 MediaPipe Image로 전달하여 손의 Landmark를 검출합니다.

검지 끝에 해당하는 Landmark 8번의 좌표를 이용하여 사용자가 선택한 색상을 판단합니다.

### 4.5 손동작 입력 판정

검지 끝이 색상 영역 안에 일정 시간 유지되면 해당 색상을 선택한 것으로 판단합니다.

MediaPipe의 일시적인 인식 끊김에 대응하도록 처리하였으며, 입력이 확정된 후에는 손이 모든 패널 밖으로 나가야 다음 입력을 받을 수 있도록 구현하였습니다.

---

## 5. 개발 및 실행 환경

### 5.1 개발 환경

| 항목 | 사용 환경 |
|---|---|
| 개발 언어 | Python |
| 개발 도구 | Visual Studio Code |
| 사전 개발 장비 | Windows 노트북 |
| 사전 테스트 카메라 | 노트북 내장 웹캠 |
| 영상 처리 | OpenCV, NumPy |
| 손 인식 | MediaPipe |
| 최종 실행 장비 | NVIDIA Jetson |
| 최종 입력 장치 | USB 웹캠 |
| 결과 출력 | OpenCV GUI |

### 5.2 최종 Jetson 카메라 설정

강의실 실습에서 정상 동작을 확인했던 USB 웹캠의 GStreamer 파이프라인을 기준으로 구성하였습니다.

| 항목 | 설정값 |
|---|---|
| Camera Device | /dev/video0 |
| Camera Type | USB Webcam |
| Video Format | YUY2 |
| Resolution | 640 × 480 |
| Frame Rate | 30 FPS |
| Output Format | BGR |
| Capture Backend | GStreamer |

카메라 입력은 `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` 방식으로 처리합니다.

Jetson에서는 기존 수업 실습 환경의 OpenCV와 MediaPipe를 우선 사용하도록 구성하였습니다.

---

## 6. 프로젝트 파일 구조

```text
Project01/
│
├── main.py
├── config.py
├── jetson_preflight.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── image_processor.py
│   ├── color_detector.py
│   ├── object_tracker.py
│   ├── hand_tracker.py
│   ├── touch_detector.py
│   ├── game.py
│   ├── visualizer.py
│   └── sound_manager.py
│
├── models/
│   └── hand_landmarker.task
│
├── sounds/
│   ├── Start.wav
│   ├── Correct.wav
│   ├── Wrong.wav
│   ├── Next.wav
│   └── Victory.wav
│
├── samples/
└── screenshots/
```

### 주요 파일 설명

| 파일 | 역할 |
|---|---|
| main.py | 전체 프로그램 실행 및 기능 통합 |
| config.py | 카메라, 색상 범위 및 게임 설정 |
| jetson_preflight.py | Jetson USB 웹캠 사전 연결 테스트 |
| camera.py | 노트북 및 Jetson 카메라 입력 처리 |
| image_processor.py | 이미지 전처리 및 색공간 변환 |
| color_detector.py | 색상 분리 및 외곽선 검출 |
| object_tracker.py | 색상 영역의 실시간 위치 추적 |
| hand_tracker.py | MediaPipe 기반 손가락 인식 |
| touch_detector.py | 손가락 선택 및 중복 입력 판정 |
| game.py | 라운드, 정답 및 점수 계산 |
| visualizer.py | 게임 화면 및 결과 Overlay |
| sound_manager.py | 효과음 재생 및 중복 재생 방지 |

기능별로 코드를 분리하여 가독성과 유지보수성을 높였으며, 카메라 입력 부분을 별도로 구성하여 노트북과 Jetson에서 동일한 게임 로직을 재사용할 수 있도록 하였습니다.

---

## 7. 실행 방법

### 7.1 필요한 의존성

Python 프로그램 실행에는 다음 라이브러리가 필요합니다.

- Python
- OpenCV
- NumPy
- MediaPipe
- MediaPipe HandLandmarker 모델 파일

효과음은 Windows에서 기본 오디오 기능을, Jetson에서는 ALSA의 `aplay`를 사용하도록 구성하였습니다.

### 7.2 Windows 노트북 실행

VS Code에서 `Project01` 폴더를 열고 터미널에서 다음 명령어를 실행합니다.

```powershell
.\.venv\Scripts\python.exe main.py --mode laptop
```

라이브러리를 별도로 설치해야 하는 경우:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

노트북에서는 내장 웹캠을 이용하여 색상 검출, 손 인식, 게임 진행 및 점수 계산을 테스트할 수 있습니다.

### 7.3 Jetson 실행 전 환경 확인

Jetson에서는 수업 실습 시 사용했던 기존 Python 가상환경을 활성화합니다.

Windows에서 생성한 `.venv`는 Jetson과 호환되지 않으므로 복사하여 사용하지 않습니다.

또한 기존 GStreamer 지원 OpenCV를 유지하기 위해 `opencv-python`을 무조건 재설치하지 않도록 주의합니다.

OpenCV 및 GStreamer 지원 여부 확인:

```bash
python -c "import cv2; print(cv2.__version__); print([x for x in cv2.getBuildInformation().splitlines() if 'GStreamer:' in x])"
```

MediaPipe 확인:

```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```

USB 웹캠 확인:

```bash
ls /dev/video*
```

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

### 7.4 Jetson 카메라 사전 테스트

전체 게임을 실행하기 전에 다음 명령어로 USB 웹캠 연결 상태를 확인합니다.

```bash
python jetson_preflight.py --show
```

카메라가 정상적으로 연결되면 영상이 출력되며, 마지막으로 촬영된 프레임이 다음 위치에 저장됩니다.

```text
screenshots/jetson_camera_check.png
```

이 단계에서는 MediaPipe와 게임 로직을 제외하고 카메라 입력만 확인합니다.

### 7.5 Jetson 최종 게임 실행

```bash
python main.py --mode jetson_usb --verbose
```

기본 카메라 설정이 `jetson_usb`이므로 다음 명령어로도 실행할 수 있습니다.

```bash
python main.py
```

카메라 영상이 상하 반전되어 출력되어야 하는 경우에는 다음 옵션을 사용할 수 있습니다.

```bash
python main.py --mode jetson_usb --flip 0
```

별도의 웹 브라우저 없이 OpenCV 창에서 게임을 진행합니다.

### 7.6 MediaPipe 모델 파일

손 인식을 위해 다음 위치에 모델 파일이 필요합니다.

```text
Project01/models/hand_landmarker.task
```

수업 실습에서 사용한 HandLandmarker 모델을 동일하게 활용합니다.

최종 제출 시 모델 파일이 누락되지 않도록 확인해야 합니다.

---

## 8. 키보드 조작 방법

| 키 | 기능 |
|---|---|
| SPACE | 게임 시작 |
| R | 게임 초기화 |
| S | 현재 실행 화면 저장 |
| H | HSV 색상 마스크 확인 |
| Q | 프로그램 종료 |

게임 시작 후 색상 선택은 키보드나 마우스가 아닌 사용자의 검지 손가락으로 진행합니다.

### 추가 테스트 옵션

게임 점수 계산과 손가락 인식 문제를 구분하기 위한 디버그 기능도 포함되어 있습니다.

```bash
python main.py --mode jetson_usb --debug-input
```

디버그 모드에서는 게임 입력 단계에서 키보드의 1, 2, 3, 4를 이용하여 각각 RED, BLUE, GREEN, YELLOW를 선택할 수 있습니다.

효과음을 제외하고 실행하려면 다음 옵션을 사용할 수 있습니다.

```bash
python main.py --mode jetson_usb --mute
```

---

## 9. 효과음 구현

게임의 진행 상황에 맞춰 5가지 효과음을 적용하였습니다.

| 게임 상황 | 효과음 |
|---|---|
| 게임 시작 | Start.wav |
| 정답 입력 | Correct.wav |
| 오답 입력 | Wrong.wav |
| 다음 라운드 진행 | Next.wav |
| 최종 게임 성공 | Victory.wav |

효과음별 음량 차이를 줄이기 위해 WAV 파일을 사전에 보정하였습니다.

또한 단일 재생 대기열을 이용하여 여러 효과음이 동시에 겹치지 않도록 구성하였습니다.

라운드 종료 시에는 Correct 효과음이 재생된 후 Next 또는 Victory 효과음이 순서대로 재생됩니다.

---

## 10. 테스트 및 구현 결과

### 10.1 노트북 테스트

노트북 내장 웹캠을 이용하여 다음 기능의 정상 동작을 확인하였습니다.

- 웹캠 영상 입력 및 실시간 출력
- 색종이 4개 자동 인식
- 색상 영역의 실시간 위치 추적
- MediaPipe 손 인식 및 Landmark 출력
- 검지 위치에 따른 색상 선택
- 손가락 중복 입력 방지
- 3개, 5개, 8개 색상 기억 게임 진행
- 정답 및 오답 자동 판별
- 최종 점수 100점 계산
- 5가지 효과음 순차 재생
- 실행 화면 및 결과 Overlay

### 10.2 Jetson 실행 검증

강의실에서 실습 당시 정상 동작했던 USB 웹캠의 GStreamer 파이프라인을 최종 코드에 반영하였습니다.

노트북에서 구현한 게임 로직은 유지하고, Jetson 환경에 맞는 카메라 입력과 실행 옵션을 별도로 구성하였습니다.

최종 Jetson 테스트에서는 카메라 연결, MediaPipe 모델 로딩, 색상 인식, 손가락 입력 및 효과음 출력을 확인합니다.

※ 노트북 테스트 결과와 Jetson 실기기 실행 결과는 구분하여 기록합니다.

---

## 11. 한계점 및 개선 방향

### 한계점

1. 주변 조명이나 카메라 각도에 따라 HSV 색상 인식 결과가 달라질 수 있습니다.
2. 손이 색종이를 완전히 가리면 일시적으로 객체 인식이 어려워질 수 있습니다.
3. 일반 웹캠을 사용하므로 실제 접촉 압력보다는 영상 내 손가락 위치를 기준으로 입력을 판별합니다.
4. Jetson과 노트북의 실행 환경 차이에 따라 카메라 및 오디오 설정 확인이 필요합니다.

### 개선 방향

- 다양한 조명 환경에 대응할 수 있도록 색상 인식 범위 자동 보정
- 손가락 움직임과 선택 동작의 정확도 개선
- 2인 대결 모드 및 추가 난이도 구현
- 게임 플레이 기록 및 최고 점수 저장
- Jetson 환경에 맞는 실시간 처리 성능 최적화

---

## 12. 프로젝트 제출 및 참고사항

본 프로젝트는 수업 실습 파일과 구분된 독립 실행 프로그램으로 구성하였습니다.

개인 GitHub Repository의 `Project01` 폴더에 소스 코드와 실행에 필요한 부가 파일을 함께 제출합니다.

**Repository**

https://github.com/es1206/Vision-LLM-202431880

**제출 폴더**

```text
Vision-LLM-202431880/Project01/
```

제출 시 다음 항목을 확인합니다.

- Python 소스 코드 전체
- README.md
- requirements.txt
- MediaPipe 모델 파일
- WAV 효과음 5개
- Jetson 카메라 테스트 코드
- 실제 실행에 필요한 설정 파일

Windows에서 생성한 `.venv`, 캐시 파일 등 불필요한 환경 파일은 `.gitignore`를 이용하여 제출에서 제외합니다.

GitHub Commit 메시지는 작업 내용을 구체적으로 확인할 수 있도록 작성합니다.

**최종 제출 Commit 메시지 예시**

```bash
git commit -m "Complete AI Color Memory Game with Jetson USB camera support"
```

프로젝트의 제출 시각은 GitHub Commit 기록을 기준으로 합니다.

---

## 13. 참고 자료

- GachonUniv On-Device AI 수업 자료
- 00_Initial-Setup.ipynb
- 01_Linux-and-Python.ipynb
- 02_Computer-Vision.ipynb
- OpenCV
- MediaPipe Tasks API
- NVIDIA Jetson
- 기존 Simon 기억력 게임의 진행 방식

본 프로젝트는 수업에서 학습한 이미지 처리, 객체 추적 및 MediaPipe 손 인식 기술을 기반으로 구현하였습니다.

---

**202431880 최은수**  
GachonUniv On-Device AI  
Section Project 01 — AI Color Memory Game