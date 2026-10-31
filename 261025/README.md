# 261025 Computer Vision Assignment 03

**학번: 202431880**  
**이름: 최은수**

---

## 1. 과제 개요

본 과제는 NumPy와 CuPy를 활용하여 이미지 처리 연산을 구현하고, CPU와 GPU의 처리 방식 및 성능을 비교하는 것을 목표로 한다.

주요 구현 내용은 다음과 같다.

1. **Motion Trail (잔상 효과)**
   - NumPy를 이용한 CPU 처리
   - CuPy를 이용한 GPU 병렬 처리
   - CPU/GPU 처리 시간 및 결과 이미지 비교

2. **2D Convolution (2차원 합성곱)**
   - Sobel 커널을 이용한 수직/수평 방향 에지 검출
   - NumPy와 CuPy를 이용한 Convolution 구현
   - ReLU 적용 및 CPU/GPU 결과 비교

3. **MNIST 데이터셋 `main_gpu()` 함수**
   - PyTorch를 이용한 CNN 모델 학습 및 평가
   - `main_cpu()`와 `main_gpu()`로 CPU/GPU 학습 실행
   - 학습 시간과 정확도 및 GPU 속도 향상 비율 비교

4. **CIFAR-10 데이터셋 모델 학습 및 저장**
   - PyTorch를 이용한 CIFAR-10 CNN 모델 구성
   - CUDA GPU 환경에서 5 Epoch 학습 및 평가
   - 학습한 모델을 `.pth` 파일로 저장하고 다시 불러오기

5. **IoU, NMS 구현**
   - Bounding Box 좌표 변환 및 IoU 계산
   - Confidence Thresholding을 이용한 Bounding Box 필터링
   - Non-Maximum Suppression(NMS)으로 중복 Bounding Box 제거

<br>

## 2. 파일 구조

```text
261030/
├── assignment03.py
├── README.md
└── src/
    ├── images/
    │   └── seagull.jpg
    ├── datasets/                 # 실행 시 MNIST, CIFAR-10 데이터셋 다운로드
    └── models/
        └── CIFAR10/
            └── CIFAR10_CNN.pth  # CIFAR-10 학습 완료 후 생성
```

- **assignment03.py**: NumPy와 CuPy를 활용한 이미지 처리 및 CPU/GPU 성능 비교 코드, PyTorch MNIST/CIFAR-10 모델 학습, IoU 및 NMS 구현
- **README.md**: 과제 설명 및 실행 방법
- **src/images/seagull.jpg**: 이미지 처리에 사용하는 입력 이미지
- **src/datasets/**: 실행 시 다운로드하는 MNIST 및 CIFAR-10 데이터셋 저장 위치
- **src/models/CIFAR10/CIFAR10_CNN.pth**: CIFAR-10 모델 학습 완료 후 생성되는 저장 모델 파일

<br>

## 3. 실행 방식 (필수)

### 3-1. 실행 환경

- Python 3.x
- NumPy
- CuPy
- OpenCV
- Matplotlib
- PyTorch
- Torchvision
- scikit-image
- NVIDIA CUDA 지원 GPU

### 3-2. 라이브러리 설치

```bash
pip install numpy opencv-python matplotlib scikit-image
```

CUDA 12.x 환경 기준 CuPy 설치:

```bash
pip install cupy-cuda12x
```

PyTorch와 Torchvision은 사용 중인 CUDA 버전에 맞는 빌드로 설치한다.

설치 안내: https://pytorch.org/get-started/locally/

### 3-3. 실행 방법

프로젝트 폴더에서 다음 명령어를 실행한다.

```bash
python assignment03.py
```

실행 시 이미지 처리, MNIST CPU/GPU 학습, CIFAR-10 GPU 학습 및 모델 저장, IoU/NMS 구현 코드가 순서대로 실행된다. MNIST와 CIFAR-10 데이터셋은 최초 실행 시 다운로드된다.

### 3-4. 주요 실행 과정

**① 이미지 불러오기**

- 입력 이미지를 그레이스케일로 불러온다.
- 이미지 크기를 1280×720으로 변경한다.
- 픽셀 값을 0~1 범위로 정규화한다.

**② Motion Trail 구현**

- 이동 거리에 따라 감소하는 가중치를 생성한다.
- NumPy에서는 반복문을 사용하여 픽셀별 잔상 효과를 계산한다.
- CuPy에서는 배열 슬라이싱을 이용해 GPU 병렬 연산을 수행한다.
- CPU/GPU 처리 시간과 결과 이미지를 비교한다.

**③ 2D Convolution 구현**

- Sobel 커널을 이용해 수직 및 수평 방향 에지를 검출한다.
- CPU에서는 NumPy를 사용하여 Convolution 연산을 수행한다.
- GPU에서는 CuPy 배열 연산을 사용하여 병렬 처리한다.
- ReLU를 적용한 후 CPU/GPU 결과를 비교한다.

**④ MNIST 데이터셋 `main_gpu()` 구현**

- `SimpleCNN` 모델과 `train_one_epoch()`, `evaluate()` 함수를 사용한다.
- `main_cpu()`에서 CPU로, `main_gpu()`에서 CUDA GPU로 학습한다.
- 각각 3 Epoch 동안 학습하고 테스트 데이터셋으로 평가한다.
- 학습 시간과 CPU/GPU 속도 향상 비율을 출력한다.

**⑤ CIFAR-10 데이터셋 모델 학습 및 저장**

- CIFAR-10 데이터셋을 불러오고 학습/평가용 전처리를 적용한다.
- `ConvBlock`과 `CIFAR10CNN` 모델을 사용한다.
- `train_one_epoch()`, `evaluate()` 함수를 재사용해 5 Epoch 동안 학습 및 평가한다.
- Epoch별 손실, 정확도, 실행 시간과 전체 학습 시간을 출력한다.
- 모델 가중치 및 학습 정보를 `.pth` 파일로 저장하고 불러온다.

**⑥ IoU, NMS 구현**

- Bounding Box 좌표를 `(x, y, w, h)`에서 `(x1, y1, x2, y2)`로 변환한다.
- 두 Bounding Box의 교집합 및 합집합 넓이로 IoU를 계산한다.
- Confidence Score가 임계값보다 낮은 Bounding Box를 제거한다.
- NMS를 적용해 IoU 임계값 이상으로 겹치는 Bounding Box를 제거한다.

<br>

## 4. 구현 내용

### 4-1. Motion Trail

`motion_trail_np()` 함수는 Python 반복문을 사용하여 각 출력 픽셀의 값을 계산한다.

`motion_trail_cp()` 함수는 이동 거리별로 이미지 배열을 슬라이싱하고, 각 영역의 연산을 GPU에서 병렬 처리한다.

가중치는 `np.linspace()`를 이용하여 1.0부터 0.05까지 32개를 생성하고, 전체 합이 1이 되도록 정규화하였다.

### 4-2. 2D Convolution

`convolution2d()` 함수는 NumPy를 이용하여 각 출력 위치에서 커널 연산을 수행한다.

`convolution2d_cp()` 함수는 커널 위치를 반복문으로 순차 처리하고, 각 위치의 이미지 전체 연산은 GPU에서 수행하도록 구현하였다.

수직 및 수평 방향의 특징을 추출하기 위해 Sobel 커널을 사용하였다.

**수직 방향 에지 검출 (dx)**

```text
[-1,  0,  1]
[-2,  0,  2]
[-1,  0,  1]
```

**수평 방향 에지 검출 (dy)**

```text
[-1, -2, -1]
[ 0,  0,  0]
[ 1,  2,  1]
```

Convolution 결과에는 `np.maximum()`과 `cp.maximum()`을 이용하여 ReLU를 적용하였다.

### 4-3. CPU/GPU 성능 비교

`time.perf_counter()`를 이용하여 연산 시간을 측정하였다.

GPU 처리에서는 `cp.cuda.Stream.null.synchronize()`를 사용하여 연산이 완료된 후 시간을 측정하도록 하였다.

다음 항목을 비교하도록 구현하였다.

- CPU 처리 시간
- GPU 처리 시간
- CPU → GPU 데이터 업로드 시간
- GPU → CPU 데이터 다운로드 시간
- 순수 연산 기준 속도 향상 비율
- 데이터 전송 시간을 포함한 속도 향상 비율

Convolution에서는 CPU와 GPU 결과 배열의 최대 차이도 계산하도록 구현하였다.

### 4-4. MNIST 데이터셋 `main_gpu()` 함수

`SimpleCNN`은 MNIST의 1채널 이미지를 입력받는 CNN 모델로, Convolution, ReLU, MaxPooling, Fully Connected 계층을 사용한다.

`train_one_epoch()`는 학습 데이터의 손실과 정확도를 계산하고, `evaluate()`는 평가 데이터의 손실과 정확도를 계산한다.

`main_cpu()`와 `main_gpu()`는 각각 CPU와 CUDA GPU에서 동일한 MNIST 모델을 학습하며, `run_mnist()`에서 학습 시간과 GPU 속도 향상 비율을 비교한다.

### 4-5. CIFAR-10 데이터셋 모델 학습 및 저장

`ConvBlock`은 Convolution, Batch Normalization, ReLU, MaxPooling으로 구성되며, `CIFAR10CNN`은 3개의 `ConvBlock`과 Global Average Pooling, Dropout, Linear 계층을 사용한다.

학습 데이터에는 RandomCrop과 RandomHorizontalFlip을 적용하고, 학습 및 평가 데이터 모두 정규화한다.

`run_cifar10()`에서는 CUDA GPU에서 5 Epoch 동안 학습 및 평가를 진행하고, Epoch별 손실·정확도·실행 시간 및 전체 학습 시간을 출력한다.

`torch.save()`로 모델의 가중치, 옵티마이저 상태, 테스트 정확도, 학습 Epoch, 클래스 이름, 정규화 값을 `src/models/CIFAR10/CIFAR10_CNN.pth`에 저장한다.

저장된 파일을 `torch.load()`로 불러와 새 `CIFAR10CNN` 모델에 가중치를 적용한다.

### 4-6. IoU, NMS 구현

`convert_bbox_coord()` 함수는 Bounding Box를 `(x, y, w, h)`에서 `(x1, y1, x2, y2)` 형식으로 변환한다.

`calculate_iou()` 함수는 두 Bounding Box의 교집합 넓이를 합집합 넓이로 나누어 IoU를 계산하며, 합집합 넓이가 0인 경우 0을 반환한다.

`prune_low_conf()` 함수는 Confidence Score가 기본 임계값인 0.5보다 낮은 Bounding Box를 제거한다.

`apply_nms()` 함수는 Confidence Score가 높은 순서대로 Bounding Box를 선택하고, 선택한 Bounding Box와 IoU가 임계값 이상인 나머지 Bounding Box를 제거한다.

`run_iou_nms()`에서는 예시 Bounding Box에 IoU와 Confidence Thresholding 및 NMS를 적용한다.

<br>

## 5. 기타 참고 사항

### 5-1. 코드 관련 추가 설명

- `motion_trail_np()`: NumPy를 이용한 CPU 잔상 효과 처리
- `motion_trail_cp()`: CuPy를 이용한 GPU 잔상 효과 처리
- `convolution2d()`: NumPy 기반 2차원 Convolution
- `convolution2d_cp()`: CuPy 기반 2차원 Convolution
- `np.linspace()`: 잔상 효과에 사용하는 가중치 생성
- `cp.asarray()`: NumPy 배열을 CuPy 배열로 변환
- `cp.asnumpy()`: CuPy 배열을 NumPy 배열로 변환
- `time.perf_counter()`: 실행 시간 측정
- `cp.cuda.Stream.null.synchronize()`: GPU 연산 동기화
- `SimpleCNN`: MNIST 데이터셋 CNN 모델
- `train_one_epoch()`: 학습 1 Epoch의 손실 및 정확도 계산
- `evaluate()`: 평가 데이터셋 손실 및 정확도 계산
- `main_cpu()`, `main_gpu()`: MNIST CPU/GPU 학습 실행
- `ConvBlock`, `CIFAR10CNN`: CIFAR-10 CNN 모델 구성
- `run_cifar10()`: CIFAR-10 학습, 평가 및 모델 저장
- `torch.save()`, `torch.load()`: 모델 파일 저장 및 불러오기
- `convert_bbox_coord()`: Bounding Box 좌표 형식 변환
- `calculate_iou()`: 두 Bounding Box의 IoU 계산
- `prune_low_conf()`: 낮은 Confidence Score의 Bounding Box 제거
- `apply_nms()`: 중복 Bounding Box 제거

### 5-2. 실행 환경 관련 참고 사항

본 과제의 GPU 처리 부분은 NVIDIA CUDA를 지원하는 환경에서 실행하도록 작성하였다.

MNIST 및 CIFAR-10 학습에는 PyTorch와 Torchvision이 필요하며, CUDA GPU를 사용하는 `main_gpu()`와 `run_cifar10()`은 호환되는 PyTorch CUDA 환경에서 실행해야 한다.

CIFAR-10 모델 파일은 학습이 정상적으로 완료된 뒤 생성되므로 제출 전 `.pth` 파일 생성 여부를 확인해야 한다.
