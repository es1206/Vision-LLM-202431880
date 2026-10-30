
# 261031 Computer Vision Assignment 03

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

<br>

## 2. 파일 구조

```text
261030/
├── assignment03.py
├── README.md
└── src/
    └── images/
        └── seagull.jpg
```

- **assignment03.py**: NumPy와 CuPy를 활용한 이미지 처리 및 CPU/GPU 성능 비교 코드
- **README.md**: 과제 설명 및 실행 방법
- **src/images/seagull.jpg**: 이미지 처리에 사용하는 입력 이미지

<br>

## 3. 실행 방식 (필수)

### 3-1. 실행 환경

- Python 3.x
- NumPy
- CuPy
- OpenCV
- Matplotlib
- NVIDIA CUDA 지원 GPU

### 3-2. 라이브러리 설치

```bash
pip install numpy opencv-python matplotlib
```

CUDA 12.x 환경 기준 CuPy 설치:

```bash
pip install cupy-cuda12x
```

### 3-3. 실행 방법

프로젝트 폴더에서 다음 명령어를 실행한다.

```bash
python assignment03.py
```

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

### 5-2. 실행 환경 관련 참고 사항

본 과제의 GPU 처리 부분은 NVIDIA CUDA를 지원하는 환경에서 실행하도록 작성하였다.
