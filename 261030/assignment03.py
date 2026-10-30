
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    if padding > 0:
        image = np.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode="constant"
        )

    kernel_h, kernel_w = kernel.shape
    image_h, image_w = image.shape

    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = np.zeros((output_h, output_w), dtype=np.float32)

    for output_y in range(output_h):
        for output_x in range(output_w):
            input_y = output_y * stride
            input_x = output_x * stride

            region = image[
                input_y:input_y + kernel_h,
                input_x:input_x + kernel_w
            ]

            output[output_y, output_x] = np.sum(region * kernel) + bias

    return output

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0

plt.imshow(input_np, cmap="gray")
plt.axis("off");

# TODO: NumPy를 사용하여 잔상 효과 함수 정의

def motion_trail_np(image, weights):
   
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):

            pixel_sum = 0.0

            for shift in range(TRAIL_LENGTH):
                # TODO: x값 shift
                input_x = output_x - shift

                # TODO: shift한 픽셀을 누적
                if 0 <= input_x < width:
                    pixel_sum += image[output_y, input_x] * weights[shift]

            # TODO: 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = pixel_sum

    return output

# TODO: CuPy를 사용하여 잔상 효과 함수 정의

def motion_trail_cp(image, weights):
 
    # TODO: 필요한 변수 정의 및 출력 배열 생성
    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(min(len(weights), width)):
        # TODO: 배열 slicing
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region += input_region * weights[shift]

    return output

weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

weights_np /= np.sum(weights_np)

cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# TODO: NumPy 배열을 CuPy 배열로 변환

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start

warmup_image = cp.zeros((64, 64), dtype=cp.float32)
warmup_weights = cp.asarray(weights_np[:4])

_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

cp.cuda.Stream.null.synchronize()


# CuPy GPU 처리:

cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# TODO: output_cp를 CPU 메모리로 복사

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")

plt.figure(figsize=(18, 12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")


plt.figure(figsize=(12, 8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off");


plt.imshow(input_np, cmap="gray")
plt.axis("off")

# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):

    # TODO: padding 처리
    if padding > 0:
        image = cp.pad(
            image,
            ((padding, padding), (padding, padding)),
            mode="constant"
        )

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    kernel_h, kernel_w = kernel.shape
    image_h, image_w = image.shape

    output_h = (image_h - kernel_h) // stride + 1
    output_w = (image_w - kernel_w) // stride + 1

    output = cp.zeros((output_h, output_w), dtype=cp.float32)

    # TODO: 커널 위치를 Python 반복문으로 순차 처리
    for kernel_y in range(kernel_h):
        for kernel_x in range(kernel_w):

            # TODO: 각 커널 위치의 이미지 전체 연산을 GPU에서 처리
            input_region = image[
                kernel_y:kernel_y + output_h * stride:stride,
                kernel_x:kernel_x + output_w * stride:stride
            ]

            output += input_region * kernel[kernel_y, kernel_x]

    output += bias

    return output

kernel_dx_np = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

kernel_dy_np = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

kernel_dx_cp = cp.asarray(kernel_dx_np)
kernel_dy_cp = cp.asarray(kernel_dy_np)


# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

cpu_start = time.perf_counter()

output_dx_np = convolution2d(
    image=input_np,
    kernel=kernel_dx_np,
    padding=1
)

output_dy_np = convolution2d(
    image=input_np,
    kernel=kernel_dy_np,
    padding=1
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

relu_dx_np = np.maximum(output_dx_np, 0)
relu_dy_np = np.maximum(output_dy_np, 0)


# TODO: convolution을 GPU로 처리 (업로드-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
kernel_dx_cp = cp.asarray(kernel_dx_np)
kernel_dy_cp = cp.asarray(kernel_dy_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()
upload_time = upload_end - upload_start

warmup_image = cp.zeros((64, 64), dtype=cp.float32)

_ = convolution2d_cp(
    image=warmup_image,
    kernel=kernel_dx_cp,
    padding=1
)

cp.cuda.Stream.null.synchronize()

# GPU 처리
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=kernel_dx_cp,
    padding=1
)

output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=kernel_dy_cp,
    padding=1
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

download_start = time.perf_counter()

relu_dx_cp_np = cp.asnumpy(cp.maximum(output_dx_cp, 0))
relu_dy_cp_np = cp.asnumpy(cp.maximum(output_dy_cp, 0))

download_end = time.perf_counter()
download_time = download_end - download_start

# TODO: CPU/GPU 결과 비교 및 출력

compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

dx_difference = np.max(np.abs(relu_dx_np - relu_dx_cp_np))
dy_difference = np.max(np.abs(relu_dy_np - relu_dy_cp_np))

print()
print("========== Convolution CPU ==========")
print(f"CPU processing time: {cpu_time:.4f} seconds")
print()

print("========== Convolution GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Convolution Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")
print(f"Vertical max difference: {dx_difference:.6f}")
print(f"Horizontal max difference: {dy_difference:.6f}")

plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
