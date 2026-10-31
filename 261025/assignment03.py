#assignment03

#GPU 가속 (잔상효과, Convolution2D)

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

# MNIST 데이터셋 main_gpu()
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
DATA_ROOT = "src/datasets"

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    
    model.train()

    
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    
    for images, labels in data_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        
        optimizer.zero_grad()

        
        logits = model(images)

        
        loss = criterion(logits, labels)
        
        loss.backward()
        
        optimizer.step()

        
        batch_size = images.size(0)

        
        total_loss += loss.item() * batch_size                 
        predictions = logits.argmax(dim=1)                     
        correct_count += (predictions == labels).sum().item()  
        sample_count += batch_size                             

    
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    
    model.eval()

    
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    
    with torch.no_grad():
        
        for images, labels in data_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            
            logits = model(images)

            
            loss = criterion(logits, labels)

            
            batch_size = images.size(0)

            
            total_loss += loss.item() * batch_size                 
            predictions = logits.argmax(dim=1)                     
            correct_count += (predictions == labels).sum().item()  
            sample_count += batch_size                             

        
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


def main_cpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="src/datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cpu")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    return total_training_time

# TODO: GPU 기반 MNIST 데이터셋 학습 코드 작성
def main_gpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="src/datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time


def run_mnist():
    print("=== CPU Training ===")
    cpu_training_time = main_cpu()
    print("\n=== GPU Training ===")
    gpu_training_time = main_gpu()
    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time
    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")


# CIFAR-10 데이터셋 모델 학습 및 저장

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

         
        layers = [

            
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            
            nn.BatchNorm2d(out_channels),

            
            nn.ReLU(),

            
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            
            nn.BatchNorm2d(out_channels),

            
            nn.ReLU(),

            
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
            
        ]

        
        self.block = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        
        self.features = nn.Sequential(

            
            ConvBlock(                 
                in_channels=3,
                out_channels=64,
            ),

            
            ConvBlock(                 
                in_channels=64,
                out_channels=128,
            ),

            
            ConvBlock(                 
                in_channels=128,
                out_channels=256,
            ),

        )

        
        self.global_average_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)     
                )
        )

        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),         

            nn.Linear(                 
                in_features=256,
                out_features=10,
            ),
        )

    
    def forward(self, x):
        
        
        x = self.features(x)

        
        
        x = self.global_average_pool(x)

        
        
        x = torch.flatten(x, start_dim=1)

        
        
        logits = self.classifier(x)

        return logits

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def run_cifar10():

    train_dataset = datasets.CIFAR10(
        root="src/datasets",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="src/datasets",
        train=False,
        transform=test_transform,
        download=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = CIFAR10CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    warmup_input = torch.zeros(64, 3, 32, 32, device=device)

    model.eval()

    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)

    torch.cuda.synchronize()

    # TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
    # TODO: 각 Epoch 및 전체 학습 시간을 출력

    EPOCHS = 5


    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []


    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    # TODO: 5번의 Epoch 동안 학습 및 평가 진행
    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device,
        )
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    # TODO: 전체 학습 시간 계산 및 출력
    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    from pathlib import Path
    Path("src/models/CIFAR10").mkdir(parents=True, exist_ok=True)

    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    torch.save(
        save_data,
        "src/models/CIFAR10/CIFAR10_CNN.pth",
    )

    loaded_model = CIFAR10CNN()

    state_dict = torch.load(
        "src/models/CIFAR10/CIFAR10_CNN.pth",
        map_location="cpu",
        weights_only=True,
    )

    loaded_model.load_state_dict(state_dict["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()


# IoU, NMS 구현
# TODO: 1) BBox 좌표를 (x,y,w,h)에서 (x1,y1,x2,y2)로 변환하는 함수 정의
# TODO: 2) IoU 계산 함수 정의

def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h

def calculate_iou(bbox_a, bbox_b):
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)
    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h
    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)
    union_area = area_a + area_b - intersection_area
    # TODO: ZeroDivisionError 주의
    return intersection_area / union_area if union_area > 0 else 0.0

# TODO: Confidence Thresholding 함수 정의
# TODO: Confidence Score가 특정 임계값보다 낮으면 해당 BBox 제거

def prune_low_conf(boxes, scores, conf_threshold=0.5):
    pruned_boxes = []
    pruned_scores = []
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)
    return pruned_boxes, pruned_scores

# TODO: NMS 함수 정의
# TODO: np.sort() 함수를 사용하면 1차원 배열을 정렬
# TODO: np.argsort() 함수를 사용하면 1차원 배열의 정렬된 인덱스를 반환

def apply_nms(boxes, scores, iou_threshold=0.5):
    sorted_indices = np.argsort(scores)[::-1]
    nms_boxes = []
    nms_scores = []
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        nms_boxes.append(boxes[current_idx])
        nms_scores.append(scores[current_idx])
        remaining_indices = []
        for idx in sorted_indices[1:]:
            iou = calculate_iou(boxes[current_idx], boxes[idx])
            if iou < iou_threshold:
                remaining_indices.append(idx)
        sorted_indices = np.array(remaining_indices, dtype=int)
    return nms_boxes, nms_scores

def run_iou_nms():
    from random import randint, gauss
    import skimage.data
    hubble = skimage.data.hubble_deep_field()
    def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
        img_h, img_w = img.shape[:2]
        w = randint(min_w, min(max_w, img_w))
        h = randint(min_h, min(max_h, img_h))
        x1 = randint(0, max(0, img_w - w))
        y1 = randint(0, max(0, img_h - h))
        return (x1, y1, x1 + w, y1 + h)
    boxes = [random_bbox(hubble) for _ in range(20)]
    scores = [max(0.0, min(1.0, gauss(0.7, 0.2))) for _ in range(20)]
    print("IoU:", calculate_iou((0, 0, 10, 10), (5, 5, 15, 15)))
    print(len(boxes), len(scores))
    new_boxes, new_scores = prune_low_conf(boxes, scores)
    new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)
    print(len(new_boxes), len(new_scores))


if __name__ == "__main__":
    run_mnist()
    run_cifar10()
    run_iou_nms()
