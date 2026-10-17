# 261017 Computer Vision Assignment 02

**학번: 202431880**  
**이름: 최은수**

## 1. 과제 개요

OpenCV를 활용하여 농구공 이미지에서 LAB 색 공간 기반 색상 분할(Color Segmentation)을 수행하고, Morphological Operation을 적용하여 최종 농구공 영역의 Contour를 검출하는 과제입니다.

주요 구현 내용은 다음과 같습니다.

- LAB 색 공간을 활용한 농구공 영역 추출
- CLAHE를 통한 이미지 명암 대비 개선
- Opening 및 Closing을 활용한 Binary Mask 노이즈 제거
- Contour 검출 및 면적이 가장 큰 농구공 영역 선택
- 최종 농구공 외곽선 시각화

## 2. 파일 구조

```text
Vision-LLM-202431880/
│
├── 261017/
│   ├── assignment02.py
│   └── README.md
│
└── src/
    └── images/
        └── basketball_crop.jpg
```

- `assignment02.py`: 농구공 이미지 처리 및 최종 Contour 검출 코드
- `README.md`: 과제 설명 및 실행 방법
- `basketball_crop.jpg`: 농구공 검출에 사용되는 원본 이미지

## 3. 실행 방법

### 실행 환경 및 필요한 라이브러리

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

필요한 라이브러리는 다음 명령어로 설치할 수 있습니다.

```bash
pip install opencv-python numpy matplotlib
```

### 실행 순서

1. 프로젝트 최상위 폴더에서 과제 폴더로 이동합니다.

```bash
cd 261017
```

2. 다음 명령어로 과제를 실행합니다.

```bash
python assignment02.py
```

3. 실행 후 농구공의 최종 검출 결과를 확인합니다.

※ 이미지 파일은 `../src/images/basketball_crop.jpg` 상대경로를 통해 불러오므로, `261017` 폴더를 현재 작업 디렉터리로 설정한 상태에서 실행해야 합니다.

## 4. 기타 참고 사항

- 수업 시간에 학습한 OpenCV의 색 공간 변환, CLAHE, Morphological Operation 및 Contour 검출 방법을 활용하였습니다.
- 농구공 검출을 위한 LAB 색상 범위는 수업에서 제시된 값을 사용하였습니다.
- Opening으로 작은 노이즈를 제거하고 Closing으로 내부의 작은 빈 공간을 보완하였습니다.
- 검출된 Contour 중 면적이 가장 큰 영역을 최종 농구공 영역으로 선택하였습니다.
- 수업 실습 파일과 구분하여 독립적으로 실행할 수 있는 Python 파일로 작성하였습니다.