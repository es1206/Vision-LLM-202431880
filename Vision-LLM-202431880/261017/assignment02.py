import cv2
import numpy as np
import matplotlib.pyplot as plt

basketball = cv2.imread('../src/images/basketball_crop.jpg')

basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(basketball_lab)

clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(8, 8))
l_clahe_enhanced = clahe.apply(l)

basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

mask_enhanced = cv2.inRange(
    basketball_enhanced_lab,
    lower_basketball_lab,
    upper_basketball_lab
)

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
final_ball_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

# TODO: Binary mask로 외곽선 검출
contours, _ = cv2.findContours(
    final_ball_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# TODO: 최종 농구공 영역 및 외곽선 선택
final_img = basketball_enhanced_rgb.copy()

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)

    largest_mask = np.zeros_like(final_ball_mask)

    cv2.drawContours(
        largest_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 0] = largest_mask

    final_img = cv2.addWeighted(final_img, 1.0, mask_color, 0.15, 0)

    cv2.drawContours(final_img, [largest_contour], -1, (255, 0, 0), 2)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(final_ball_mask, cmap='gray')
plt.title('Final Binary Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_img)
plt.title('Final Basketball Contour')
plt.axis('off')

plt.show()