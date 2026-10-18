import cv2
import numpy as np
from numpy.linalg import inv

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

# 초록 LAB lower/upper range
green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

kernel =np.ones((5,5), np.uint8)

TRACKING_MODE = 1

def kalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)

    sigma_bar = (
        A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    )

    if z is None:
        return mu_bar, sigma_bar

    else:
        K_t = (
            sigma_bar.dot(C_t.transpose()).dot(
                inv(
                    C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t
                )
            )
        )

        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))

        sigma = (
            np.identity(2) - K_t.dot(C_t)
        ).dot(sigma_bar)

        return mu, sigma

A_t = np.array([
    [1, 1],
    [0, 1]
], dtype=float)

G = np.array([
    [0.5],
    [1]
], dtype=float)

R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]], dtype=float)
Q_t = np.array([[1]], dtype=float)

mu_x = np.zeros((2, 1), dtype=float)
sigma_x = np.identity(2, dtype=float)

mu_y = np.zeros((2, 1), dtype=float)
sigma_y = np.identity(2, dtype=float)

kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array(
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ],
    dtype=np.float32,
)

kalman.measurementMatrix = np.array(
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ],
    dtype=np.float32,
)

kalman.processNoiseCov = np.array(
    [
        [1e-2,0,0,0],
        [0,1e-2,0,0],
        [0,0,5e-2,0],
        [0,0,0,5e-2],
    ],
    dtype=np.float32,
)

kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 1.0
)
kalman.errorCovPost = np.eye(4, dtype=np.float32)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera Open Failed")
    exit()

found = False

radius = 0

x_bel = 0
y_bel = 0

while True:
    ret, frame =cap.read()

    if not ret:
        break

    # TODO: 각 프레임 이미지를 전처리
    frame_blur = cv2.GaussianBlur(frame, (5,5), 0)

    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB) 

    # TODO: Color Segmentation으로 Binary Mask 생성
    green_mask =cv2.inRange(
        frame_lab,
        green_lower,
        green_upper
    )

    # TODO: Morphological Operation을 적용하여 객체 영역 추출
    mask_opened = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_closed = cv2.morphologyEx(
        mask_opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    final_ball_mask = mask_closed

    # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
    contours_lst, _ =cv2.findContours(
        final_ball_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_frame = frame.copy()

    if len(contours_lst) > 0:
        largest_contour = max(
            contours_lst,
            key=cv2.contourArea
        )

        if cv2.contourArea(largest_contour) > 500:
            (x, y),r =cv2.minEnclosingCircle(
                largest_contour
            )
            M = cv2.moments(largest_contour)

            if M['m00'] != 0:
                cX =int(M["m10"] / M["m00"])
                cY =int(M["m01"] / M["m00"])

                center = (cX, cY)
                radius = int(r)

                visible = True

                cv2.circle(
                    final_frame,
                    center,
                    radius,
                    (255, 255, 0),
                    2
                )
                cv2.circle(
                    final_frame,
                    center,
                    5,
                    (255, 0, 0),
                    -1
                )

                if not found:
                    found = True

                    mu_x = np.array(
                        [[cX],[0]],
                        dtype=float
                    )

                    mu_y = np.array(
                        [[cY],[0]],
                        dtype=float
                    )

                    sigma_x = np.eye(2, dtype=float)
                    sigma_y = np.eye(2, dtype=float)

                    initial_state = np.array(
                        [[cX], [cY], [0], [0]],
                        dtype=np.float32
                    )

                    kalman.statePre = initial_state.copy()
                    kalman.statePost = initial_state.copy()

                    x_bel = cX
                    y_bel = cY

    if found:
        if visible:
            z_x = np.array(
                [[center[0]]],
                dtype=float
            )

            z_y = np.array(
                [[center[1]]],
                dtype=float
            )

            mu_x, sigma_x = kalmanFilter(
                mu_x, sigma_x, z_x
            )

            mu_y, sigma_y = kalmanFilter(
                mu_y, sigma_y, z_y
            )

        else:
            mu_x, sigma_x = kalmanFilter(
                mu_x, sigma_x, None
            )

            mu_y, sigma_y = kalmanFilter(
                 mu_y, sigma_y, None
            )

        x_bel = mu_x[0, 0]
        y_bel = mu_y[0, 0]

    elif TRACKING_MODE ==2:
        predicted_state = kalman.predict()

        if visible:
            z = np.array(
                [
                    [center[0]],
                    [center[1]]
                ],
                dtype=np.float32
            )

            corrected_state = kalman.correct(z)

            x_bel = corrected_state[0, 0]
            y_bel = corrected_state[1, 0]

        else:
            x_bel = predicted_state[0, 0]
            y_bel = predicted_state[1, 0]

        predicted_center = (
            int(x_bel),
            int(y_bel)
        )

        cv2.circle(
            final_frame,
            predicted_center,
            radius,
            (0, 255, 255),
            2
        )

        cv2.circle(
            final_frame,
            predicted_center,
            5,
            (0, 255, 255),
            -1
        )

    cv2.imshow("VideoCapture with GStreamet", frame)
    cv2.imshow("Real-time Object Detrection", final_frame)
    cv2.imshow("Binary Mask", final_ball_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()