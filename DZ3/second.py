import cv2
import numpy as np


# ===== НАСТРОЙКИ =====
CHESSBOARD_SIZE = (7, 5)   # внутренние углы
SQUARE_SIZE = 30.0         # мм

# Загружаем калибровку
data = np.load("calibration_data.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# 3D-точки шахматной доски
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        success, rvec, tvec = cv2.solvePnP(
            objp, corners2, camera_matrix, dist_coeffs
        )

        if success:
            # Полное расстояние от камеры до начала координат доски
            distance = np.linalg.norm(tvec)

            # Глубина вдоль оси камеры
            z = tvec[2][0]

            cv2.putText(frame, f"Distance: {distance:.1f} mm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Z: {z:.1f} mm", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Рисуем оси
            axis = np.float32([
                [3 * SQUARE_SIZE, 0, 0],
                [0, 3 * SQUARE_SIZE, 0],
                [0, 0, -3 * SQUARE_SIZE]
            ])

            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

            corner = tuple(corners2[0].ravel().astype(int))
            imgpts = imgpts.astype(int)

            cv2.line(frame, corner, tuple(imgpts[0].ravel()), (0, 0, 255), 3)   # X
            cv2.line(frame, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 3)   # Y
            cv2.line(frame, corner, tuple(imgpts[2].ravel()), (255, 0, 0), 3)   # Z

        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, found)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()