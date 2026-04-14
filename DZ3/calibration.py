import cv2
import numpy as np
import glob


# ===== НАСТРОЙКИ =====
CHESSBOARD_SIZE = (9, 6)   # число ВНУТРЕННИХ углов (columns, rows)
SQUARE_SIZE = 30.0         # размер клетки в мм
IMAGE_PATHS = "D:\\YandexDisk\\MAI\\2nd term\\CV\\DZ3\\practical_part\\completed\\calibration_data\\*.jpeg"   # путь к фото


# Критерий уточнения углов
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D-точки шахматной доски в реальном мире
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Массивы для всех изображений
objpoints = []  # 3D точки
imgpoints = []  # 2D точки на изображении

images = glob.glob(IMAGE_PATHS)

if len(images) == 0:
    print("Не найдены изображения для калибровки")
    exit()

image_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners2)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners2, found)
        cv2.imshow("Corners", vis)
        cv2.waitKey(300)
    else:
        print(f"Углы не найдены: {fname}")

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("Слишком мало удачных кадров для калибровки")
    exit()

# Калибровка
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("RMS error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Сохраняем параметры
np.savez(
    "calibration_data.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs
)

# Подсчёт reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print("Mean reprojection error:", mean_error)