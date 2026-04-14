import cv2
import numpy as np

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def perspective_correction(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    max_width = max(max_width, 300)
    max_height = max(max_height, 300)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def preprocess_for_qr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # увеличение помогает для мелкого QR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # чуть сгладим шум
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # бинаризация
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return gray, thresh

def decode_qr_with_correction(frame, detector):
    # 1. Сначала пробуем напрямую
    data, points, _ = detector.detectAndDecode(frame)
    if data:
        return data, points, frame, True

    # 2. Если не получилось — пробуем только найти QR
    ok, points = detector.detect(frame)
    if not ok or points is None:
        return "", None, None, False

    pts = points[0]
    warped = perspective_correction(frame, pts)

    # 3. Пробуем на выровненном изображении
    data, _, _ = detector.detectAndDecode(warped)
    if data:
        return data, points, warped, True

    # 4. Пробуем после предобработки
    gray, thresh = preprocess_for_qr(warped)

    data, _, _ = detector.detectAndDecode(gray)
    if data:
        return data, points, gray, True

    data, _, _ = detector.detectAndDecode(thresh)
    if data:
        return data, points, thresh, True

    return "", points, warped, False

def draw_qr_polygon(frame, points):
    if points is not None:
        pts = points.astype(int).reshape(-1, 2)
        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    print("Нажми ESC для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        data, points, corrected, success = decode_qr_with_correction(frame, detector)

        view = frame.copy()

        if points is not None:
            draw_qr_polygon(view, points)

        if success:
            cv2.putText(view, f"QR: {data}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(view, "QR not recognized", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Camera", view)

        if corrected is not None:
            cv2.imshow("Corrected QR", corrected)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()