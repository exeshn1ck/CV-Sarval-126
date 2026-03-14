import cv2
import numpy as np

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    topLeft = pts[np.argmin(s)]
    bottomRight = pts[np.argmax(s)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    return np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.float32)

def detect_screen_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    h, w = image.shape[:2]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < 50000:
            continue

        pts = approx.reshape(4, 2)
        x, y, ww, hh = cv2.boundingRect(pts)

        aspect = ww / float(hh)
        if not (1.2 <= aspect <= 2.2):
            continue

        if area > 0.9 * w * h:
            continue

        if area > best_area:
            best_area = area
            best_quad = pts

    if best_quad is None:
        return None
    
    return order_points(best_quad)

def overlay_frame_on_screen(background, frame_to_project, screen_pts):
    screen_pts = np.array(screen_pts, dtype=np.float32).reshape(4, 2)
    screen_pts = order_points(screen_pts).astype(np.float32)

    h, w = frame_to_project.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, screen_pts)

    warped = cv2.warpPerspective(frame_to_project, H, (background.shape[1], background.shape[0]))

    mask = np.zeros(background.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, screen_pts.astype(np.int32), 255)

    mask_inv = cv2.bitwise_not(mask)

    bg_cut = cv2.bitwise_and(background, background, mask=mask_inv)
    fg_cut = cv2.bitwise_and(warped, warped, mask=mask)

    result = cv2.add(bg_cut, fg_cut)
    return result

def main():
    image_path = 'tv.jpg'
    video_path = 'video.mp4'

    image = cv2.imread(image_path)
    if image is None:
        print(f'Не удалось открыть изображение: {image_path}')
        return
    
    screen_pts = detect_screen_corners(image)

    if screen_pts is None:
        print('Экран не найден')
        return
    
    print('Найденные углы экрана')    
    print(screen_pts.astype(int))

    debug = image.copy()
    cv2.polylines(debug, [screen_pts.astype(np.int32)], True, (0, 255, 0), 3)
    for i, p in enumerate(screen_pts.astype(int)):
        cv2.circle(debug, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(debug, str(i), tuple(p + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    cv2.namedWindow("Detected screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected screen", 800, 600)
    cv2.imshow("Detected screen", debug)
    cv2.waitKey(500)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Не удалось открыть видео: {video_path}')
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25

    delay = int(1000 / fps)

    cv2.namedWindow("Video on screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video on screen", 800, 600)

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        result = overlay_frame_on_screen(image, frame, screen_pts)

        cv2.imshow("Video on screen", result)

        key = cv2.waitKey(delay) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
