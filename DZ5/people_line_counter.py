import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


VIDEO_PATH = "8.avi"          # путь к видео
MODEL_PATH = "yolov8n.pt"     # предобученная модель YOLO
CONF_THRESHOLD = 0.35         # порог уверенности
MAX_AGE = 30                  # сколько кадров хранить пропавший трек


# ----------------------------
# Глобальные переменные для выбора линии
# ----------------------------
line_points = []
line_ready = False


def mouse_callback(event, x, y, flags, param):
    """
    Обработчик мыши:
    пользователь кликает 2 точки, через которые проводится линия.
    """
    global line_points, line_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
        if len(line_points) == 2:
            line_ready = True


def side_of_line(point, line_p1, line_p2):
    """
    Определяет, по какую сторону от линии лежит точка.
    Возвращает положительное/отрицательное/нулевое значение.
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def get_line_zone(point, line_p1, line_p2, offset=12):
    """
    Возвращает:
    -1 -> точка по одну сторону линии
     0 -> точка вблизи линии
     1 -> точка по другую сторону линии
    """
    value = side_of_line(point, line_p1, line_p2)

    if value > offset:
        return 1
    elif value < -offset:
        return -1
    else:
        return 0

def is_point_near_segment(point, p1, p2, margin=20):
    """
    Проверяет, находится ли точка рядом с отрезком p1-p2,
    а не где-то на бесконечном продолжении линии.
    margin — допуск в пикселях.
    """
    px, py = point
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # Если отрезок вырожден в точку
    if dx == 0 and dy == 0:
        return np.hypot(px - x1, py - y1) <= margin

    # Параметр проекции точки на прямую
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # Проверяем, что проекция попадает в район отрезка
    return -0.05 <= t <= 1.05

def bbox_bottom_center(bbox):
    """
    Центр bounding box.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_info(frame, count):
    """
    Рисует текст со счётчиком.
    """
    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        3
    )


def main():
    global line_points, line_ready

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр")
        return

    preview = first_frame.copy()
    cv2.namedWindow("Select line")
    cv2.setMouseCallback("Select line", mouse_callback)

    while True:
        temp = preview.copy()

        for p in line_points:
            cv2.circle(temp, p, 5, (0, 0, 255), -1)

        if len(line_points) == 2:
            cv2.line(temp, line_points[0], line_points[1], (255, 0, 0), 2)

        cv2.putText(
            temp,
            "Click 2 points to draw the line. Press ENTER to continue.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Select line", temp)
        key = cv2.waitKey(1) & 0xFF

        # Enter — продолжить, если линия выбрана
        if key == 13 and line_ready:
            break

        # r — сбросить выбор
        if key == ord('r'):
            line_points = []
            line_ready = False

        # Esc — выход
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select line")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = YOLO(MODEL_PATH)

    tracker = DeepSort(
        max_age=MAX_AGE,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3
    )

    
    last_zone_by_id = {}       
    counted_ids = set()        
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        detections_for_tracker = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls != 0 or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1

                detections_for_tracker.append(([x1, y1, w, h], conf, "person"))

        # Обновляем трекер
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        # Рисуем линию
        cv2.line(frame, line_points[0], line_points[1], (255, 0, 0), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            point = bbox_bottom_center([x1, y1, x2, y2])

            current_zone = get_line_zone(point, line_points[0], line_points[1], offset=12)
            near_segment = is_point_near_segment(point, line_points[0], line_points[1], margin=20)

            if track_id not in last_zone_by_id:
                if current_zone != 0:
                    last_zone_by_id[track_id] = current_zone
            else:
                previous_zone = last_zone_by_id[track_id]

                if near_segment and previous_zone != 0 and current_zone != 0:
                    if previous_zone != current_zone and track_id not in counted_ids:
                        total_count += 1
                        counted_ids.add(track_id)

                if current_zone != 0:
                    last_zone_by_id[track_id] = current_zone

            # Рисуем bbox, ID и точку
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

        draw_info(frame, total_count)

        cv2.imshow("People Line Counter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Итоговое количество пересечений: {total_count}")

if __name__ == "__main__":
    main()