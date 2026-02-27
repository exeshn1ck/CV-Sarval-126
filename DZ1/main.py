import cv2
import sys
import os

points = []  # список центров прямоугольников
RECT_SIZE = 60  # размер прямоугольника
WINDOW_WIDTH = 800  # ширина окна
WINDOW_HEIGHT = 600  # высота окна

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Добавлен прямоугольник #{len(points)} в ({x}, {y})")

def main():
    global points
    
    # Проверка аргументов
    if len(sys.argv) < 2:
        print("Использование: python script.py 0 (камера) или python script.py video.mp4")
        return
    
    # Открываем источник
    video_source = sys.argv[1]
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
    else:
        if not os.path.exists(video_source):
            print(f"Файл {video_source} не найден")
            return
        cap = cv2.VideoCapture(video_source)
        is_camera = False
    
    if not cap.isOpened():
        print("Не удалось открыть видео источник")
        return
    
    # Создаем окно
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback('Window', mouse_callback)
    
    print("Управление: ЛКМ - отметить, C - очистить, Q - выход")
    
    while True:
        ret, frame = cap.read()

        if not ret and not is_camera:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        elif not ret:
            break

        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Рисуем прямоугольники
        for (cx, cy) in points:
            half = RECT_SIZE // 2
            pt1 = (cx - half, cy - half)
            pt2 = (cx + half, cy + half)
            cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)      
        
        cv2.imshow('Window', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Закрытие программы")
            break
        elif key == ord('c'):
            points.clear()
            print("Очищено")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()