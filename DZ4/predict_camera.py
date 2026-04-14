import cv2
from ultralytics import YOLO

MODEL_PATH = r"D:\YandexDisk\MAI\2nd term\CV\DZ4\runs\detect\robot_detector_v24\weights\best.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.45

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print('ERROR: Не удалось открыть веб камеру!')
    raise SystemExit

print('Нажмите ESC для выхода')

while True:
    ret, frame = cap.read()
    if not ret:
        print('ERROR: не удалось прочитать кадр')
        break

    results = model(frame, conf=CONFIDENCE, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLO Webcam Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()