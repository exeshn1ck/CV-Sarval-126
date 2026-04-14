from ultralytics import YOLO

model = YOLO(r"D:\YandexDisk\MAI\2nd term\CV\DZ4\runs\detect\robot_detector_v24\weights\best.pt")

results = model(
    r"D:\YandexDisk\MAI\2nd term\CV\DZ4\dataset\images\test\frame_00087.jpg",
    save=True,
    conf=0.1
)

print(results[0].boxes)
print("Готово")