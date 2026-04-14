from ultralytics import YOLO

model = YOLO(r"D:\YandexDisk\MAI\2nd term\CV\DZ4\runs\detect\robot_detector_v24\weights\best.pt")

model.predict(
    source=r"D:\YandexDisk\MAI\2nd term\CV\DZ4\video.mp4",
    save=True,
    conf=0.5
)