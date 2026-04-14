from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=4,
    name="robot_detector_v2",
    patience=20,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.05,
    scale=0.1,
    fliplr=0.5
)