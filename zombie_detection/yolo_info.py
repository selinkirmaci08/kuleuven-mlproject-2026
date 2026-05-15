from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.info(verbose=True)  # prints every layer, shape, and parameter count