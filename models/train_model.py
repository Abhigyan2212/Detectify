from ultralytics import YOLO

model = YOLO('yolov10.pt')  # Pre-trained model

# Fine-tune on BCCD dataset
model.train(data='BCCD.yaml', epochs=50, imgsz=640)
