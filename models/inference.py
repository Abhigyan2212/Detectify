from ultralytics import YOLO

def predict(image_path, model_path='best.pt'):
    model = YOLO(model_path)
    results = model.predict(image_path)
    return results
