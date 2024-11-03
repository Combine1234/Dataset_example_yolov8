from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # use 'yolov8s.pt' or other versions for different model sizes

# Train the model
model.train(data='dataset.yaml', epochs=80, imgsz=1024)
