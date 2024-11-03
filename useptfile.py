import cv2
from ultralytics import YOLO

def count_objects_in_multiple_rois_webcam(model_path, roi_coords_list, class_names):
    """
    Continuously count objects detected within specified ROIs using YOLOv8 with a webcam feed.
    
    Parameters:
        model_path (str): Path to the YOLOv8 model file (e.g., 'best.pt').
        roi_coords_list (list): List of ROI coordinates, where each ROI is a tuple (x, y, width, height).
        class_names (list): List of class names corresponding to the model's classes.
    """
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame to speed up processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Run object detection on the full frame
        results = model.predict(frame, imgsz=640, conf=0.25)  # Adjust confidence threshold as needed

        # Initialize a list to store object counts for each ROI
        object_counts = [0] * len(roi_coords_list)

        # Iterate over detected boxes
        for result in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = result.xyxy[0]
            class_id = int(result.cls)

            # Always get the label from class_names if within range, else use class_id
            label = class_names[class_id] if class_id < len(class_names) else "unknow"

            # Check if the detected box intersects with any ROI
            for i, roi_coords in enumerate(roi_coords_list):
                roi_x, roi_y, roi_w, roi_h = roi_coords

                # Check if the bounding box is within the ROI
                if (x1 >= roi_x and y1 >= roi_y and x2 <= roi_x + roi_w and y2 <= roi_y + roi_h):
                    object_counts[i] += 1
                    
                    # Draw bounding box and label on the original frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the ROI rectangles on the frame
        for roi_coords in roi_coords_list:
            roi_x, roi_y, roi_w, roi_h = roi_coords
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Display the object counts for each ROI
        for i, count in enumerate(object_counts):
            cv2.putText(frame, f"ROI {i + 1}: {count} objects", (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow("YOLOv8 Detection with Multiple ROIs", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
model_path = r"C:\Users\thana\Desktop\Project Research\Yolov8\runs\detect\train2\weights\best.pt"  # Path to your YOLOv8 model
roi_coords_list = [(95, 6, 359, 274), (300, 200, 200, 200)]  # List of ROIs (x, y, width, height)
class_names = ["water", "mouse"]  # Ensure this matches your model's training

count_objects_in_multiple_rois_webcam(model_path, roi_coords_list, class_names)



