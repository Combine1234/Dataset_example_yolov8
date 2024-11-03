import cv2

# Initialize global variables
roi_coords = (0, 0, 0, 0)  # x, y, width, height
drawing = False  # True if the mouse is being clicked and dragged
ix, iy = -1, -1  # Initial x, y coordinates

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_coords

    # Start drawing the rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Update the rectangle while the mouse is dragged
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Copy frame to display live drawing
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Set ROI", frame_copy)

    # Finalize the rectangle on release
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Define ROI coordinates as (x, y, width, height)
        roi_coords = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        print("ROI Coordinates:", roi_coords)
        cv2.imshow("Set ROI", frame)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up mouse callback to draw ROI
cv2.namedWindow("Set ROI")
cv2.setMouseCallback("Set ROI", draw_rectangle)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Show the frame
    cv2.imshow("Set ROI", frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Output the final ROI coordinates
print("Final ROI Coordinates:", roi_coords)
