################################################################################
#### Made by Thanawat Sukamporn ; President of Return to monkey - Tech Team ####
################################################################################
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from pathlib import Path  # Import Path for path manipulation

# Load the trained model
model = YOLO(r'C:\Users\thana\Desktop\Project Research\Yolov8\runs\detect\train2\weights\best.pt')

# Run inference on an image
results = model(r'C:\Users\thana\Desktop\Project Research\Yolov8\dataset\val\images\24591_0.jpg')

# Access the first result and show it
result = results[0]  # Access the first item in the list
result.show()

# Save the result and prepare to load it
result.save()  # Save the result to the default directory (e.g., runs/predict)

# Convert save_dir to Path and construct the full image path
pred_image_path = Path(result.save_dir) / "24591_0.jpg"  # Adjust the filename if necessary
image = cv2.imread(str(pred_image_path))  # Read the saved image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

