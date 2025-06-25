# detection using YOLOv8n with webcam by showing what is detected in real-time
# Install the required packages using pip:  
# pip install ultralytics opencv-python
import cv2
from ultralytics import YOLO
# Load the YOLOv8n model
model = YOLO("yolov8n.pt")
# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# Process frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # Perform inference on the frame
    results = model(frame)
    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()
    # Display the annotated frame
    cv2.imshow("YOLOv8n Detection", annotated_frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

