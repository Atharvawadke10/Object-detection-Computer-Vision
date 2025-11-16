import cv2
import torch
import numpy as np

# Load YOLOv5 model from PyTorch Hub (Pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use 'yolov5m', 'yolov5l', 'yolov5x' for larger models

# For video capture (replace with your video file or use webcam)
cap = cv2.VideoCapture(0)  # 0 for webcam, or 'video_file.mp4' for video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference (detect objects)
    results = model(frame)

    # Render results on the frame
    frame = np.squeeze(results.render()[0])

    # Show the frame
    cv2.imshow("Vehicle Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
