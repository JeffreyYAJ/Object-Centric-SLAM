import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import urllib.request

print("Chargement de l'IA YOLOv8...")
model = YOLO('yolov8n.pt') 

focal_length = 700.0
REAL_CAR_HEIGHT = 1.5 

url = "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4"
urllib.request.urlretrieve(url, "test_video.mp4")

cap = cv2.VideoCapture("test_video.mp4")
ret, frame = cap.read()
cap.release() 

results = model(frame, classes=[2, 5, 7], conf=0.4)[0]

img_result = frame.copy()

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    pixel_height = y2 - y1
    
    distance_z = (focal_length * REAL_CAR_HEIGHT) / pixel_height
    
    cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    label = f"Voiture: {distance_z:.1f} m"
    cv2.putText(img_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 7))
plt.imshow(img_rgb)
plt.title("Détection et Estimation de Distance avec YOLOv8")
plt.axis('off')
plt.show()
