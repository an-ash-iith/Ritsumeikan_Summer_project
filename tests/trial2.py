import cv2
import numpy as np
import pandas as pd
import time
from math import dist
from ultralytics import YOLO
from tracker import* # Assuming you have a Tracker class defined in tracker.py

# Load YOLO model
model = YOLO('yolov8s.pt')

# Open a file for writing vehicle speed data
file_1 = open("data.txt", "w")

tracker=Tracker()

# Open the COCO class names file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define the line positions for vehicle counting
cy1 = 322
cy2 = 368
offset = 6

# Initialize dictionaries to store vehicle information
vh_down = {}
vh_down_time = {}
counter = []

vh_up = {}
vh_up_time = {}
counter1 = []

# Initialize video capture from a file
cap = cv2.VideoCapture('veh2.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))

    # Predict with YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Extract bounding boxes for cars
    list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        c = class_list[int(d)]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    # Update the tracker
    bbox_id = tracker.update(list)

    # Process each bounding box
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Check if the vehicle crossed the down line
        if cy1 - offset < cy < cy1 + offset:
            vh_down_time[id] = time.time()
            vh_down[id] = cy

        # Check if the vehicle crossed the up line
        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = cy
            vh_up_time[id] = time.time()

        # Calculate the speed for vehicles crossing the down line
        if id in vh_down:
            if cy2 - offset < cy < cy2 + offset:
                elapsed_time = time.time() - vh_down_time[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    file_1.write(str(a_speed_kh) + "\n")
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

        # Calculate the speed for vehicles crossing the up line
        if id in vh_up:
            if cy1 - offset < cy < cy1 + offset:
                elapsed_time = time.time() - vh_up_time[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)

    # Draw lines and text on the frame
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'Line-1', (274, cy1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'Line-2', (177, cy2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    d = len(counter)
    cv2.putText(frame, 'Going_Down-' + str(d), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    d1 = len(counter1)
    cv2.putText(frame, 'Going_Up-' + str(d1), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
