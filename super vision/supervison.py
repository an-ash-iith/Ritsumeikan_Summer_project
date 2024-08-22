import cv2
import datetime
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from math import dist
import supervision as sv

model=YOLO('yolov8s.pt')
video_info = sv.VideoInfo.from_video_path('veh2.mp4')
thickness =sv.calculate_dynamic_line_thickness(resolution_wh = video_info.resolution_wh)
text_scale =sv.calculate_dynamic_text_scale(resolution_wh = video_info.resolution_wh)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

byte_track = sv.ByteTrack(frame_rate=video_info.fps)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('veh2.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
#     frame=cv2.resize(frame,(1020,500))
   

    results=model(frame)[0]
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    print(px)
#     print(results)

    detections = sv.Detections.from_ultralytics(results)
    detections= byte_track.update_with_detections(detections=detections)
    print(detections)
    
    labels= [
        f"#{tracker_id}"
        for tracker_id in detections.tracker_id
    ]
    
    for class_id,xyxy, confidence in zip(detections.class_id, detections.xyxy, detections.confidence):
        class_name =str(class_list[class_id])
        if class_name == 'car':
            print(class_name + "\t" + str(confidence) + "\t" + str(xyxy[0]) + "\t" + str(xyxy[1]) + "\t" + str(xyxy[2]) + "\t" + str(xyxy[3]))
            
        
    RGB=frame.copy()
    RGB=bounding_box_annotator.annotate(scene=RGB, detections=detections)
    RGB=label_annotator.annotate(scene=RGB, detections=detections, labels= labels)
    
    
    cv2.imshow("RGB",RGB)
    if cv2.waitKey(1)&0xFF==27:
        break
    
cap.release()
cv2.destroyAllWindows()
