#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 01:12:29 2024

@author: ash
"""
import cv2
import datetime
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
#from tracker import*
from math import dist
#from code_light import*
import supervision as sv
import xlsxwriter as xs


model=YOLO('bestt.pt')
video_info = sv.VideoInfo.from_video_path('/home/ash/Downloads/OneDrive_2_2-29-2024/camera1_Oct12-121509.mp4')
thickness =sv.calculate_dynamic_line_thickness(resolution_wh = video_info.resolution_wh)
text_scale =sv.calculate_dynamic_text_scale(resolution_wh = video_info.resolution_wh)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

byte_track = sv.ByteTrack(frame_rate=video_info.fps)

class ViewTransformer:
   def __init__(self,source: np.ndarray, target:np.ndarray):
        source = source.astype(np.float32)
        target=target.astype(np.float32)
        self.m=cv2.getPerspectiveTransform(source,target)
        print(self.m)
        
   def transform_points(self,points:np.ndarray) -> np.ndarray:
        reshaped_points= points.reshape(-1,1,2).astype(np.float32)
        transform_points= cv2.perspectiveTransform(reshaped_points,self.m)
        return transform_points.reshape(-1,2)

def getDistanceFromPointToLine(p1, p2, p3):
    return abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))


def drawLine(event, x, y, flags, param):
    # Mouse event handlers for drawing lines
    global x1, y1, drawing, detectionLines
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:  # Start drawing a line
            x1, y1 = x, y
            drawing = True
        else:  # Stop drawing a line
            x2, y2 = x, y
            detectionLines.append([x1, y1, x2, y2])
            drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete right clicked line
        for i in detectionLines:
            p1 = np.array([i[0], i[1]])
            p2 = np.array([i[2], i[3]])
            p3 = np.array([x, y])
            if i[0] < i[2]:
                largerX = i[2]
                smallerX = i[0]
            else:
                largerX = i[0]
                smallerX = i[2]
            # Distance between the detection line and the point right clicked
            if getDistanceFromPointToLine(p1, p2, p3) < 10 and smallerX - 10 < x < largerX + 10:
                detectionLines.remove(i)





def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
global x1, y1, drawing, detectionLines
cap=cv2.VideoCapture('/home/ash/Downloads/OneDrive_2_2-29-2024/camera1_Oct12-121509.mp4')
my_file = open("/home/ash/Desktop/PYHTON YOLO/yolov8counting-trackingvehicles-main/coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)
x1 = 0
y1 = 0
drawing = False

detectionLines = []
count=0

# tracker=Tracker()
# speed = Speed()

# cy1=322
# cy2=368
offset=20

vh_down={}
vh_down_time={}
counter=[]
vh_count_down={}

vh_up={}
vh_up_time={}
counter1=[]
vh_count_up={}

ret,frame1 = cap.read()
# if not ret:
#     break
    
if count == 0:
    # User draws the detection lines on preferred lanes in the first frame
    cv2.namedWindow("RGB")
    cv2.setMouseCallback("RGB", drawLine)
    while 1:
        frameCopy1 = frame1.copy()
        frameCopy1=cv2.resize(frameCopy1,(1020,500))
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == 32 or k == 13:
            cv2.destroyAllWindows()    # Finish drawing lines by pressing enter, space or escape
            lanesCount = [0] * len(detectionLines)
            break
        for l in detectionLines:       # Plot existing lines
            cv2.line(frameCopy1, (l[0], l[1]), (l[2], l[3]), (255, 203, 48), 6)
        cv2.imshow("RGB", frameCopy1)
        
x001=detectionLines[0][0]
c0y1=detectionLines[0][1]
x002=detectionLines[0][2]
y002=detectionLines[0][3]
    
x003=detectionLines[1][0]
c0y2=detectionLines[1][1]
x004=detectionLines[1][2]
y004=detectionLines[1][3]

TARGET_WIDTH=20
TARGET_HEIGHT=30

TARGET=np.array(
    [
        [0,0],
        [TARGET_WIDTH -1,0],
        [TARGET_WIDTH-1,TARGET_HEIGHT-1],
        [0,TARGET_HEIGHT-1],
    ]
    
 )
    
SOURCE= np.array([[x001,c0y1],[x002, y002], [x003, c0y2] , [x004, y004]])
polygon_zone = sv.PolygonZone( SOURCE, frame_resolution_wh=video_info.resolution_wh, triggering_position = sv.Position.CENTER)

view_transformer= ViewTransformer(source=SOURCE,target=TARGET)

wbook=xs.Workbook('/home/ash/Desktop/PYHTON YOLO/yolov8counting-trackingvehicles-main/transport.xlsx')
ws=wbook.add_worksheet("first_sheet")
ws.write(0,0,"Date")
ws.write(0,1,"class")
ws.write(0,2,"confidence")
ws.write(0,3,"x coordinate")
ws.write(0,4,"y coordinate")
ws.write(0,5,"pixel x1")
ws.write(0,6,"pixel y1")
ws.write(0,7,"pixel x2")
ws.write(0,8,"pixel y2")
ws.write(0,9,"tracking id")



file = open("data3.txt" ,"a")
# file.write("Time_stamp"+"\t"+"\t     "+"Tracking_id" +"\t"+ "Vehicle_class" + "\t" +"Speed_of_the_Vehicle"+"\n")
row=1
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    
#     if count == 0:
#             # User draws the detection lines on preferred lanes in the first frame
#             cv2.namedWindow("RGB")
#             cv2.setMouseCallback("RGB", drawLine)
#             while 1:
#                 frameCopy = frame.copy()
#                 frameCopy=cv2.resize(frameCopy,(1020,500))
#                 k = cv2.waitKey(1) & 0xFF
#                 if k == 27 or k == 32 or k == 13:
#                     cv2.destroyAllWindows()    # Finish drawing lines by pressing enter, space or escape
#                     lanesCount = [0] * len(detectionLines)
#                     break
#                 for l in detectionLines:       # Plot existing lines
#                     cv2.line(frameCopy, (l[0], l[1]), (l[2], l[3]), (255, 203, 48), 6)
#                 cv2.imshow("RGB", frameCopy)
    frame=cv2.resize(frame,(1020,500))           
#     for dl in detectionLines:        # Plot all detection lines
#             cv2.line(frame, (dl[0], dl[1]), (dl[2], dl[3]), (255, 203, 48), 6)
    x01=detectionLines[0][0]
    cy1=detectionLines[0][1]
    x02=detectionLines[0][2]
    y02=detectionLines[0][3]
    
    x03=detectionLines[1][0]
    cy2=detectionLines[1][1]
    x04=detectionLines[1][2]
    y04=detectionLines[1][3]
    
    
    
    
    
    count += 1
    if count % 0.5 != 0:
        continue
    
   

    results=model(frame)[0]
 #   print(results)
#     a=results[0].boxes.data
#     px=pd.DataFrame(a).astype("float")
#    print(px)
#     list=[]
#              
#     for index,row in px.iterrows():
# #        print(row)
#  
#         x1=int(row[0])
#         y1=int(row[1])
#         x2=int(row[2])
#         y2=int(row[3])
#         d=int(row[5])
#         c=class_list[d]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[polygon_zone.trigger(detections)]
    detections= byte_track.update_with_detections(detections=detections)
    print(detections)
    points=[[]]
    if(len(detections.xyxy) !=0):
        points=detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        points=view_transformer.transform_points(points=points).astype(int)
    
    print(points)
        
    labels= [
        f"#{tracker_id}"
        for tracker_id in detections.tracker_id
    ]
    
    for class_id,xyxy, confidence,points, tracker_id  in zip(detections.class_id, detections.xyxy, detections.confidence,points, detections.tracker_id):
        class_name =str(class_list[class_id])
        #if class_name == 'car'or'bus'or'truck':
        if class_name in ['car', 'bus', 'truck'] and class_name != 'LCV':
            print(class_name + "\t" + str(confidence) + "\t" + str(xyxy[0]) + "\t" + str(xyxy[1]) + "\t" + str(xyxy[2]) + "\t" + str(xyxy[3])+ "\t" + str(tracker_id))
            file.write(str(datetime.datetime.fromtimestamp(time.time()))+"\t"+class_name + "\t" +str(points[0])+"\t"+ str(points[1])+"\t"+str(confidence) + "\t" + str(round(xyxy[0],6)) + "\t" + str(round(xyxy[1],6)) + "\t" + str(round(xyxy[2],6)) + "\t" + str(round(xyxy[3],6))+ "\t" + str(tracker_id)+"\n")
            ws.write(row,0,str(datetime.datetime.fromtimestamp(time.time())))
            ws.write(row,1,str(class_name))
            ws.write(row,2,str(confidence))
            ws.write(row,3,str(points[0]))
            ws.write(row,4,str(points[1]))
            ws.write(row,5,str(round(xyxy[0],6)))
            ws.write(row,6,str(round(xyxy[1],6)))
            ws.write(row,7,str(round(xyxy[2],6)))
            ws.write(row,8,str(round(xyxy[3],6)))
            ws.write(row,9,str(tracker_id))
            row=row+1
        
    RGB=frame.copy()
    RGB=sv.draw_polygon(RGB, polygon=SOURCE, color=sv.Color.RED)
    RGB=bounding_box_annotator.annotate(scene=RGB, detections=detections)
    RGB=label_annotator.annotate(scene=RGB, detections=detections, labels= labels)
    
    
#     cv2.imshow("RGB",RGB)
#     if cv2.waitKey(1)&0xFF==27:
#         break
    
# cap.release()
# cv2.destroyAllWindows()
        
        
#         if 'car' in c or 'bus' in c or 'truck' in c or 'motorcycle' in c:
# #             check = str(c)
#             list.append([x1,y1,x2,y2,str(c)])
#     bbox_id=tracker.update(list)
# 
#     
#     for bbox in bbox_id:
#         x3,y3,x4,y4,id,check=bbox
#         cx=int(x3+x4)//2
#         cy=int(y3+y4)//2
#         
#         if cy1<(cy+offset) and  cy1>(cy-offset):
#             vh_down_time[id]=time.time()
#             vh_down[id]=cy
#             vh_count_down[id]=count
#             
#         if id in vh_down:
#             if cy2<(cy+offset) and  cy2>(cy-offset):
#                 elapsed_time=time.time() - vh_down_time[id]
# #                 cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
# #                 cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                 if counter.count(id)==0:
#                     counter.append(id)
#                     distance = 10 # meters
#                     diff=count-vh_count_down[id]
# #                     speed.speed_color(diff)
#                     a_speed_ms = distance / (diff/30)
#                     a_speed_kh = a_speed_ms * 3.6
#                     times=time.time()
#                     file.write(str(datetime.datetime.fromtimestamp(times))+"\t"+str(id)+"\t           "+ str(check) + "\t"+"\t" + str(int(a_speed_kh))+"(Going_down)"+"\n")
#                     print("Hello")
# #                     file_1.write(str(time.time()) + "	" + str(id) + "	" + str(c) + "	" + str(a_speed_kh) + "\n") 
#                     cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
#                     cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                     cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                 
#                 
#         if cy2<(cy+offset) and  cy2>(cy-offset):
#             vh_up[id]=cy
#             vh_up_time[id]=time.time()
#             vh_count_up[id]=count
#         if id in vh_up:
#             if cy1<(cy+offset) and  cy1>(cy-offset):
#                 cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
#                 cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                 if counter1.count(id)==0:
#                     counter1.append(id)
#                     distance1 = 10 # meters
#                     diff1=count-vh_count_up[id]
#                     a_speed_ms1 = distance / (diff1/30)
#                     a_speed_kh1 = a_speed_ms * 3.6
#                     file.write(str(datetime.datetime.fromtimestamp(times))+"\t"+str(id)+"\t           "+ str(check) + "\t"+"\t" + str(int(a_speed_kh1))+"(Going_up)"+"\n")
#                     cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
#                     cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                     cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#                 
#                 
        
           


    cv2.line(RGB,(x01,cy1),(x02,y02),(255,255,255),1)
    cv2.putText(RGB,('Line-1'),(x01,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.line(RGB,(x03,cy2),(x04,y04),(255,255,255),1)
    cv2.putText(RGB,('Line-2'),(x03,cy2),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d=len(counter)
    #cv2.putText(RGB,('Going_Down-')+str(d),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d1=len(counter1)
    #cv2.putText(RGB,('Going_Up-')+str(d1),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", RGB)
    if cv2.waitKey(1)&0xFF==27:
        file.close()
        wbook.close()
        break
    
    
cap.release()
cv2.destroyAllWindows()

