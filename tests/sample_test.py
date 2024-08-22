import cv2
import datetime
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from tracker import*

from math import dist

model=YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('camera1_Oct12-120506.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0

tracker=Tracker()


cy1=200
cy2=210
offset=10

vh_down={}
vh_down_time={}
counter=[]
vh_count_down={}

vh_up={}
vh_up_time={}
counter1=[]
vh_count_up={}

file = open("data1.txt" ,"w")

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
             
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'car' in c:
            list.append([x1,y1,x2,y2,str(c)])
    bbox_id=tracker.update(list)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(frame,str(c),(x1,y1 ),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

    for bbox in bbox_id:
        x3,y3,x4,y4,id,check=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        if cy1<(cy+offset) and  cy1>(cy-offset):
            vh_down_time[id]=time.time()
            vh_down[id]=cy
            vh_count_down[id]=count
            
        if id in vh_down:
            if cy2<(cy+offset) and  cy2>(cy-offset):
                elapsed_time=time.time() - vh_down_time[id]
                if counter.count(id)==0:
                    counter.append(id)
                    distance = 10 # meters
                    diff=count-vh_count_down[id]
                
#                     print(diff)
                    a_speed_ms = distance / (diff/30) # Convert seconds to hours
                    a_speed_kh = a_speed_ms * 3.6 # Convert m/s to km/h
                    times=time.time()
                    file.write(str(datetime.datetime.fromtimestamp(times))+"\t"+str(id)+"\t"+ str(check) + "\t" + str(int(a_speed_kh))+"\n")
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                
        if cy2<(cy+offset) and  cy2>(cy-offset):
            vh_up[id]=cy
            vh_up_time[id]=time.time()
            vh_count_up[id]=count
        if id in vh_up:
            if cy1<(cy+offset) and  cy1>(cy-offset):
                if counter1.count(id)==0:
                    counter1.append(id)
                    distance1 = 10 # meters
                    diff1=count-vh_count_up[id]
#                     print(diff1)
                    a_speed_ms1 = distance / (diff1/30) # Convert seconds to hours
                    a_speed_kh1 = a_speed_ms * 3.6 # Convert m/s to km/h
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.line(frame,(600,cy1),(1100,cy1),(255,255,255),1)
    cv2.putText(frame,('Line-1'),(600,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.line(frame,(600,cy2),(1100,cy2),(255,255,255),1)
    cv2.putText(frame,('Line-2'),(600,cy2),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d=len(counter)
    cv2.putText(frame,('Going_Down-')+str(d),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d1=len(counter1)
    cv2.putText(frame,('Going_Up-')+str(d1),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        file.close()
        break
    
cap.release()
cv2.destroyAllWindows()
