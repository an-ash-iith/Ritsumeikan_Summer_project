import cv2
import datetime
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from tracker import*
from math import dist
from code_light import*
import RPi.GPIO as GPIO

model=YOLO('yolov8s.pt')

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

tracker=Tracker()
speed = Speed()

cy1=322
cy2=368
offset=6

nx=1920
ny=1080
sx=(nx//1020)

l1x1=274
l1x2=814
    
l2x1=177
l2x2=927

cl1x1=l1x1*sx
cl1x2=l1x2*sx
    
cl2x1=l2x1*sx
cl2x2=l2x2*sx



cy1=cy1*(ny//500)
    
cy2=cy2*(ny//500)

vh_down={}
vh_down_time={}
counter=[]
vh_count_down={}

vh_up={}
vh_up_time={}
counter1=[]
vh_count_up={}

file = open("data1.txt" ,"a")
file.write("Time_stamp"+"\t"+"\t     "+"Tracking_id" +"\t"+ "Vehicle_class" + "\t" +"Speed_of_the_Vehicle"+"\n")

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(nx,ny))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        
        if 'car' in c or 'bus' in c or 'truck' in c or 'motorcycle' in c:
#             check = str(c)
            list.append([x1,y1,x2,y2,str(c)])
    bbox_id=tracker.update(list)

    
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
#                 cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
#                 cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if counter.count(id)==0:
                    counter.append(id)
                    distance = 10 # meters
                    diff=count-vh_count_down[id]
                    speed.speed_color(diff)
                    a_speed_ms = distance / (diff/30)
                    a_speed_kh = a_speed_ms * 3.6
                    times=time.time()
                    file.write(str(datetime.datetime.fromtimestamp(times))+"\t"+str(id)+"\t           "+ str(check) + "\t"+"\t" + str(int(a_speed_kh))+"(Going_down)"+"\n")
                    print("Hello")
#                     file_1.write(str(time.time()) + "	" + str(id) + "	" + str(c) + "	" + str(a_speed_kh) + "\n") 
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                
                
        if cy2<(cy+offset) and  cy2>(cy-offset):
            vh_up[id]=cy
            vh_up_time[id]=time.time()
            vh_count_up[id]=count
        if id in vh_up:
            if cy1<(cy+offset) and  cy1>(cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if counter1.count(id)==0:
                    counter1.append(id)
                    distance1 = 10 # meters
                    diff1=count-vh_count_up[id]
                    a_speed_ms1 = distance / (diff1/30)
                    a_speed_kh1 = a_speed_ms * 3.6
                    file.write(str(datetime.datetime.fromtimestamp(times))+"\t"+str(id)+"\t           "+ str(check) + "\t"+"\t" + str(int(a_speed_kh1))+"(Going_up)"+"\n")
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                
                
        
           

    
    print(f"{cy1},{cy2},{cl1x1},{cl1x2},{cl1x1},{cl1x2}")

    cv2.line(frame,(cl1x1,cy1),(cl1x2,cy1),(255,255,255),1)
    cv2.putText(frame,('Line-1'),(cl1x1,cy1),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.line(frame,(cl2x1,cy2),(cl2x2,cy2),(255,255,255),1)
    cv2.putText(frame,('Line-2'),(cl2x1,cy2),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d=len(counter)
    cv2.putText(frame,('Going_Down-')+str(d),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    d1=len(counter1)
    cv2.putText(frame,('Going_Up-')+str(d1),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        file.close()
        GPIO.cleanup()
        break
    
    
cap.release()
cv2.destroyAllWindows()
