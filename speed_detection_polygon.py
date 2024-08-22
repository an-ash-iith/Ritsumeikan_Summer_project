import cv2

def get_mouse_coordinates(event, x, y, flags, param):
 if event == cv2.EVENT_MOUSEMOVE:
    print("Coordinates (x, y):", x, y)
        
cap=cv2.VideoCapture("/home/raspproject/yolov8counting-trackingvehicles-main/camera1_Oct12-121008.mp4")
ret,img = cap.read()
rows,cols,_=img.shape

print("Number of rows:", rows)
print("Number of columns:", cols)

# Create a window
cv2.namedWindow('img')
cv2.setMouseCallback('img', get_mouse_coordinates)
count=0
while True:
    ret,img=cap.read()
    if not ret:
         break
        
    count+=1
    if count%3 != 0:
        continue
    
    img=img[230:rows,400:cols]
    
#     if class_name in ["car" ,"truck"]:
        
        cv2.rectangle(img,(x,y),(x2,y2),color,2)
        cv2.rectangle(img,(x,y),(x+100,y-10),color,-1)
        cv2.putText(frame,class_name +" " + str(object_id) ,(x,y-10),0,0.75,(255,255,255),6)
    
    #showing the image
    cv2.imshow("img",img)
    key=cv2.waitKey(33)
    if key == 27:
        break
   
    
    
cv2.destroyAllwindows()    