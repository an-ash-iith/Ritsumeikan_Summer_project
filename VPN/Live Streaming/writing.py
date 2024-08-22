import cv2

cap=cv2.VideoCapture('/home/raspproject/yolov8counting-trackingvehicles-main/veh2.mp4')
output_filename ='output3.avi'
fps=30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))





fourcc = cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter(output_filename,fourcc,fps,(frame_width, frame_height))

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        out.write(frame)
        
        cv2.imshow('frame', frame)
        
        
        if cv2.waitKey(1) & 0xFF ==27:
            break
        
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
            