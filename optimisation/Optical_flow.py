import cv2
import numpy as np

# Function to calculate the speed of the vehicle
def calculate_speed(previous_points, current_points, fps):
    # Calculate the distance moved by the points
    distance = np.linalg.norm(current_points - previous_points, axis=1)
    
    # Calculate the speed in pixels per second
    speed = np.mean(distance) * fps
    
    return speed

# Initialize the video capture object
cap = cv2.VideoCapture("veh2.mp4")

# Set the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# Initialize the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Initialize the previous points
previous_points = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Initialize the optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize the frame counter and the speed
frame_counter = 0
speed = 0

# Loop through the video frames
while True:
    # Read the next frame
    ret, frame2 = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the optical flow
    current_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, previous_points, None, **lk_params)
    
    # Select good points
    good_new = current_points[status == 1]
    good_old = previous_points[status == 1]
    
    # Calculate the speed
    speed = calculate_speed(good_old, good_new, cap.get(cv2.CAP_PROP_FPS))
    
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)q
        frame2 = cv2.line(frame2, (a, b), (c, d), (0, 255, 0), 2)
        frame2 = cv2.circle(frame2, (a, b), 5, (0, 0, 255), -1)
    
    # Update the previous points and the previous frame
    previous_points = good_new.reshape(-1, 1, 2)
    gray1 = gray2.copy()
    
    # Increment the frame counter
    frame_counter += 1
    
    # Display the frame
    cv2.imshow('Frame', frame2)
    
    # Write the frame to the output video
    out.write(frame2)
    
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and the output video writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print the average speed
print("Average Speed:", speed)
