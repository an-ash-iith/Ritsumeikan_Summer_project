import cv2
from flask import Flask, Response

app = Flask(__name__)

# Set the custom resolution
custom_width = 1920
custom_height = 1080

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, custom_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, custom_height)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (custom_width, custom_height))

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Save the frame as an image
            out.write(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
