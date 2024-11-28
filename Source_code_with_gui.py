import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout, QWidget, QFileDialog
)
from PyQt5.QtCore import QPropertyAnimation, QRect, Qt, QEasingCurve
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from supervision.geometry.core import Point, Position
from supervision.detection.line_zone import LineZone,LineZoneAnnotator
import supervision as sv 


# drawing_polygon = True
# poly_coords =[]



TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

# SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

class ViewTransformer:
    # source: and target: this is not necessary but will help what intput will be
    # -> is arrow which show return datatype -- 

    def _init_(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # this is generating the homography or m matrix-- 


        self.m = cv2.getPerspectiveTransform(source, target)
        print(self.m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        
        # we have traformation matrix of 3*3 -- so we have to change the coordinate into that 
        # so reshapint it and converting into float as it only do in floats
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
        # after that it again transforming into 2d coordinte as we want  --- 
    
class Main():
   
#    this is the default instructor -- 
    def _init_(self,source_video_path,target_video_path, Yolo_model, webcam_resolution, Confidence_Threshold, iou_threshold):
        self.source_video_path= source_video_path
        self.target_video_path= target_video_path
        self.Yolo_model=Yolo_model
        self.webcam_resolution= webcam_resolution
        self.Confidence_Threshold= float(Confidence_Threshold)
        self.iou_threshold= float(iou_threshold)
        self.drawing_polygon = True
        self.poly_coords=[]

    def mouse_callback(self, event, x, y, flags, param):
        # Capture two points on the first frame with mouse clicks
        # global drawing_polygon, poly_coords

        if event == cv2.EVENT_LBUTTONDOWN and self.drawing_polygon:
            self.poly_coords.append((x, y))
            cv2.circle(self.first_frame, (x, y), 5, (0, 255, 0), -1)
            # if you got 6 coordinates then make it false for furthur drawing --- 
            if len(self.poly_coords) == 6:
                # Set the y-coordinates based on chosen line points
                # SOURCE = np.array([poly_coords[0], poly_coords[1], poly_coords[2], poly_coords[3]])
                self.drawing_polygon = False  # Disable further drawing
                print(f"Line coordinates set to: {self.poly_coords}")

        # Deselect the last added point using right mouse click
        elif event == cv2.EVENT_RBUTTONDOWN and self.poly_coords:
            removed_point = self.poly_coords.pop()
            # Redraw the frame to remove the visual marker of the deselected point
            self.first_frame = self.org_frame.copy()  # Reset to the original frame
            for coord in self.poly_coords:
                cv2.circle(self.first_frame, coord, 5, (0, 255, 0), -1)  # Redraw remaining points
                print(f"Removed point: {removed_point}. Remaining points: {self.poly_coords}")
            self.drawing_polygon = True  # Enable drawing again if a point is removed


    def process(self):
        frame_width, frame_height = 1020,500
        # sv supervision -- 
        video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        model = YOLO(self.Yolo_model)
        names = model.names
        print(names)
        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps, track_activation_threshold=self.Confidence_Threshold
        )
         #this will automatically find the thickness of the line 
        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=[frame_width,frame_height]
        )

        text_scale = sv.calculate_optimal_text_scale(resolution_wh=[frame_width,frame_height])
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        # where you want to show the annotation --- 
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )
        
        # in real time you can't 
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        iterator = iter(frame_generator)

        

        if self.drawing_polygon:
            self.first_frame = next(iterator)
            self.first_frame=cv2.resize(self.first_frame,(frame_width, frame_height))
            self.org_frame =  self.first_frame.copy()
            while True:
                cv2.imshow("Select Line", self.first_frame)
                cv2.setMouseCallback("Select Line", self.mouse_callback)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            cv2.destroyAllWindows()
            

        SOURCE = np.array([self.poly_coords[0], self.poly_coords[1], self.poly_coords[2], self.poly_coords[3]])
        LINE_START=Point(self.poly_coords[4][0],self.poly_coords[4][1])
        LINE_END =Point(self.poly_coords[5][0],self.poly_coords[5][1])

        polygon_zone = sv.PolygonZone(polygon=SOURCE)
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        line_zone = LineZone(start=LINE_START,end=LINE_END, triggering_anchors=[Position.BOTTOM_CENTER])
        line_zone_annotator = LineZoneAnnotator(thickness=thickness, text_thickness=thickness, text_scale=text_scale)
        frame_count = video_info.fps
        
        # this is the dequeue to contain all the frames of prev one second --- 
        coordinates = defaultdict(lambda: deque(maxlen=frame_count))

        with sv.VideoSink(self.target_video_path, video_info) as sink:
            for frame in frame_generator:
                    
                    # for make process faster you are changing the frame_width and height --

                    frame=cv2.resize(frame,(frame_width, frame_height))
                    result = model(frame)[0]
                    detections = sv.Detections.from_ultralytics(result)
                    # you are extracting form same variale and equating to same -- 
                    detections = detections[detections.confidence > self.Confidence_Threshold]
                    # this is for the particular calss with id = 2 (car ) and class_id =  7 ( for bus )
                    detections = detections[(detections.class_id==2) | (detections.class_id==7)]
                    detections = detections[polygon_zone.trigger(detections)]
                    # this is to find the intersection over union threshold --- 
                    detections = detections.with_nms(threshold=self.iou_threshold)
                    detections = byte_track.update_with_detections(detections=detections)

                    points = detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )
                    points = view_transformer.transform_points(points=points).astype(int)

                    for tracker_id, [_, y] in zip(detections.tracker_id, points):
                        coordinates[tracker_id].append(y)

                    labels = []
                    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                        # this is taking care of flickering -- if number of frame is not this 
                        # then only show tracker id and class name not speed 
                        if len(coordinates[tracker_id]) < video_info.fps / 2:
                            labels.append(f"#{tracker_id} {names[class_id]}")
                        else:
                            # *************** speed procedure  ***************
                            coordinate_start = coordinates[tracker_id][-1]
                            coordinate_end = coordinates[tracker_id][0]
                            distance = abs(coordinate_start - coordinate_end)
                            time = len(coordinates[tracker_id]) / (video_info.fps)
                            speed = distance / time * 3.6
                            labels.append(f"#{tracker_id} {names[class_id]} {int(speed)} km/h")

                    # it will trigger in and out detection 
                    line_zone.trigger(detections=detections)

                    print(detections)

                    annotated_frame = frame.copy()
                    annotated_frame = sv.draw_polygon(annotated_frame, polygon= SOURCE, color= sv.Color.RED)
                    annotated_frame = trace_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    print(line_zone.in_count_per_class)
                    print(line_zone.out_count_per_class)
                    annotated_frame= line_zone_annotator.annotate(frame=annotated_frame, 
                                                                  line_counter=line_zone
                    )
                    annotated_frame = box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )
                    annotated_frame = cv2.resize(annotated_frame,(video_info.resolution_wh[0],video_info.resolution_wh[1]))
                    sink.write_frame(annotated_frame)
                    annotated_frame = cv2.resize(annotated_frame,(1020,500))
                    cv2.imshow("frame", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cv2.destroyAllWindows()


class ArgumentGUI(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Vehicle Speed Estimation")
        self.setGeometry(300, 100, 600, 500)
        self.initUI()

    def initUI(self):
        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        self.main_layout = QVBoxLayout(self.central_widget)
        self.form_layout = QFormLayout()
        
        # Widgets
        self.source_video_path = QLineEdit()
        self.source_video_path.setPlaceholderText("Path to the source video file")
        self.source_browse_button = QPushButton("Browse")
        self.source_browse_button.clicked.connect(self.browse_source_video)
        
        self.target_video_path = QLineEdit()
        self.target_video_path.setPlaceholderText("Path to the target video file")
        self.target_browse_button = QPushButton("Browse")
        self.target_browse_button.clicked.connect(self.browse_target_video)

        self.Yolo_model = QLineEdit("yolo11n.pt")
        self.Yolo_model.setPlaceholderText("Yolo Model (e.g., yolov8n.pt)")
        self.Yolo_model.setValidator(QIntValidator())

        self.webcam_resolution = QLineEdit("1020,500")
        self.webcam_resolution.setPlaceholderText("Webcam resolution (e.g., 1920,1080)")
        self.webcam_resolution.setValidator(QIntValidator())

        self.confidence_threshold = QLineEdit("0.3")
        self.confidence_threshold.setPlaceholderText("Confidence threshold (e.g., 0.3)")
        self.confidence_threshold.setValidator(QDoubleValidator(0.0, 1.0, 2))

        self.iou_threshold = QLineEdit("0.7")
        self.iou_threshold.setPlaceholderText("IOU threshold (e.g., 0.7)")
        self.iou_threshold.setValidator(QDoubleValidator(0.0, 1.0, 2))

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_arguments)

        # Add to form layout
        self.form_layout.addRow("Source Video Path:", self.create_input_with_button(self.source_video_path, self.source_browse_button))
        self.form_layout.addRow("Target Video Path:", self.create_input_with_button(self.target_video_path, self.target_browse_button))
        self.form_layout.addRow("Yolo Model:", self.Yolo_model)
        self.form_layout.addRow("Webcam Resolution:", self.webcam_resolution)
        self.form_layout.addRow("Confidence Threshold:", self.confidence_threshold)
        self.form_layout.addRow("IOU Threshold:", self.iou_threshold)

        # Add animations
        self.add_animations()

        # Add layouts
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.submit_button)

    def create_input_with_button(self, line_edit, button):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        layout.setSpacing(5)
        widget.setLayout(layout)
        return widget

    def browse_source_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Source Video")
        if path:
            self.source_video_path.setText(path)

    def browse_target_video(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Target Video")
        if path:
            self.target_video_path.setText(path)

    def add_animations(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(800)
        self.animation.setStartValue(QRect(300, 100, 600, 500))
        self.animation.setEndValue(QRect(250, 80, 700, 600))
        self.animation.setEasingCurve(QEasingCurve.Type.OutBounce)
        # self.animation.setEasingCurve(Qt.OutBounce)
        self.animation.start()

    def submit_arguments(self):
        # Collect inputs
        args = {
            "source_video_path": self.source_video_path.text(),
            "target_video_path": self.target_video_path.text(),
            "yolo_model" : self.Yolo_model.text(),
            "webcam_resolution": self.webcam_resolution.text(),
            "confidence_threshold": self.confidence_threshold.text(),
            "iou_threshold": self.iou_threshold.text(),
        }
        # Basic validation
        missing_fields = [k for k, v in args.items() if not v]
        if missing_fields:
            self.show_message("Error", f"Please fill all the fields: {', '.join(missing_fields)}")
        else:
            self.show_message("Success", f"Arguments Submitted: {args}")
        main = Main(self.source_video_path.text(),self.target_video_path.text(),self.Yolo_model.text(), self.webcam_resolution.text(), self.confidence_threshold.text(), self.iou_threshold.text())
        main.process()

    def show_message(self, title, message):
        self.message_box = QLabel(message)
        self.message_box.setWindowTitle(title)
        self.message_box.setGeometry(300, 200, 400, 200)
        self.message_box.setAlignment(Qt.AlignCenter)
        self.message_box.show()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ArgumentGUI()
    gui.show()
    sys.exit(app.exec_())