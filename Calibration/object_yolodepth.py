import sys
import cv2
import numpy as np
import time
import torch

from picamera2 import Picamera2
import triangulation as tri
import imageprocessor as calibration

from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLO-based Object Detection Class
class YoloDetector():

    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.3):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height
        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                class_label = self.class_to_label(labels[i])
                tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                confidence = float(row[4].item())
                detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), class_label))
        return frame, detections

# Initialize Video Capture for both cameras
cap_right = Picamera2(0)
cap_left = Picamera2(1)
cap_right.start()
cap_left.start()

# Stereo vision setup parameters
frame_rate = 120  # Camera frame rate (maximum at 120 fps)
B = 6  # Distance between the cameras [cm]
f = 2.6  # Camera lens's focal length [mm]
alpha = 73  # Camera field of view in the horizontal plane [degrees]

# Initialize YOLO Detector
detector = YoloDetector(model_name=None)

# Initialize DeepSORT Tracker
object_tracker = DeepSort(max_age=5,
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

while True:
    frame_right = cap_right.capture_array()
    frame_left = cap_left.capture_array()

    # Calibration
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    # If cannot catch any frame, break
    if frame_right is None or frame_left is None:
        break

    start = time.time()

    # Object detection on both frames
    results_right = detector.score_frame(frame_right)
    results_left = detector.score_frame(frame_left)

    # Plot boxes and get detections
    frame_right, detections_right = detector.plot_boxes(results_right, frame_right, height=frame_right.shape[0], width=frame_right.shape[1], confidence=0.5)
    frame_left, detections_left = detector.plot_boxes(results_left, frame_left, height=frame_left.shape[0], width=frame_left.shape[1], confidence=0.5)

    # Tracking objects in the right frame
    tracks_right = object_tracker.update_tracks(detections_right, frame=frame_right)

    # Calculating Depth
    if detections_right and detections_left:
        for det_right, det_left in zip(detections_right, detections_left):
            bbox_right = det_right[0]
            bbox_left = det_left[0]

            center_point_right = (bbox_right[0] + bbox_right[2] / 2, bbox_right[1] + bbox_right[3] / 2)
            center_point_left = (bbox_left[0] + bbox_left[2] / 2, bbox_left[1] + bbox_left[3] / 2)

            depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "Distance: " + str(round(depth, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame_left, "Distance: " + str(round(depth, 1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            print("Depth: ", str(round(depth, 1)))

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show the frames
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
