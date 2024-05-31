import sys
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Function for stereo vision and depth estimation
import triangulation as tri
import imageprocessor as calibration

# Open both cameras
cap_right = Picamera2(0)                 
cap_left =  Picamera2(1)

cap_right.start()
cap_left.start()


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 6               #Distance between the cameras [cm]
f = 2.6             #Camera lense's focal length [mm]
alpha = 73        #Camera field of view in the horisontal plane [degrees]




# Main program loop with depth-based object detection and tracking
while True:
    frame_right = cap_right.capture_array()
    frame_left = cap_left.capture_array()

    # If cannot capture any frame, break
    if frame_right is None or frame_left is None:                    
        break

    else:
        ################## CALIBRATION #########################################################

        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        ########################################################################################

        # Convert the BGR image to RGB
        frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        ################## CALCULATING DEPTH #########################################################

        # Calculate depth from stereo images
        depth = tri.find_depth(frame_right, frame_left, B, f, alpha)

        # Detect objects based on depth changes
        threshold_depth_change = 10  # Example threshold for depth change detection in mm
        depth_difference = depth - 40  # Calculate depth difference (refe depth)
        objects = np.where(np.abs(depth_difference) > threshold_depth_change, 255, 0).astype(np.uint8)

        # Track objects
        contours, _ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_right, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #################################################################################################

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
