
import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
from picamera2 import Picamera2

# Function for stereo vision and depth estimation
import disparitymap as dismap
import triangulation as tri
import imageprocessor as calibration

# Mediapipe for face detection
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import mediapipe as mp

# Open both cameras
cap_right = Picamera2(1)                 
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

# Define necessary functions and variables
def obstacle_avoid(depthmap, depth_thresh, output_canvas):
    # Mask to segment regions with depth less than threshold
    mask = cv2.inRange(depthmap, 10, depth_thresh)

    # Check if a significantly large obstacle is present and filter out smaller noisy regions
    if np.sum(mask) / 255.0 > 0.01 * mask.shape[0] * mask.shape[1]:
        # Contour detection 
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check if detected contour is significantly large (to avoid multiple tiny regions)
        if cv2.contourArea(cnts[0]) > 0.01 * mask.shape[0] * mask.shape[1]:
            x, y, w, h = cv2.boundingRect(cnts[0])

            # Finding average depth of region represented by the largest contour 
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, (255), -1)

            # Calculating the average depth of the object closer than the safe distance
            depth_mean, _ = cv2.meanStdDev(depthmap, mask=mask2)
            
            # Display warning text
            cv2.putText(output_canvas, "WARNING!", (x + 5, y - 40), 1, 2, (0, 0, 255), 2, 2)
            cv2.putText(output_canvas, "Object at", (x + 5, y), 1, 2, (100, 10, 25), 2, 2)
            cv2.putText(output_canvas, "%.2f cm" % depth_mean, (x + 5, y + 40), 1, 2, (100, 10, 25), 2, 2)
    else:
        cv2.putText(output_canvas, "SAFE!", (100, 100), 1, 3, (0, 255, 0), 2, 3)

    cv2.imshow('output_canvas', output_canvas)

# Stereo vision setup parameters
frame_rate = 120  # Camera frame rate (maximum at 120 fps)
B = 6  # Distance between the cameras [cm]
f = 2.6  # Camera lens's focal length [mm]
alpha = 70  # Camera field of view in the horizontal plane [degrees]

# Capture and process loop
while True:
    frame_right = cap_right.capture_array()
    frame_left = cap_left.capture_array()
    
    grayimgLeft = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    grayimgRight = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Calibration step (assuming you have a calibration module)
    frame_rightGray, frame_leftGray = calibration.undistortRectify(grayimgRight, grayimgLeft)

    # Compute the depth map
    depthmap = dismap.ShowDisparity(frame_leftGray, frame_rightGray, 5, 16)

    # Display the depth map
    cv2.imshow("depthmap", depthmap)
    
    # Create a canvas for displaying output
    output_canvas = np.zeros_like(frame_left)

    # Perform obstacle avoidance
    obstacle_avoid(depthmap, depth_thresh=100, output_canvas=output_canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_right.stop()
cap_left.stop()
cv2.destroyAllWindows()
