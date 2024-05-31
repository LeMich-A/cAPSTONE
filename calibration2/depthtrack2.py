import cv2
import numpy as np
from picamera2 import Picamera2

# Import your calibration modules
import imageprocessor as calibration

# Open both cameras
cap_right = Picamera2(0)                 
cap_left = Picamera2(1)

cap_right.start()
cap_left.start()

# Stereo vision setup parameters
frame_rate = 120    # Camera frame rate (maximum at 120 fps)
B = 6               # Distance between the cameras [cm]
f = 2.6             # Camera lens's focal length [mm]
alpha = 73          # Camera field of view in the horizontal plane [degrees]

# Compute focal length in pixels
f_pixel = (cap_right.capture_array().shape[1] * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

# Define parameters for stereo block matching
numDisparities = 32  # Must be divisible by 16
blockSize = 5      # Must be an odd number

# Create StereoBM object
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

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

        # Convert to grayscale for disparity calculation
        frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

        ################## CALCULATING DEPTH #########################################################
        # Compute the disparity map
        disparity = stereo.compute(frame_left_gray, frame_right_gray)

        # Normalize the disparity map for better visualization
        disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)

        # Calculate the depth map
        depth = (B * f_pixel) / (disparity + 1e-6)  # Add a small value to avoid division by zero

        # Detect objects based on depth changes
        threshold_depth_change = 10  # Example threshold for depth change detection in cm
        reference_depth = 40  # Reference depth in cm
        depth_difference = depth - reference_depth  # Calculate depth difference
        objects = np.where(np.abs(depth_difference) > threshold_depth_change, 255, 0).astype(np.uint8)

        # Track objects
        contours, _ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Draw bounding boxes on the right frame
            cv2.rectangle(frame_right, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Annotate the depth on the bounding box
            depth_value = depth[y + h // 2, x + w // 2]
            cv2.putText(frame_right, f'{depth_value:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #################################################################################################
        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)
        # cv2.imshow("disparity", disparity_normalized)
        # cv2.imshow("objects", objects)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

