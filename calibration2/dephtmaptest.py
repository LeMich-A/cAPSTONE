import numpy as np
import cv2
from picamera2 import Picamera2

# Initialize cameras
cap_right = Picamera2(1)                 
cap_left = Picamera2(0)

# Start the cameras
cap_right.start()
cap_left.start()

def capture_grayscale_image(camera):
    img = camera.capture_array()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def ShowDisparity(imgLeft, imgRight, bSize):
    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)

    # Normalize the image for representation
    min_val = disparity.min()
    max_val = disparity.max()
    disparity = np.uint8(255 * (disparity - min_val) / (max_val - min_val))
    
    return disparity

# Main loop to display the depth map
while True:
    # Capture grayscale images from the cameras
    imgLeft = capture_grayscale_image(cap_left)
    imgRight = capture_grayscale_image(cap_right)
    
    # Compute the depth map
    depthmap = ShowDisparity(imgLeft, imgRight, 5)

    # Generating thresholded GS image
    ret,thresh = cv2.threshold(depthmap,127,255,0)
    
    # Detecting contours in thresholded image
    im2,contours,heiarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Creating bounding box around contours
    rect = cv2.minAreaRect(contours)
    box = np.int0(cv2.boxPoints(rect))
    
    # Display the depth map
    #cv2.imshow("depthmap", depthmap)
    
    #display bounding box 
    cv2.drawContours(depthmap,[box],0,(0,0,255),2)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

            
