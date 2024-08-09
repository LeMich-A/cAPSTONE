import numpy as np
import cv2
from picamera2 import Picamera2



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