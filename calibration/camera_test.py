import time
import cv2
import numpy as np
from picamera2 import Picamera2

def capture_image(camera_id, output_filename):
    # Initialize the camera
    picam2 = Picamera2(camera_num=camera_id)
    
    # Configure the camera
    picam2.configure(picam2.create_still_configuration())
    
    # Start the camera
    picam2.start()
    
    # Allow the camera to warm up
    time.sleep(2)
    
    # Capture image
    image = picam2.capture_array()
    
    # Save the image using OpenCV
    cv2.imwrite(output_filename, image)
    print(f"Image saved from camera {camera_id} as '{output_filename}'")
    
    # Stop the camera
    picam2.stop()

def capture_images():
    try:
        # Capture image from camera 0
        capture_image(0, 'camera0.jpg')
    except Exception as e:
        print(f"Error: Could not capture image from camera 0 - {e}")

    try:
        # Capture image from camera 1
        capture_image(1, 'camera1.jpg')
    except Exception as e:
        print(f"Error: Could not capture image from camera 1 - {e}")

if __name__ == "__main__":
    capture_images()
