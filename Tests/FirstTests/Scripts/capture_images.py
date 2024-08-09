import cv2
import numpy as np
from datetime import datetime
import time
import os
from picamera2 import Picamera2, Preview

# Photo Taking presets
total_photos = 10  # Number of images to take
countdown = 5  # Interval for count-down timer, seconds
font = cv2.FONT_HERSHEY_SIMPLEX  # Countdown timer font

def TakePictures():
    val = input("Would you like to start the image capturing? (Y/N) ")

    if val.lower() == "y":
        # Initialize the camera
        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration() 
        picam2.configure(camera_config) 
        picam2.start_preview(Preview.QTGL) 
        
        picam2.start()

        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)

        counter = 0
        t2 = datetime.now()
        while counter < total_photos:
            # Setting the countdown
            t1 = datetime.now()
            countdown_timer = countdown - int((t1 - t2).total_seconds())

            # Capture image
            picam2.capture_file(f"image_{counter:02d}.png")

            counter += 1
            print(f"Image {counter} captured!")

            # Suspends execution for a few seconds
            time.sleep(1)
            countdown_timer = 0

        # Stop the camera
        picam2.stop()

    elif val.lower() == "n":
        print("Quitting! ")
    else:
        print("Please try again! ")

if __name__ == "__main__":
    TakePictures()
