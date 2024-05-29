from picamera2 import Picamera2
import cv2
import os

# Create directories if they don't exist
left_dir = 'images/stereoLeft/'
right_dir = 'images/stereoRight/'
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

camera = Picamera2(0)
camera2 = Picamera2(1)

num = 0

while True:
    camera.start()
    camera2.start()

    img = camera.capture_array()
    img2 = camera2.capture_array()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite(left_dir + 'imageL' + str(num) + '.png', img)
        cv2.imwrite(right_dir + 'imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)

# Release resources
camera.stop()
camera2.stop()
cv2.destroyAllWindows()