from picamera2 import Picamera2
import cv2
import numpy as np
import tempfile

# Import your circle detection function
from Tests.SecondTests.PerimiterDetectROI import detect_and_draw_circles

def start_stream(cap_right, cap_left):
    
    cap_right.start()
    cap_left.start()
    while True:
        # Capture an image array
        imageleft = cap_left.capture_array()
        imageright = cap_right.capture_array()
        
        # Save the image array as a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(cv2.imencode('.jpg', imageleft)[1])
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file2:
            temp_file2.write(cv2.imencode('.jpg', imageright)[1])
            
        # Call your circle detection function on the temporary file
        original_img1, original_img2, messages = detect_and_draw_circles(temp_file.name,temp_file2.name)

        # Draw messages on the image
        for message in messages:
            cv2.putText(original_img1,original_img2, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            
        

        # Display the processed images
        cv2.imshow('Original Image', original_img1)
        cv2.imshow('Original Image', original_img2)

        # Exit the loop if 'm' is pressed
        key = cv2.waitKey(1)
        if key == ord('m'):
            break

    # Stop the camera before exiting
    cap_right.stop()
    cap_left.stop()

    # Close OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize Picamera2
    cap_right = Picamera2(1)
    cap_left = Picamera2(0)

   

    # Start the stream
    start_stream(cap_left,cap_right)