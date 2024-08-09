from picamera2 import Picamera2
import cv2
from Perimiterdetect import detect_and_draw_circles # Import the function from the other file

def start_stream(picam2):
    picam2.start()
    while True:
        image = picam2.capture_array()
        cv2.imshow("Frame", image)
        if cv2.waitKey(1) == ord("q"):
            cv2.imwrite("Michting.jpg", image)  # Save the captured image as Michting.jpg
            break
    cv2.destroyAllWindows()

    # Call the perimeter check function from the other file
    original_img, messages = detect_and_draw_circles("Michting.jpg")  # Call the function with the image path

    # Display images
    cv2.imshow('Original Image', original_img)
    for message in messages:
        print(message)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print messages
    
if __name__ == "__main__":
    # Initialize Picamera2
    picam2 = Picamera2()

    # Start the stream
    start_stream(picam2)




