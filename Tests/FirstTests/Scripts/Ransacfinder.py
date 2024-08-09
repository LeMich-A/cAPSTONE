


from picamera2 import Picamera2
import cv2
import numpy as np
import os

class Streamer:
    def __init__(self, picam2):
        self.picam2 = picam2

    def start_stream(self):
        self.picam2.start()
        while True:
            image = self.picam2.capture_array()
            cv2.imshow("Frame", image)
            if cv2.waitKey(1) == ord("q"):
                cv2.imwrite("test_frame.png", image)
                break
        cv2.destroyAllWindows()

class CircleFinder:
    def find_circles(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "File could not be read, check with os.path.exists()"
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        return cimg
    
    def save_image_with_circles(self, image_path, output_path):
        cimg = self.find_circles(image_path)
        cv2.imwrite(output_path, cimg)

if __name__ == "__main__":
    picam = Picamera2(0)
    streamer = Streamer(picam)
    streamer.start_stream()

    circle_finder = CircleFinder()
    image_path = 'test_frame.png'
    output_path = 'detected_circles.png'
    circle_finder.save_image_with_circles(image_path, output_path)
    print(f"Image with detected circles saved to: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
