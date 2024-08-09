

import cv2
import numpy as np
import math

from Tests.SecondTests.depthperi import findobject

def detect_and_draw_circles(image_path, image_path2):
    # Load the image
    img = cv2.imread(image_path)
    img2 = cv2.imread(image_path2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude2 = np.sqrt(sobel_x2 ** 2 + sobel_y2 ** 2)

    # Apply thresholding to obtain binary edges
    threshold = 110
    edges = np.uint8(gradient_magnitude > threshold) * 255
    edges2 = np.uint8(gradient_magnitude2 > threshold) * 255

    # Convert edges to binary image
    ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    ret, binary_edges2 = cv2.threshold(edges2, 0, 255, cv2.THRESH_BINARY)

    # Define parameter values
    param1_values = [200]  # Upper threshold for the edge detector
    param2_values = [45]   # Threshold for circle detection
    dp_values = [1]        # Inverse ratio of the accumulator resolution to the image resolution

    best_circles = None
    best_circle_count = 0
    best_circles2 = None
    best_circle_count2 = 0

    # Calculate minDist value
    minDist_value = img.shape[0] // 4

    # Iterate over parameter combinations
    for dp in dp_values:
        for param1 in param1_values:
            for param2 in param2_values:
                circles = cv2.HoughCircles(binary_edges,
                                           cv2.HOUGH_GRADIENT,
                                           dp=dp,
                                           minDist=minDist_value,
                                           param1=param1,
                                           param2=param2,
                                           minRadius=0,
                                           maxRadius=0)
                circles2 = cv2.HoughCircles(binary_edges2,
                                            cv2.HOUGH_GRADIENT,
                                            dp=dp,
                                            minDist=minDist_value,
                                            param1=param1,
                                            param2=param2,
                                            minRadius=0,
                                            maxRadius=0)
                if circles is not None:
                    circle_count = len(circles[0])
                    if circle_count > best_circle_count:
                        best_circle_count = circle_count
                        best_circles = np.uint16(np.around(circles))

                if circles2 is not None:
                    circle_count2 = len(circles2[0])
                    if circle_count2 > best_circle_count2:
                        best_circle_count2 = circle_count2
                        best_circles2 = np.uint16(np.around(circles2))

    if best_circles is not None:
        for circle in best_circles[0, :1]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Draw circle
            cv2.circle(img, center, radius, (0, 255, 0), 5)

            # Create a mask for the region inside of the ROI
            mask = np.zeros_like(img)
            cv2.circle(mask, center, radius - 40, (255, 255, 255), thickness=-1)  # Fill the circle in the mask

            # Set pixels outside of the ROI to a dark color (e.g., black)
            img[mask[:, :, 0] == 0] = [0, 0, 0]

        # Convert the image to RGB for displaying with matplotlib
        ROI = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        ROI = None

    if best_circles2 is not None:
        for circle in best_circles2[0, :1]:
            center2 = (circle[0], circle[1])
            radius2 = circle[2]

            # Draw circle
            cv2.circle(img2, center2, radius2, (0, 255, 0), 5)

            # Create a mask for the region inside of the ROI
            mask2 = np.zeros_like(img2)
            cv2.circle(mask2, center2, radius2 - 40, (255, 255, 255), thickness=-1)  # Fill the circle in the mask

            # Set pixels outside of the ROI to a dark color (e.g., black)
            img2[mask2[:, :, 0] == 0] = [0, 0, 0]

        # Convert the image to RGB for displaying with matplotlib
        ROI2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    else:
        ROI2 = None
    
    # roi being sent to third script
    
    

    return ROI,ROI2




