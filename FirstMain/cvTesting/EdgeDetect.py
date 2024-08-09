import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Read in image
img = cv.imread('ps_black.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

new_width = 300
new_height = 300

# Resize the image
resized_image = cv.resize(img, (new_width, new_height))


# gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) 

# Blur using 3 * 3 kernel.
gray_blurred = cv.blur(resized_image, (3, 3)) 

#Detect all edges
edges = cv.Canny(gray_blurred,100,200)

# cv.imshow('original', img)
cv.imshow('edge detected image', edges)
cv.imwrite("edge_detected_image.png", edges)


# # Parameters to be optimized
dp_values = [1, 2]  # Inverse ratio of the accumulator resolution to the image resolution
minDist_values = [10, 20, 30]  # Minimum distance between the centers of detected circles
param1_values = [30, 40, 50]  # Upper threshold for the edge detector
param2_values = [20, 30, 40]  # Threshold for circle detection

best_params = None
best_circles = None
best_circle_count = 0

# Iterate over parameter combinations
for dp in dp_values:
    for minDist in minDist_values:
        for param1 in param1_values:
            for param2 in param2_values:
                circles = cv.HoughCircles(img,
                                           cv.HOUGH_GRADIENT,
                                           dp=dp,
                                           minDist=minDist,
                                           param1=param1,
                                           param2=param2,
                                           minRadius=0,
                                           maxRadius=0)
                if circles is not None:
                    circle_count = len(circles[0])
                    if circle_count > best_circle_count:
                        best_circle_count = circle_count
                        best_params = (dp, minDist, param1, param2)
                        best_circles = circles

# Print the best parameters and detected circles
print("Best parameters:", best_params)
print("Detected circles:", best_circles)
# cv2.imshow("Detected Circle", resized_image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()



cv.imshow('edge detected image', best_circles)
cv.waitKey(0)
cv.destroyAllWindows()