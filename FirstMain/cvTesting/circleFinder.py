# # # import cv2 
# # # import numpy as np 

# # # # Read image.
# # # img = cv2.imread('back_Airpods.jpg', cv2.IMREAD_COLOR) 

# # # # Convert to grayscale.
# # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# # # # Blur using 3 * 3 kernel.
# # # gray_blurred = cv2.blur(gray, (3, 3)) 

# # # # Apply Hough transform on the blurred image.
# # # detected_circles = cv2.HoughCircles(gray_blurred,
# # # cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
# # # param2 = 30, minRadius = 10, maxRadius = 40) 

# # # # Draw circles that are detected.
# # # if detected_circles is not None: 

# # # # Convert the circle parameters a, b and r to integers.
# # #     detected_circles = np.uint16(np.around(detected_circles)) 

# # #     for pt in detected_circles[0, :]: 
# # #         a, b, r = pt[0], pt[1], pt[2] 

# # #     # Draw the circumference of the circle.
# # #     cv2.circle(img, (a, b), r, (0, 255, 0), 2) 

# # #     # Draw a small circle (of radius 1) to show the center.
# # #     cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
# # #     cv2.imshow("Detected Circle", img) 
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()





# # import cv2 
# # import numpy as np 

# # # Read image.
# # img = cv2.imread('ps_black.jpg', cv2.IMREAD_GRAYSCALE)
# # assert img is not None, "file could not be read, check with os.path.exists()"

# # new_width = 300
# # new_height = 300

# # # Resize the image
# # resized_image = cv2.resize(img, (new_width, new_height))


# # #Detect all edges
# # edges = cv2.Canny(resized_image,100,200)

# # cv2.imshow("Detected Circle", edges) 


# # # # Convert to grayscale.
# # # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) 

# # # Blur using 3 * 3 kernel.
# # # gray_blurred = cv2.blur(resized_image, (3, 3)) 

# # # Apply Hough transform on the blurred image.
# # detected_circles = cv2.HoughCircles(edges,
# #                                     cv2.HOUGH_GRADIENT,
# #                                     1,
# #                                     20,
# #                                     param1=50,
# #                                     param2=200,  # Increased param2 value
# #                                     minRadius=10,
# #                                     maxRadius=100)  # Increased maxRadius value

# # # Draw circles that are detected.
# # if detected_circles is not None: 
# #     # Convert the circle parameters a, b and r to integers.
# #     detected_circles = np.uint16(np.around(detected_circles)) 

# #     for pt in detected_circles[0, :1]: 
# #         a, b, r = pt[0], pt[1], pt[2] 

# #         # Draw the circumference of the circle.
# #         cv2.circle(resized_image, (a, b), r, (0, 255, 0), 2) 

# #         # Draw a small circle (of radius 1) to show the center.
# #         cv2.circle(resized_image, (a, b), 1, (0, 0, 255), 3) 

# # # cv2.imshow("Detected Circle", resized_image) 
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# #Read in image
# img = cv.imread('ps_black.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"

# new_width = 300
# new_height = 300

# # Resize the image
# resized_image = cv.resize(img, (new_width, new_height))

# # Blur using 3 * 3 kernel.
# gray_blurred = cv.blur(resized_image, (3, 3)) 

# #Detect all edges

# #Detect all edges
# edges = cv.Canny(gray_blurred,100,200)

# # Parameters to be optimized
# dp_values = [1, 2]  # Inverse ratio of the accumulator resolution to the image resolution
# minDist_values = [10, 20, 30]  # Minimum distance between the centers of detected circles
# param1_values = [30, 40, 50]  # Upper threshold for the edge detector
# param2_values = [20, 30, 40]  # Threshold for circle detection

# best_params = None
# best_circles = None
# best_circle_count = 0

# # Iterate over parameter combinations
# for dp in dp_values:
#     for minDist in minDist_values:
#         for param1 in param1_values:
#             for param2 in param2_values:
#                 circles = cv.HoughCircles(edges,
#                                            cv.HOUGH_GRADIENT,
#                                            dp=dp,
#                                            minDist=minDist,
#                                            param1=param1,
#                                            param2=param2,
#                                            minRadius=0,
#                                            maxRadius=0)
#                 if circles is not None:
#                     circle_count = len(circles[0])
#                     if circle_count > best_circle_count:
#                         best_circle_count = circle_count
#                         best_params = (dp, minDist, param1, param2)
#                         best_circles = circles

# # Draw detected circles on the original image
# if best_circles is not None:
#     best_circles = np.uint16(np.around(best_circles))
#     for i in best_circles[0, :1]:
#         center = (i[0], i[1])
#         # circle center
#         cv.circle(resized_image, center, 1, (0, 100, 100), 3)
#         # circle outline
#         radius = i[2]
#         cv.circle(resized_image, center, radius, (255, 0, 255), 3)

# # Show the original and edge-detected images
# cv.imshow('Original Image', img)
# cv.imshow('Edge Detected Image', edges)

# # Show the original image with detected circles
# cv.imshow('Detected Circles', resized_image)
# cv.imwrite("detected_circles.png", resized_image)

# cv.waitKey(0)
# cv.destroyAllWindows()






import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read in image
img = cv.imread('ps_black.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

new_width = 300
new_height = 300

# Resize the image
resized_image = cv.resize(img, (new_width, new_height))

# Blur using 3 * 3 kernel.
gray_blurred = cv.blur(resized_image, (3, 3)) 

# Dilate and erode a couple of times
kernel = np.ones((5,5), np.uint8)
dilated_image = cv.dilate(gray_blurred, kernel, iterations=2)
eroded_image = cv.erode(dilated_image, kernel, iterations=5)


# Detect edges using Canny
edges = cv.Canny(eroded_image, 100, 200)

# Parameters to be optimized
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
                circles = cv.HoughCircles(edges,
                                           cv.HOUGH_GRADIENT,
                                           dp=dp,
                                           minDist=minDist,
                                           param1=param1,
                                           param2=param2,
                                           minRadius=10,
                                           maxRadius=150)
                if circles is not None:
                    circle_count = len(circles[0])
                    if circle_count > best_circle_count:
                        best_circle_count = circle_count
                        best_params = (dp, minDist, param1, param2)
                        best_circles = circles

# Draw detected circles on the original image
if best_circles is not None:
    best_circles = np.uint16(np.around(best_circles))
    for i in best_circles[0, :1]:
        center = (i[0], i[1])
        # circle center
        cv.circle(resized_image, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(resized_image, center, radius, (255, 0, 255), 3)

# Show the original and edge-detected images
cv.imshow('Original Image', img)
cv.imshow('Edge Detected Image', edges)

# Show the original image with detected circles
cv.imshow('Detected Circles', resized_image)
cv.imwrite("detected_circles.png", resized_image)

cv.waitKey(0)
cv.destroyAllWindows()
