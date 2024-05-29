
import cv2
import numpy as np
import math


image_path = "Michting.jpg"  # Image path

def perimeter_check_function(image_path):
    # Load the image

    
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply mean filter
    kernelMean = np.ones((31,31), np.float32) / 961
    imgMean = cv2.filter2D(gray, -1, kernelMean)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(imgMean, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imgMean, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Apply thresholding to obtain binary edges
    threshold = 11
    edges = np.uint8(gradient_magnitude > threshold) * 255

    # Convert edges to binary image
    ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

    # Perform dilation and erosion to enhance edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(binary_edges, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Hough Circle parameters
    param1_values = [200]  # Upper threshold for the edge detector
    param2_values = [10]   # Threshold for circle detection
    dp_values = [1]        # Inverse ratio of the accumulator resolution to the image resolution

    # Initialize best parameters and circles
    best_params = None
    best_circles = None
    best_circle_count = 0

    # Calculate minDist value
    minDist_value = img.shape[0] // 4

    # Iterate over parameter combinations
    for dp in dp_values:
        for param1 in param1_values:
            for param2 in param2_values:
                circles = cv2.HoughCircles(eroded_image,
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
                        best_params = (dp, minDist_value, param1, param2)
                        best_circles = circles

    # Draw detected circles on the original image
    if best_circles is not None:
        best_circles = np.uint16(np.around(best_circles))
        for i in best_circles[0, :1]:
            center = (i[0], i[1])
            radius = i[2]
            x, y = center[0] - radius, center[1] - radius
            w, h = 2 * radius, 2 * radius
            roi = img[y:y+h, x:x+w].copy()  # Extract ROI from the image

            # Create a mask for the region outside of the ROI
            mask = np.zeros_like(img)
            cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)  # Draw circle in the mask

            # Set pixels outside of the ROI to a dark color (e.g., black)
            img[mask[:, :, 0] == 0] = [0, 0, 0]

            # Create a copy of the resized image for drawing offset circles
            offset_image = roi.copy()

            # Offset circles based on the detected circles
            for j in best_circles[0, :1]:
                x_offset, y_offset, r_offset = j[0] - x, j[1] - y, j[2]
                # Calculate inward offset
                inward_offset = 100  # Define your inward offset here
                offset = int(inward_offset / math.sqrt(2))
                # Adjust radius with inward offset
                r_offset -= offset
                # Draw inner circle
                cv2.circle(offset_image, (x_offset, y_offset), r_offset, (255, 0, 0), 10)
                # Calculate perimeter
                perimeter = 2 * math.pi * r_offset

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_edges = cv2.Canny(roi_gray, 100, 200)

            # Initialize flag for edge within perimeter
            edge_within_perimeter = False

            # Check for overlapping edges in ROI and offset circle
            for row in range(roi_edges.shape[0]):
                for col in range(roi_edges.shape[1]):
                    if roi_edges[row, col] == 255:
                        # Calculate distance between edge point and circle center
                        dist = math.sqrt((x_offset - col) ** 2 + (y_offset - row) ** 2)
                        # Check if distance is less than offset circle radius
                        if dist <= r_offset:
                            # Check if distance is less than perimeter
                            if dist <= perimeter:
                                # Set flag indicating edge is within perimeter
                                edge_within_perimeter = True
                            else:
                                # Print a warning indicating overlap beyond perimeter
                                print("Warning: Overlapping beyond perimeter")
                            # Draw a red dot indicating overlap
                            cv2.circle(offset_image, (col, row), 1, (0, 0, 255), -1)

            # Print message based on edge within perimeter flag
            if edge_within_perimeter:
                msg1 = "Good: Edge within the perimeter"
               
            

            else:
                msg2 = "Warning: No edge within the perimeter"
                

    

    # # Display images
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Offset Image', offset_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return msg1 , msg2 


msg1, msg2 = perimeter_check_function(image_path)
print(msg1)
print(msg2)






# import cv2
# import numpy as np
# import math

# # Load the image
# img = cv2.imread('Michting.jpg')

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply mean filter
# kernelMean = np.ones((31,31), np.float32) / 961
# imgMean = cv2.filter2D(gray, -1, kernelMean)

# # Apply Sobel edge detection
# sobel_x = cv2.Sobel(imgMean, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(imgMean, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

# # Apply thresholding to obtain binary edges
# threshold = 11
# edges = np.uint8(gradient_magnitude > threshold) * 255

# # Convert edges to binary image
# ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)

# # Perform dilation and erosion to enhance edges
# kernel = np.ones((5, 5), np.uint8)
# dilated_image = cv2.dilate(binary_edges, kernel, iterations=1)
# eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

# # Hough Circle parameters
# param1_values = [200]  # Upper threshold for the edge detector
# param2_values = [10]   # Threshold for circle detection
# dp_values = [1]        # Inverse ratio of the accumulator resolution to the image resolution

# # Initialize best parameters and circles
# best_params = None
# best_circles = None
# best_circle_count = 0

# # Calculate minDist value
# minDist_value = img.shape[0] // 4

# # Iterate over parameter combinations
# for dp in dp_values:
#     for param1 in param1_values:
#         for param2 in param2_values:
#             circles = cv2.HoughCircles(eroded_image,
#                                        cv2.HOUGH_GRADIENT,
#                                        dp=dp,
#                                        minDist=minDist_value,
#                                        param1=param1,
#                                        param2=param2,
#                                        minRadius=0,
#                                        maxRadius=0)
#             if circles is not None:
#                 circle_count = len(circles[0])
#                 if circle_count > best_circle_count:
#                     best_circle_count = circle_count
#                     best_params = (dp, minDist_value, param1, param2)
#                     best_circles = circles

# # Draw detected circles on the original image
# if best_circles is not None:
#     best_circles = np.uint16(np.around(best_circles))
#     for i in best_circles[0, :1]:
#         center = (i[0], i[1])
#         radius = i[2]
#         x, y = center[0] - radius, center[1] - radius
#         w, h = 2 * radius, 2 * radius
#         roi = img[y:y+h, x:x+w].copy()  # Extract ROI from the image

#         # Create a mask for the region outside of the ROI
#         mask = np.zeros_like(img)
#         cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)  # Draw circle in the mask

#         # Set pixels outside of the ROI to a dark color (e.g., black)
#         img[mask[:, :, 0] == 0] = [0, 0, 0]

#         # Create a copy of the resized image for drawing offset circles
#         offset_image = roi.copy()

#         # Offset circles based on the detected circles
#         for j in best_circles[0, :1]:
#             x_offset, y_offset, r_offset = j[0] - x, j[1] - y, j[2]
#             # Calculate inward offset
#             inward_offset = 100  # Define your inward offset here
#             offset = int(inward_offset / math.sqrt(2))
#             # Adjust radius with inward offset
#             r_offset -= offset
#             # Draw inner circle
#             cv2.circle(offset_image, (x_offset, y_offset), r_offset, (255, 0, 0), 10)
#             # Calculate perimeter
#             perimeter = 2 * math.pi * r_offset

#         roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         roi_edges = cv2.Canny(roi_gray, 100, 200)

#         # Initialize flag for edge within perimeter
#         edge_within_perimeter = False

#         # Check for overlapping edges in ROI and offset circle
#         for row in range(roi_edges.shape[0]):
#             for col in range(roi_edges.shape[1]):
#                 if roi_edges[row, col] == 255:
#                     # Calculate distance between edge point and circle center
#                     dist = math.sqrt((x_offset - col) ** 2 + (y_offset - row) ** 2)
#                     # Check if distance is less than offset circle radius
#                     if dist <= r_offset:
#                         # Check if distance is less than perimeter
#                         if dist <= perimeter:
#                             # Set flag indicating edge is within perimeter
#                             edge_within_perimeter = True
#                         else:
#                             # Print a warning indicating overlap beyond perimeter
#                             print("Warning: Overlapping beyond perimeter")
#                         # Draw a red dot indicating overlap
#                         cv2.circle(offset_image, (col, row), 1, (0, 0, 255), -1)

#         # Print message based on edge within perimeter flag
#         if edge_within_perimeter:
#             print("Good: Edge within the perimeter")
#         else:
#             print("Warning: No edge within the perimeter")

# # Display images
# cv2.imshow('Original Image', img)
# cv2.imshow('Offset Image', offset_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
