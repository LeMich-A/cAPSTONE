import cv2
import numpy as np
import math
from picamera2 import Picamera2

# Function to detect and draw circles
def detect_and_draw_circles(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Apply thresholding to obtain binary edges
    threshold = 110
    edges = np.uint8(gradient_magnitude > threshold) * 255
    ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    
    messages = []
    param1_values = [200]
    param2_values = [45]
    dp_values = [1]
    best_params = None
    best_circles = None
    best_circle_count = 0
    minDist_value = img.shape[0] // 4

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
                if circles is not None:
                    circle_count = len(circles[0])
                    if circle_count > best_circle_count:
                        best_circle_count = circle_count
                        best_params = (dp, minDist_value, param1, param2)
                        best_circles = circles
                        messages.append("Yes, object is within.")
                else:
                    messages.append("No object found.")
    
    if best_circles is not None:
        best_circles = np.uint16(np.around(best_circles))
        for circle in best_circles[0, :1]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(img, center, radius, (0, 255, 0), 2)
    
    return img, best_circles, messages

# Initialize cameras
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

# Load stereo rectification maps
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

# Load StereoBM parameters
cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

stereo = cv2.StereoBM_create()

# Process frames
while True:
    imgR = cap_right.capture_array()
    imgL = cap_left.capture_array()
    
    output_canvas = imgL.copy()
    
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    
    Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    
    disparity = stereo.compute(Left_nice, Right_nice)
    disparity = disparity.astype(np.float32)
    disparity = (disparity / 16.0 - minDisparity) / numDisparities
    depth_map = M / (disparity)
    
    mask_temp = cv2.inRange(depth_map, 20, 400)
    depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)
    
    edge_mask = cv2.inRange(depth_map, 20, 40.0)
    edges = cv2.Canny(edge_mask, 50, 150)
    
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    output_canvas = cv2.addWeighted(output_canvas, 0.8, edges_colored, 0.2, 0)
    
    # Detect and draw circles
    image_with_circles, circles, messages = detect_and_draw_circles(output_canvas)
    
    if circles is not None:
        for circle in circles[0, :1]:
            center = (circle[0], circle[1])
            radius = circle[2]
            x, y, w, h = center[0] - radius, center[1] - radius, 2 * radius, 2 * radius
            roi = output_canvas[y:y+h, x:x+w]
            roi_depth = depth_map[y:y+h, x:x+w]
            
            edge_mask_roi = cv2.inRange(roi_depth, 20, 40.0)
            edges_roi = cv2.Canny(edge_mask_roi, 50, 150)
            edges_colored_roi = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
            roi_with_edges = cv2.addWeighted(roi, 0.8, edges_colored_roi, 0.2, 0)
            
            output_canvas[y:y+h, x:x+w] = roi_with_edges
    
    cv2.resizeWindow("disp", 700, 700)
    cv2.imshow("disp", output_canvas)
    
    if cv2.waitKey(1) == 27:
        break

cap_right.stop()
cap_left.stop()
cv2.destroyAllWindows()


cv2.destroyAllWindows()
