import numpy as np
import cv2
from picamera2 import Picamera2

# Open both cameras
cap_right = Picamera2(1)
cap_left = Picamera2(0)

cap_right.start()
cap_left.start()

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('stereoMapL_x').mat()
Left_Stereo_Map_y = cv_file.getNode('stereoMapL_y').mat()
Right_Stereo_Map_x = cv_file.getNode('stereoMapR_x').mat()
Right_Stereo_Map_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

# Parameters for depth measurement
max_depth = 400  # maximum distance (cm)
min_depth = 20   # minimum distance (cm)
depth_thresh = 100.0  # threshold for safe distance (cm)

# Reading the stored StereoBM parameters
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

# Mouse callback function
def mouse_click(event, x, y, flags, param):
    global depth_map
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"Distance = {depth_map[y, x]:.2f} cm")
        print(f"x = {x:.2f} cm")
        print(f"y = {y:.2f} cm")

cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)
cv2.setMouseCallback('disp', mouse_click)

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

while True:
    imgR = cap_right.capture_array()
    imgL = cap_left.capture_array()
    output_canvas = imgL.copy()

    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # Applying stereo image rectification
    Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Setting the updated parameters before computing disparity map
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

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32)
    disparity = disparity / 2048.0

    # Normalizing the disparity map
    disparity = (disparity / 16.0 - minDisparity) / numDisparities
    depth_map = M / disparity  # for depth in (cm)

    mask_temp = cv2.inRange(depth_map, min_depth, max_depth)
    
    cv2.resizeWindow("disp", 700, 700)
    cv2.imshow("disp", disparity)

    if cv2.waitKey(1) == 27:
        break

cap_right.stop()
cap_left.stop()
cv2.destroyAllWindows()
