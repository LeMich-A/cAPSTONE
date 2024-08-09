import numpy as np
import cv2 as cv
import glob
import os
from PIL import Image

from picamera2 import Picamera2



picam2 = Picamera2(1)
picam2.start()
while True:
   image = picam2.capture_array()
   image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
   cv.imshow("Frame", image)
   if(cv.waitKey(1) == ord("q")):
      cv.imwrite("test_frame.png", image)
      break

cv.destroyAllWindows()


## Initilise ##

#Begin by initializing the Chess Board Size that is going to be assesed
#As well as the frame size and square size

  
che = (11,8) # Checker-board INNER corners
Sq = 23 # Square Dimensions
images = glob.glob('Pictures/*.png') #Folder to pull images from
savedir="Calibration_Data/" # Folder to store image data

## Read in images ##

#Pull the thermal Image

for image in images:
    Calimg = cv.imread(image)
    Calimg = cv.resize(Calimg, (500,500))
       
   

## Image Processing ##
    
    Gray = cv.cvtColor(Calimg, cv.COLOR_BGR2GRAY)


## Checker Board Finder ##


#Using the FindChessBoard() function, we can locate the corner location
#for each black square in the image


    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((che[0]*che[1],3), np.float32)
    index = 0
    for i in range(che[0]):
        for j in range(che[1]):
            objp[index][0] = i * Sq
            objp[index][1] = j * Sq
            index += 1
        
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(Gray, che, cv.CALIB_CB_NORMALIZE_IMAGE)


    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
                    
        corners2 = cv.cornerSubPix(Gray,corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners2) 

        # print("Image Points Received")
                
        cv.drawChessboardCorners(Calimg, che, corners2, ret)


    else:
            
        print("Checker Board NOT found!")


    
    cv.imshow('imgTherm',Calimg)
    cv.waitKey(0)
    cv.destroyAllWindows()


## Camera Calibration ##

#Conduct the Camera Calibration in order to obtain the Camera Intrinsic Parametrs

h,  w = Gray.shape[:2]

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

newCam_mtx, roi=cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


#Calculate Camera Extrinsic Parameters

rvec, tvec , _ = cv.solvePnP(objp, corners2, newCam_mtx, dist)

dst = cv.Rodrigues(np.array(rvecs))


#Now check for Reporjection Error

rvecs = np.array([0, 0, 0], dtype=np.float32) 
tvecs = np.array([0, 0, 100], dtype=np.float32) 

imgpoints = np.array(imgpoints)[0,:,:,:]

imgpoints2,_ = cv.projectPoints(objp, rvecs,tvecs, newCam_mtx, dist)

mean_error = 0
for i in range(len(objpoints)):
    error = cv.norm(corners2[i], imgpoints2[i], cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "\nTotal Reprojection error: {}".format(mean_error/len(objpoints)) )



## Save Calibration data ##

#Save all of the intrinsic Parameters as Numpy files
np.save(savedir+'Cam_Mtx.npy', cameraMatrix)
np.save(savedir+'Dist.npy', dist)
np.save(savedir+'Rot.npy', rvecs)
np.save(savedir+'Trans.npy', tvecs)
np.save(savedir+'roi.npy', roi)
np.save(savedir+'newCam_mtx.npy', newCam_mtx)

