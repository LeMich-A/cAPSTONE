import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt

#Read in image
img = cv.imread('test_frame.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

#Detect all edges
edges = cv.Canny(img,100,200)

cv.imshow('original', img)
cv.imshow('edge detected image', edges)
cv.imwrite("edge_detected_image.png", edges)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()




cv.waitKey(0)
cv.destroyAllWindows()