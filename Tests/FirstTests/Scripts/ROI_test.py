import cv2
import numpy as np

def detect_circles(image, min_radius, max_radius, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Initialize accumulator array to store circle votes
    height, width = edges.shape
    accumulator = np.zeros((height, width, max_radius - min_radius + 1), dtype=np.uint8)
    
    # Iterate over each pixel in the edge image
    for y in range(height):
        for x in range(width):
            # If pixel is an edge pixel
            if edges[y, x] > 0:
                # Vote for possible circle centers with different radii
                for r in range(min_radius, max_radius + 1):
                    for theta in range(0, 360):
                        a = int(x - r * np.cos(np.radians(theta)))
                        b = int(y - r * np.sin(np.radians(theta)))
                        if a >= 0 and a < width and b >= 0 and b < height:
                            accumulator[b, a, r - min_radius] += 1
    
    # Find circles with votes above threshold
    circles = []
    for r in range(max_radius - min_radius + 1):
        for y in range(height):
            for x in range(width):
                if accumulator[y, x, r] >= threshold:
                    circles.append((x, y, r + min_radius))
    
    return circles

# Read the image
image = cv2.imread('ps_black.jpg')

# Define parameters
min_radius = 10
max_radius = 100
threshold = 100

# Detect circles
detected_circles = detect_circles(image, min_radius, max_radius, threshold)

# Draw detected circles on the image
for (x, y, r) in detected_circles:
    cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

# Display the image with detected circles
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
