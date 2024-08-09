
import cv2
import numpy as np

# Define camera parameters
focal_length = 100  # Hypothetical value, in pixels
sensor_height = 8.8  # Hypothetical value, in mm
object_height = 20  # Example value, in pixels

# Define circular perimeter parameters
perimeter_center = (320, 240)  # Center of the frame
perimeter_radius = 150  # Radius of the circular perimeter, in pixels

# Define threshold distance
threshold_distance = 100  # Arbitrary units

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)  # Adjust thresholds as needed

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process detected contours
    for contour in contours:
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate object position (centroid of bounding rectangle)
        object_position = (x + w // 2, y + h // 2)

        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate distance to the object (similar to previous code)
        object_actual_height = 200  # Example value, in mm
        distance = (focal_length * object_actual_height) / (h * sensor_height)

        # Check if the object is within the perimeter and below the threshold distance
        distance_to_object = np.sqrt((object_position[0] - perimeter_center[0])**2 + (object_position[1] - perimeter_center[1])**2)
        if distance_to_object < perimeter_radius and distance < threshold_distance:
            # Trigger warning
            cv2.putText(frame, "Warning: Object too close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
