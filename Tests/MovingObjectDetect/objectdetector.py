import cv2
import numpy as np

def object_detector_and_tracker(frame, last_frame, 
                                gaussian_blur=15, threshold=15,
                                dilation_value=6, dilation_iterations=2, 
                                contour_min_area=1, contour_max_area=80,
                                target_on_contour=True, 
                                target_return_box=False, target_return_size=False,
                                contour_draw=True, contour_line=1, 
                                contour_point=4, contour_pline=-1, contour_color=(0,255,255),
                                contour_box_draw=True, contour_box_line=1,
                                contour_box_point=4, contour_box_pline=-1, contour_box_color=(0,255,0),
                                targets_draw=True, targets_point=4, targets_pline=-1, targets_color=(0,0,255)):
    
    dilation_kernel = np.ones((dilation_value, dilation_value), np.uint8)

    # Frame dimensions
    width, height, depth = np.shape(frame)
    area = width * height

    # Convert frame to grayscale
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    frame2 = cv2.GaussianBlur(frame2, (gaussian_blur, gaussian_blur), 0)

    # Initialize compare frame
    if last_frame is None:
        return [], frame2

    # Compute the absolute difference between current frame and last frame
    frame3 = cv2.absdiff(last_frame, frame2)

    # Apply threshold
    frame3 = cv2.threshold(frame3, threshold, 255, cv2.THRESH_BINARY)[1]

    # Apply dilation
    frame3 = cv2.dilate(frame3, dilation_kernel, iterations=dilation_iterations)

    # Find contours
    contours, hierarchy = cv2.findContours(frame3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variable for the largest target
    largest_target = None

    for c in contours:
        # Compute contour area
        ca = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ba = bw * bh

        # Choose target area based on settings
        ta = ca if target_on_contour else ba

        # Accept targets only in size range
        ta = 100 * ta / area
        if ta < contour_min_area or ta > contour_max_area:
            continue

        # Compute contour centroid
        M = cv2.moments(c)
        if M["m00"]:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(bx + bw / 2)
            cy = int(by + bh / 2)

        # Determine target information
        if not target_return_box and not target_return_size:
            target = (cx, cy)
        elif target_return_size:
            target = (cx, cy, ta)
        else:
            target = (cx, cy, bx, by, bw, bh)

        # Update largest target
        if largest_target is None or (target_on_contour and ta > largest_target[-1]) or (not target_on_contour and ba > largest_target[-1]):
            largest_target = target + (ta if target_on_contour else ba,)

    # Draw contours on the frame
    if contour_draw and largest_target:
        cv2.drawContours(frame, contours, -1, contour_color, contour_line)
        cv2.circle(frame, (largest_target[0], largest_target[1]), contour_point, contour_color, contour_pline)

    # Draw bounding boxes around contours
    if contour_box_draw and largest_target and len(largest_target) == 7:
        bx, by, bw, bh = largest_target[2], largest_target[3], largest_target[4], largest_target[5]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), contour_box_color, contour_box_line)
        cv2.circle(frame, (largest_target[0], largest_target[1]), contour_box_point, contour_box_color, contour_box_pline)

    # Draw target points
    if targets_draw and largest_target:
        cv2.circle(frame, (largest_target[0], largest_target[1]), targets_point, targets_color, targets_pline)

    # Return the single largest target and the current frame to be used as last_frame in next call
    return largest_target[:-1] if largest_target else [], frame2

# Example usage:
# last_frame = None
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     target, last_frame = object_detector_and_tracker(frame, last_frame)
#     # Do something with target
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
