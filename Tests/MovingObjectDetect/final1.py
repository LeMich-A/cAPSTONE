import numpy as np
import cv2

# User Variables
path_length = 30
path_color = (0, 0, 255)
path_line = 1
path_point = 0  # 0 for no points

still_ok = False  # requires still detection to return check as valid
still_points = 15  # last x points
still_max = 10  # max range of points to consider still

# System Variables
path = []
detected = False
frame_count = 0
last_frame = None

# Functions
def reset():
    global detected, last_frame, path
    detected = False
    last_frame = None
    reset_path()

def reset_path():
    global path
    path = []

def check_movement(x, y, frame):
    global path

    path.append((x, y))
    if len(path) > path_length:
        path.pop(0)

    if path_line:
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], path_color, path_line)
    if path_point:
        for i in range(1, len(path)):
            cv2.circle(frame, path[i], path_point, path_color, path_point)

    if len(path) < still_points:
        return -1

    x_points = [point[0] for point in path[-still_points:]]
    y_points = [point[1] for point in path[-still_points:]]

    x_range = max(x_points) - min(x_points)
    y_range = max(y_points) - min(y_points)

    if x_range <= still_max and y_range <= still_max:
        return 0
    return 1

def object_detector_and_tracker(frame):
    global detected, last_frame, frame_count

    frame_count += 1

    # Define parameters
    gaussian_blur = 15
    threshold = 15
    dilation_value = 6
    dilation_iterations = 2
    contour_min_area = 1
    contour_max_area = 80
    contour_color = (0, 0, 255)
    contour_line = 1
    contour_box_draw = True
    contour_box_line = 1
    contour_box_color = (0, 255, 0)
    target_return_box = False
    target_return_size = False
    target_draw = True
    target_point = 4
    target_pline = -1
    target_color = (0, 0, 255)

    dilation_kernel = np.ones((dilation_value, dilation_value), np.uint8)
    width, height, depth = np.shape(frame)
    area = width * height

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.GaussianBlur(frame2, (gaussian_blur, gaussian_blur), 0)

    if last_frame is None:
        last_frame = frame2
        return None

    frame3 = cv2.absdiff(last_frame, frame2)
    last_frame = frame2

    frame3 = cv2.threshold(frame3, threshold, 255, cv2.THRESH_BINARY)[1]
    frame3 = cv2.dilate(frame3, dilation_kernel, iterations=dilation_iterations)

    contours, hierarchy = cv2.findContours(frame3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for c in contours:
        ca = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ba = bw * bh

        ta = ca
        ta = 100 * ta / area
        if ta < contour_min_area or ta > contour_max_area:
            continue

        M = cv2.moments(c)
        if M["m00"]:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(bx + bw / 2)
            cy = int(by + bh / 2)

        if target_return_box:
            targets.append((cx, cy, bx, by, bw, bh))
        elif target_return_size:
            targets.append((cx, cy, ta))
        else:
            targets.append((cx, cy))

    if targets:
        if target_return_box:
            targets = sorted(targets, key=lambda x: x[5] * x[6], reverse=True)[0]
        elif target_return_size:
            targets = sorted(targets, key=lambda x: x[2], reverse=True)[0]
        else:
            targets = sorted(targets, key=lambda x: x[2], reverse=True)[0]
        detected = True
    else:
        detected = False
        return None

    if contour_line:
        cv2.drawContours(frame, contours, -1, contour_color, contour_line)

    if contour_box_draw:
        cv2.rectangle(frame, (targets[2], targets[3]), (targets[2] + targets[4], targets[3] + targets[5]), contour_box_color, contour_box_line)
        cv2.circle(frame, (targets[0], targets[1]), target_point, target_color, target_pline)

    if target_draw:
        cv2.circle(frame, (targets[0], targets[1]), target_point, target_color, target_pline)

    return targets

# Main
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    targets = object_detector_and_tracker(frame)
    if targets:
        x, y = targets[:2]
        check_movement(x, y, frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
