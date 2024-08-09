import cv2
import numpy as np

# ------------------------------
# User Variables
# ------------------------------

# target method
contour = False

# contour setup
contour_min_area = 1  # percent of frame area
contour_max_area = 80 # percent of frame area
contour_color    = (0,0,255)
contour_line     = 1

# target select
target_return_box  = False # True = return (x,y,bx,by,bw,bh), else check target_return_size
target_return_size = False # True = return (x,y,percent_frame_size), else just (x,y)

# display contour box
contour_box_draw  = True
contour_box_line  = 1 # border width
contour_box_color = (0,255,0) # BGR color

# display target
target_draw  = True
target_point = 4 # centroid radius
target_pline = -1 # border width
target_color = (0,0,255) # BGR color

# ------------------------------
# System Variables
# ------------------------------

# status
detected = False

# counts and amounts
frame_count = 0

# current frame
last_frame = None

# ------------------------------
# Functions
# ------------------------------

def reset():
    global detected, last_frame
    detected = False
    last_frame = None

def next(frame):
    global detected, last_frame

    # frame dimensions
    width, height, depth = np.shape(frame)
    area = width * height

    # grayscale
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur
    frame2 = cv2.GaussianBlur(frame2, (15, 15), 0)

    # delta
    if last_frame is None:
        last_frame = frame2
        return None

    frame3 = cv2.absdiff(last_frame, frame2)
    last_frame = frame2

    # threshold
    frame3 = cv2.threshold(frame3, 15, 255, cv2.THRESH_BINARY)[1]

    # dilation
    frame3 = cv2.dilate(frame3, np.ones((21, 21), np.uint8), iterations=2)

    # get contours
    contours, hierarchy = cv2.findContours(frame3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # targets
    targets = []
    for c in contours:
        # basic contour data
        ca = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ba = bw * bh

        # accept targets only in size range
        ta = ca if contour else ba
        ta = 100 * ta / area
        if ta < contour_min_area or ta > contour_max_area:
            continue

        # contour centroid
        M = cv2.moments(c)
        if M["m00"]:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(bx + bw / 2)
            cy = int(by + bh / 2)

        # target details
        if target_return_box:
            targets.append((cx, cy, bx, by, bw, bh))
        elif target_return_size:
            targets.append((cx, cy, ta))
        else:
            targets.append((cx, cy))

    # get largest target
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

    # display contours
    if contour_line:
        cv2.drawContours(frame, contours, -1, contour_color, contour_line)

    # display target box
    if contour_box_draw:
        cv2.rectangle(frame, (targets[2], targets[3]), (targets[2] + targets[4], targets[3] + targets[5]), contour_box_color, contour_box_line)
        cv2.circle(frame, (targets[0], targets[1]), target_point, target_color, target_pline)

    # display targets
    if target_draw:
        cv2.circle(frame, (targets[0], targets[1]), target_point, target_color, target_pline)

    return targets

# Example usage:
# frame = ... (capture a frame from your video source)
# result = next(frame)
# if detected:
#     print('TARGET AT:', result)
