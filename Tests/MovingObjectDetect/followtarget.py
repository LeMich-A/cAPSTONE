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

