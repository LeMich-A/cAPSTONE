import cv2
import libcamera

# Initialize libcamera
libcamera.start()

# Create a camera configuration
config = libcamera.CameraConfiguration()
config.add_options({
    'stream.format': 'YUV420',
    'stream.size': '640x480',
})

# Create a camera instance
camera = libcamera.Camera(config)

# Start capturing frames
camera.start()

while True:
    frame = camera.get_frame()
    if frame:
        # Convert frame to OpenCV format (BGR)
        cv_frame = cv2.cvtColor(frame.data, cv2.COLOR_YUV2BGR_I420)

        # Process the frame (e.g., display, analyze, etc.)
        cv2.imshow('libcamera Feed', cv_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
camera.stop()
cv2.destroyAllWindows()
