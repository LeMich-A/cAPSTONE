import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the pre-trained model
model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
model_path = f'http://download.tensorflow.org/models/object_detection/{model_name}.tar.gz'
model_file = f'{model_name}.tar.gz'
model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=model_path,
    untar=True
)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_ckpt = f'{model_dir}/frozen_inference_graph.pb'

# Path to the label map
label_map_path = 'path_to_your_label_map/label_map.pbtxt'
num_classes = 90  # Change this to the number of classes in your model

# Load label map
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the frozen TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Main program loop with object detection and depth estimation using stereo vision
with detection_graph.as_default():
    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while True:
            frame_right = cap_right.capture_array()
            frame_left = cap_left.capture_array()

            ################## CALIBRATION #########################################################
            frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
            ########################################################################################

            # If cannot catch any frame, break
            if frame_right is None or frame_left is None:                    
                break

            else:
                start = time.time()
                
                # Convert the BGR image to RGB
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

                # Perform object detection
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_expanded_right = np.expand_dims(frame_right, axis=0)
                image_expanded_left = np.expand_dims(frame_left, axis=0)

                # Perform the actual detection
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded_right})

                # Visualize the results
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame_right,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Convert the RGB image to BGR
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

                ################## CALCULATING DEPTH #########################################################
                # Your existing depth estimation code goes here
                # You can use the bounding box information from object detection for depth estimation
                ################################################################################################

                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime
                cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   

                # Show the frames
                cv2.imshow("frame right", frame_right) 
                cv2.imshow("frame left", frame_left)

                # Hit "q" to close the window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
