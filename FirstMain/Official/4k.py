# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile

# # Import your circle detection function
# from Perimiterdetect import detect_and_draw_circles

# def start_stream(picam2):
#     picam2.start()
#     while True:
#         # Capture an image array
#         image = picam2.capture_array()
        
#         # Save the image array as a temporary file
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#             temp_file.write(cv2.imencode('.jpg', image)[1])

#         # Call your circle detection function on the temporary file
#         original_img, messages = detect_and_draw_circles(temp_file.name)

#         # Draw messages on the image
#         for message in messages:
#             cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         # Display the processed images
#         cv2.imshow('Original Image', original_img)

#         # Exit the loop if 'm' is pressed
#         key = cv2.waitKey(1)
#         if key == ord('m'):
#             break

#     # Stop the camera before exiting
#     picam2.stop()

#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Initialize Picamera2
#     picam2 = Picamera2()

#     # Start the stream
#     start_stream(picam2)




# ###### SERVER CODE ################################
# from picamera2 import Picamera2
# import cv2
# import numpy as np
# import tempfile
# import socket

# # Import your circle detection function
# from Perimiterdetect import detect_and_draw_circles

# def start_stream(picam2, connection):
#     picam2.start()
#     while True:
#         # Capture an image array
#         image = picam2.capture_array()
        
#         # Save the image array as a temporary file
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#             temp_file.write(cv2.imencode('.jpg', image)[1])

#         # Call your circle detection function on the temporary file
#         original_img, messages = detect_and_draw_circles(temp_file.name)

#         # Draw messages on the image and send them over the network
#         for message in messages:
#             cv2.putText(original_img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             connection.sendall(message.encode('ascii'))  # Send message to the client

#         # Display the processed images
#         cv2.imshow('Original Image', original_img)

#         # Exit the loop if 'm' is pressed
#         key = cv2.waitKey(1)
#         if key == ord('m'):
#             break

#     # Stop the camera before exiting
#     picam2.stop()

#     # Close OpenCV windows
#     cv2.destroyAllWindows()

# def setup_server(ip_address, port):
#     # Create a socket object
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     # Bind the socket to the address and port
#     server.bind((ip_address, port))
#     server.listen(1)
#     print("Server listening on", ip_address, ":", port)

#     # Accept a new connection
#     connection, address = server.accept()
#     print("Connection from", address)
    
#     return connection

# if __name__ == "__main__":
#     # IP address and port to listen on
#     ip_address = "192.168.2.11"
#     port = 25565

#     # Setup server and accept connection
#     connection = setup_server(ip_address, port)

#     # Initialize Picamera2
#     picam2 = Picamera2()

#     try:
#         # Start the stream
#         start_stream(picam2, connection)
#     finally:
#         # Close the connection
#         connection.close()
#         print("Connection closed")
