import socket
import threading

# Server (Raspberry Pi) IP address and port
server_ip = "192.168.2.10"
port = 65430

def receive_messages(client_socket):
    try:
        while True:
            data = client_socket.recv(1024).decode('ascii')
            if not data:
                break
            print("Received:", data)
    except Exception as e:
        print("Error:", e)

def listen_for_exit(client_socket):
    while True:
        key = input()  # Waits for user input
        if key.lower() == 'm':
            print("Exit command received. Closing connection.")
            client_socket.close()
            break

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, port))
        print("Connected to server at", server_ip, ":", port)
        
        # Start a thread to receive messages
        threading.Thread(target=receive_messages, args=(client_socket,), daemon=True).start()
        
        # Listen for exit command
        listen_for_exit(client_socket)
        
    except Exception as e:
        print("Error:", e)
    finally:
        client_socket.close()
        print("Connection closed")

if __name__ == "__main__":
    main()

