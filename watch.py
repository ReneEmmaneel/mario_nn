import sys
import time
import logging
import socket
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        pass

    def on_created(self, event):
        print(event)

    def on_changed(self, event):
        print(event)

def start_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 2222
    server.bind((socket.gethostname(), port))
    print("Hostname: %s Port: %d" % (socket.gethostname(), port))
    server.listen(1)

    while True:
        print(f"Listening for connection on port {port}...")
        (clientsocket, address) = server.accept()
        print("Received client at %s" % (address,))
        
        try:
            clientsocket.send("First message\n".encode())
        
            while True:
                receive = clientsocket.recv(2048).decode('ascii')
                print(receive)
                clientsocket.send("Other messages\n".encode())
        except Exception as e:
            print("Exception occurred. Closing connection.")
            print(e)
            clientsocket.send(b"close")
            clientsocket.close()
    print('Done!')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    """
    path = 'data/'
    event_handler = FileChangeHandler()

    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()"""

    start_socket()