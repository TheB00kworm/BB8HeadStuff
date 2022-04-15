import socket
import threading
import time
import sys

def get_state(current_state,socket):
    try:
        if 'm' in socket.recv(1024,socket.MSG_PEEK):
            print('gotteem')
            sys.exit()
            return socket.recv(1024)[1]
    except:
        return current_state

def create_server(port):
    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    s.setblocking(0)
    s.bind(('',port))
    return s

def main():

    nano_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    nano_socket.setblocking(0)
    nano_port = 321
    nano_socket.bind(('',nano_port))

    print('Socket created and bound to Port {}',nano_port)
    visionState = 't'
    while True:
        print(visionState)
        current_time = time.time()
        get_state(visionState,nano_socket)
        print(time.time() - current_time)
        time.sleep(1)


if __name__ == '__main__':
    main()