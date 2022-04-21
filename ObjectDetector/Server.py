import socket
import time
import math

# TO USE:

# IMPORT socket

# USE CREATE SERVER AND SAVE IT TO A SERVER VARIABLE (THIS WILL HOLD THE ACTIVE SOCKET)
#Example : from Server import createServer, checkState
# This will give you access to the createServer and checkState functions. createServer returns a socket, checkstate returns a string
# PERIODICALLY RUN CHECKSTATE 
def createServer(port):
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(('',port))
    sock.setblocking(0)
    return sock

def checkState(current_state,socket):
    try:
        data = socket.recv(1024).decode()
        if 'm' in data:
            return data[1:]
    except KeyboardInterrupt:
        socket.close()
    except:
        return current_state

def main():
    nano_socket = createServer(8008)
    visionState = 't'
    diff = 0
    while True:
        current_time = time.time()
        visionState = checkState(visionState,nano_socket)
        diff = max(time.time() - current_time,diff)
        print('Vision State : {} Max Time taken : {}'.format(visionState,diff))

    

if __name__ == '__main__':
    main()