import socket
import time

def main():
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    address = '10.203.157.56'
    sock.connect((address,8008))
    while True:
        sock.sendall(bytes('mt','utf-8'))
        time.sleep(10)
        sock.sendall(bytes('mv','utf-8'))
        time.sleep(10)
        sock.sendall(bytes('mo','utf-8'))
        time.sleep(10)
        sock.sendall(bytes('mv','utf-8'))
        time.sleep(10)
    

if __name__ == '__main__':
    main()