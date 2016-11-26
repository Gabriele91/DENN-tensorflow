# Echo server program
import socket
import struct

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 6540              # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            conn.sendall(struct.pack("<ii", 0, 1))
            #conn.sendall(struct.pack("<iicc", 0, 2, chr('a'), chr('b')))