#!/usr/bin/python
from http import server
import socket
from socket_chunk import chunk_send, chunk_recv
import time

def exchange(client_addr, server_addr, im_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(client_addr)

        chunk_send(im_path, s, server_addr)

        result_dict = chunk_recv(s)
        return result_dict

im_path = "D:\\Projects\\VSTech\\FieldDataApril\\output1\\flip1_0Thang-30__9090.jpg"
ip = "127.0.0.1"
server_addr = (ip, 50000)
client_addr = (ip, 50001)

result_dict = exchange(client_addr, server_addr, im_path)
print(result_dict)