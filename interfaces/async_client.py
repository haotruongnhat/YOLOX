#!/usr/bin/python
from http import server
import socket
import cv2
import numpy as np
import pickle
import time
from socket_chunk import chunk_send, chunk_recv
#Turn image into numpy-array

im_path = "/home/haotruong/VSTech/data/output1/0Thang-33.jpg"
img = cv2.imread(im_path)
arr = np.asarray(img)
ip = "127.0.0.1"
server_addr = (ip, 50000)
client_addr = (ip, 50001)

def exchange(client_addr, server_addr, img):
    #Set up socket and stuff 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(client_addr)

    chunk_send(img, s, server_addr)

    result_dict = chunk_recv(s)
    print(result_dict)
    
    s.close()

exchange(client_addr, server_addr, arr)