#!/usr/bin/python

import socket
import numpy as np
import pickle
from tqdm import tqdm
import time

import cv2

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "127.0.0.1"
server_addr = (ip, 50000)
client_addr = (ip, 50001)
s.bind(server_addr) 

def exchange(client_addr):
    chunks = []
    buf = 4096
    Dlen = int(pickle.loads(s.recv(buf)))
    pkl_str = bytearray()
    for i in tqdm(range(0,int(int(Dlen)/buf)+1)):
        data = s.recv(buf)
        pkl_str += data

    arr = pickle.loads(pkl_str)
    
    ## Inference
    result_dict = {}
    
    result_dict["counter"] = 10
    
    ## Send back
    msg = pickle.dumps(result_dict)
    s.sendto(msg, client_addr)

exchange(client_addr)

s.close()