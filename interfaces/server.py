#!/usr/bin/python

import socket
import numpy as np
import pickle
from tqdm import tqdm
import time

import cv2

def exchange(conn):
    chunks = []
    buf = 4096
    
    try:
        Dlen = int(pickle.loads(conn.recv(buf)))
        print("Ready to receive {} bytes".format(Dlen))
        
        pkl_str = bytearray()
        for i in tqdm(range(0,int(int(Dlen)/buf)+1)):
            data = conn.recv(buf)
            pkl_str += data

        arr = pickle.loads(pkl_str)
        print("Received image with size {}".format(str(arr.shape)))

        ## Inference
        result_dict = {}
        
        result_dict["counter"] = 10
        
        ## Send back
        msg = pickle.dumps(result_dict)
        conn.sendto(msg, client_addr)
    except Exception as e:
        print(e)
        

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip = "127.0.0.1"
server_addr = (ip, 50000)
client_addr = (ip, 50001)
s.bind(server_addr)
s.listen()
conn, addr = s.accept()

print(addr)
exchange(conn)

s.close()