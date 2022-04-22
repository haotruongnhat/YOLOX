#!/usr/bin/python
import socket
import cv2
import numpy as np
import pickle
import time

#Turn image into numpy-array

im_path = "/home/haotruong/VSTech/data/output1/0Thang-33.jpg"
img = cv2.imread(im_path)
arr = np.asarray(img)
ip = "127.0.0.1"
server_addr = (ip, 50000)
client_addr = (ip, 50001)

def exchange(client_addr, server_addr, img):
    #Set up socket and stuff 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(client_addr)
        s.connect(server_addr)
        
        try:
            buf = 4096
            i = 0
            j = buf
            msg = pickle.dumps(img)

            packet_len = pickle.dumps(len(msg))
            s.sendto(packet_len, server_addr)

            # Send pickle
            while(i<len(msg)):
                
                if j>(len(msg)-1):
                    j=(len(msg))
                    
                ###send pickle chunks
                s.sendto(msg[i:j], server_addr)
                i += buf
                j += buf
            
            data = s.recv(buf)
            result_dict = pickle.loads(data)
            print(result_dict)
        except Exception as e:
            print(e)
    
exchange(client_addr, server_addr, arr)