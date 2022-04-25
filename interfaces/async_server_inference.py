#!/usr/bin/python

from http import server
import socket
import numpy as np
from infer_utils import preproc as preprocess
import onnxruntime
from infer_utils import multiclass_nms, demo_postprocess
from socket_chunk import chunk_send, chunk_recv
import json
import time
import cv2
import os
from pathlib import Path

def onnx_inference(session, origin_img, size_str, nms):
    input_shape = tuple(map(int, size_str.split(',')))
    img, ratio = preprocess(origin_img, input_shape)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=False)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        return final_boxes, final_scores

    return [], []

if __name__ == "__main__":
    with open("config.json") as f:
        args = json.load(f)
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        ip = "127.0.0.1"
        server_addr = (ip, args["server_port"])
        print("Server address: ", server_addr)

        client_addr = (ip, args["client_port"])
        print("Client address: ", client_addr)

        s.bind(server_addr) 
        
        model_path = None
        findall_onnx = list(Path(".").glob("*.onnx"))
        if len(findall_onnx) > 0:
            model_path = str(findall_onnx[0])
        else:
            model_path = args["model_path"]

        session = onnxruntime.InferenceSession(model_path)
        print("Load ONNX file:", model_path)

        print("Running on {}".format(onnxruntime.get_device()))

        print("Start loop")
        while True:
            print("Waiting for message")
            image_path = chunk_recv(s)

            print("Receive data: {}".format(image_path))

            arr = cv2.imread(image_path)
            
            ## Inference
            start_time = time.time()
            final_boxes, final_scores = onnx_inference(session, arr, args["size"], args["nms"])
            end_time = time.time()
            print("Time elapsed: {} ms".format((end_time - start_time)*1000))

            result_dict = {}
            
            result_dict["boxes"] = final_boxes.tolist()
            result_dict["scores"] = final_scores.tolist()
            print("Results: {}".format(result_dict))

            ## Send back
            print("Send result back to: {}".format(client_addr))
            chunk_send(result_dict, s, client_addr)