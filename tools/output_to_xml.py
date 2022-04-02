#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:10:01 2018

@author: Caroline Pacheco do E. Silva
"""

import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join
from yolox.data.datasets import STEEL_CLASSES

## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


## converts coco into xml 
def xml_transform(output, img_info, output_file, database_name = "field"):  
    # class_path  = join(root, 'labels')
    # ids = list()
    # l=os.listdir(class_path)
        
    # ids=[x.split('.')[0] for x in l]   

    # annopath = join(root, 'labels', '%s.txt')
    # imgpath = join(root, 'images', '%s.jpg')
    
    # os.makedirs(join(root, 'outputs'), exist_ok=True)
    # outpath = join(root, 'outputs', '%s.xml')
    if "raw_img" in img_info.keys(): 
      height, width, channels = img_info["raw_img"].shape # pega tamanhos e canais das images
    else:
      height, width, channels = img_info["height"], img_info["width"], img_info["channels"]
      
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'
    img_name = img_info["file_name"]

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name
    
    node_source= SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = database_name
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(channels)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    ratio = img_info["ratio"]
    # img = img_info["raw_img"]
    if output is None:
        return None
    output = output.numpy()

    bboxes = np.round(output[:, 0:4] / ratio).astype(np.int)
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]


    for (bbox, class_index, score) in zip(bboxes, cls, scores):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = STEEL_CLASSES[int(class_index)]
        
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'
        
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(bbox[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(bbox[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text =  str(bbox[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(bbox[3])
        xml = tostring(node_root, pretty_print=True)  
        dom = parseString(xml)

    f =  open(output_file, "wb")
    #f = open(os.path.join(outpath, img_id), "w")
    #os.remove(target)
    f.write(xml)
    f.close()     