import json
import base64
import io

import numpy as np
import cv2
from PIL import Image, ImageDraw
from pycocotools import mask as CocoMask

import matplotlib.pyplot as plt 

def load_json(files):
    """Load a json file
    """
    with open(files, "r") as fp:
        file = json.load(fp)
    return file

def compute_colors_for_labels(labels):
    palette = np.int64([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors 

def xywh_xyxy(boxes):
    """xyhw_xyxy ----> we are using h,w and not w, h format .. Please be careful
    """
    bbox = np.zeros(boxes.shape)
    bbox[:, 0] = boxes[:, 0] 
    bbox[:, 1] = boxes[:, 1]  
    bbox[:, 2] = boxes[:, 0] + 1 * boxes[:, 2]
    bbox[:, 3] = boxes[:, 1] + 1 * boxes[:, 3]
    return bbox

def draw_grid(image, bbox, label, outline="white", input_format="xyxy"):
    """draws rectangles on the image given by bbox 
    
    image: PIL image 
    bbox: numpy with each box representing the format defined by "format"
    outline: color of the bbox 
    input_format: "xxyy" or "xyhw": use one of this 
    
    """
    draw = ImageDraw.Draw(image)
    if input_format == "xywh":
        bbox = xywh_xyxy(bbox)
        
    for num,  i in enumerate(bbox):
        x0, y0, x1, y1 = i
        l = label[num]
        draw.rectangle([x0, y0, x1, y1], outline=outline)
        draw.text((x0,y0), l, fill=(255, 0, 0))
    return image

def plot_images(img_list, labels, cols=2):
    fig = plt.figure(figsize=(39, 33))
    rows = int(len(img_list)/cols)
    ax = []
    for i in range(cols*rows):
        ax.append( fig.add_subplot(rows, cols, i+1) )
        ax[-1].set_title(labels[i], fontsize="large")
        plt.imshow(img_list[i])
    plt.show()