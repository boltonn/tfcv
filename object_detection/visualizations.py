# Not going to use

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_boxes(img, bboxes=None, labels=None, linewidth=1.5, box_color='g', font_color='w', fontsize=16):
    """Plot boxes on an image"""
    
    def transform_bbox(bbox, img_width, img_height):
        #  (ymin, xmin, ymax, xmax) -> (xmin, ymin, w, h)
        xmin = bbox[0]*img_width
        ymin = bbox[1]*img_height
        xmax = bbox[2]*img_width
        ymax = bbox[3]*img_height
        
        w = (xmax - xmin) 
        h = ymax - ymin 
        return [xmin, ymin, w, h]
    
    #fig, ax = plt.subplots(1, figsize=(fig_size, fig_size))
    fig, ax = plt.subplots(1)
    
    if tf.is_tensor(img):
        img = img.numpy()
    ax.imshow(img)
    ax.axis('auto')
        
    if bboxes is not None:
        if tf.is_tensor(bboxes):
            bboxes = bboxes.numpy().tolist()
        else:
            assert isinstance(bboxes, list), "Bounding boxes must be a tensor or list"
            
        for bbox in bboxes:
            bbox = transform_bbox(bbox, img.shape[1], img.shape[0])
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=linewidth, edgecolor=box_color, fill=False)
            ax.add_patch(rect)
            
    if labels is not None:
        for caption in labels:
            ax.set_title(caption, color=font_color, fontsize=font_size)
                
    plt.tight_layout()
    plt.show()