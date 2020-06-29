# Not going to use

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from augmentations import transform_bbox


def plot_boxes(img=None, boxes=None, normalized=True, labels=None, linewidth=1.5, box_color='g', font_color='w', facecolor=None, fontsize=16, title=None):
    """Plot boxes on an image"""
    
    #fig, ax = plt.subplots(1, figsize=(fig_size, fig_size))
    fig, ax = plt.subplots(1)
    
    if title:
        ax.set_title(title, fontsize=20, color=font_color)
    
    if facecolor:
        ax.set_facecolor='b'
    
    if img is not None:
        if tf.is_tensor(img):
            img = img.numpy()
        ax.imshow(img)
    else:
        assert boxes is not None, "Boxes must not be None if img is None"
    ax.axis('auto')
        
    if boxes is not None:
        if tf.is_tensor(boxes):
            boxes = boxes.numpy()
        # somtimes useful to plot anchor boxes even without an image
        else:
            assert isinstance(boxes, (list, np.ndarray)), "Bounding boxes must be a tensor, list, or numpy array"
            assert normalized==False, "normalized must be False if no img is passed"
        if img is None:
            ax.set_xlim([np.min(boxes[:,0])-1, np.max(boxes[:,2])+1])
            ax.set_ylim([np.min(boxes[:,1])-1, np.max(boxes[:,3])+1])
            
        boxes = boxes.tolist()
        for bbox in boxes:
            if normalized:
                bbox = transform_bbox(bbox, img.shape[1], img.shape[0], normalized=True)
            else:
                bbox = transform_bbox(bbox, normalized=False)
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=linewidth, edgecolor=box_color, fill=False)
            ax.add_patch(rect)
            
    if labels is not None:
        for caption in labels:
            ax.set_title(caption, color=font_color, fontsize=font_size)
                
    plt.tight_layout()
    plt.show()