import numpy as np
import tensorflow as tf

#TO-DO: transformations (translation, rotation, shear, etc.)

def unnormalize_boxes(boxes, img_width, img_height):
    """Unnormalize anchor or bounding boxes by image height and width"""
    x1 = boxes[:, 0]*img_width
    y1 = boxes[:, 1]*img_height
    x2 = boxes[:, 2]*img_width
    y2 = boxes[:, 3]*img_height
    if tf.is_tensor(boxes):
        boxes = tf.stack([x1, y1, x2, y2], axis=1)
    else:
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes
        


#not sure if there is a better place for this
def transform_bbox(bbox, img_width=None, img_height=None, normalized=False):
    """Transform bbox from (ymin, xmin, ymax, xmax) -> (xmin, ymin, w, h)
        * xmin and ymin will be unormalized    
    Args:
        bbox (iter): Bounding box of form (x1, y1, x2, y2)
        img_width (int): Image width
        img_height (int): Image height
        normalized (bool): If input is normalized by image resolution
    Returns:
        bbox (list): Bounding box of form (xmin, ymin, w, h)
    """
    #  (ymin, xmin, ymax, xmax) -> (xmin, ymin, w, h)
    xmin, ymin, xmax, ymax = bbox

    if normalized:
        xmin *= img_width
        ymin *= img_height
        xmax *= img_width
        ymax *= img_height
        
    w = abs(xmax - xmin) 
    h = abs(ymax - ymin)
    return [xmin, ymin, w, h]


def resize(img, bboxes, height=640, width=640):
    """Resize img and normalized bounding boxes"""
    orig_height, orig_width, _ = img.shape
    img = tf.image.resize(img, (height, width), method='bilinear', preserve_aspect_ratio=True)
    #boxes dont need adjustments when normalized
    return img, bboxes
 
    
def filter_boxes(bboxes, min_x=0, max_x=1, min_y=0, max_y=1):
    """Filter out bounding boxes"""
    #TO-DO: Nobody does this but it seems we should crop to an area where there is at least one object
    #if it fails any of the conditions
    invalid_indices=[]
    for condition in [tf.where(tf.math.less_equal(bboxes[:, 2],  bboxes[:, 0])),
                      tf.where(tf.math.less_equal(bboxes[:, 3],  bboxes[:, 1])),
                      tf.where(tf.math.less(bboxes[:, 0], min_x)),
                      tf.where(tf.math.less(bboxes[:, 2], min_y)),
                      tf.where(tf.math.less(bboxes[:, 1], min_x)),
                      tf.where(tf.math.less(bboxes[:, 3], min_y)),
                      tf.where(tf.math.greater(bboxes[:, 1],  max_x)),
                      tf.where(tf.math.greater(bboxes[:, 3],  max_y)),
                      tf.where(tf.math.greater(bboxes[:, 0],  max_x)),
                      tf.where(tf.math.greater(bboxes[:, 2],  max_y))]:
        #make sure not empty since will break concat
        if condition.get_shape()[0]!=0:
            invalid_indices.append(condition)
            
    if invalid_indices:
        invalid_indices = tf.concat(invalid_indices, axis=0)
        invalid_indices = tf.keras.backend.flatten(invalid_indices)
        invalid_indices = tf.sort(invalid_indices, axis=0, direction='ASCENDING')
        invalid_indices, _ = tf.unique(invalid_indices)
        boxes_shape = tf.shape(bboxes)
        n_boxes = boxes_shape[0]
        indices = tf.range(n_boxes, delta=1, dtype=tf.int32)
#        indices = tf.cast([idx for idx in range(0, n_boxes)], dtype=tf.int32)
        
        #hack way for negative indexing (since tensorflow doesnt support complex indexing like numpy)
        updates = tf.math.negative(tf.ones(tf.shape(invalid_indices)[0], dtype=tf.int32)) #vector of -1s
        #updates = tf.cast([-1 for _ in range(0, len(invalid_indices))], dtype=tf.int32)
        invalid_indices = tf.expand_dims(invalid_indices, axis=-1)
        #replace indices w/ -1
        indices = tf.tensor_scatter_nd_update(tensor=indices, indices=invalid_indices, updates=updates)
        bboxes = tf.gather(bboxes, tf.keras.backend.flatten(tf.where(indices>=0)))
    return bboxes



def clip_boxes(bboxes, clip_min=0, clip_max=1):
    """Clip normalized boxes within dimensions of an image"""
    # RetinaFace keeps this as a bounding box; if not then use filter_bboxes
    x1 = tf.clip_by_value(bboxes[:, 0], clip_value_min=clip_min, clip_value_max=clip_max)
    y1 = tf.clip_by_value(bboxes[:, 1], clip_value_min=clip_min, clip_value_max=clip_max)
    x2 = tf.clip_by_value(bboxes[:, 2], clip_value_min=clip_min, clip_value_max=clip_max)
    y2 = tf.clip_by_value(bboxes[:, 3], clip_value_min=clip_min, clip_value_max=clip_max)
    
    return tf.keras.backend.stack([x1, y1, x2, y2], axis=1)



def random_crop(img, bboxes, lower_bound=0.3, upper_bound=1., clip=True, min_boxes=0, bbox_normalized=True):
    # TO-DO: want to make sure it at least have one face ?
    """Random Crop of image proportional to smallest side"""
    
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    img_height = img_shape[0]
    img_width = img_shape[1]
    
    #unnormalize bboxes
    if bbox_normalized:
        bboxes = unnormalize_boxes(bboxes, img_width=img_width, img_height=img_height)
        
    #take random proportion from the smallest side
    prop_crop_size = tf.random.uniform([1], minval=lower_bound, maxval=upper_bound, dtype=tf.float32)
    #prop_crop_size = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
    # get smallest side
    min_size = tf.cast(tf.reduce_min([img_height, img_width]), dtype=tf.float32) # not including batch_size, -1 ignoring channels
    crop_size = tf.cast(prop_crop_size*min_size, dtype=tf.int32)
    
    if min_boxes>0:
        max_val_x = tf.squeeze(tf.cast(img_width, dtype=tf.int32) - crop_size + 1)
        max_val_y = tf.squeeze(tf.cast(img_height, dtype=tf.int32) - crop_size + 1)
        
        #sometime this is impossibel to crop_size needs to be increased
        max_starting_point_x = tf.cast(tf.sort(bboxes[:, 0], direction='DESCENDING')[min_boxes-1], dtype=tf.int32)
        max_starting_point_y = tf.cast(tf.sort(bboxes[:, 1], direction='DESCENDING')[min_boxes-1], dtype=tf.int32)
        min_starting_point_x = tf.cast(tf.sort(bboxes[:, 2], direction='ASCENDING')[min_boxes-1], dtype=tf.int32)
        min_starting_point_y = tf.cast(tf.sort(bboxes[:, 3], direction='ASCENDING')[min_boxes-1], dtype=tf.int32)

        min_val_x = tf.squeeze(min_starting_point_x - crop_size)
        min_val_y = tf.squeeze(min_starting_point_y - crop_size)

        min_val_x = tf.reduce_max([0, min_val_x])
        min_val_y = tf.reduce_max([0, min_val_y])
        max_val_x = tf.reduce_min([max_starting_point_x, max_val_x])
        max_val_y = tf.reduce_min([max_starting_point_y, max_val_y])
        
        #that crop_size is impossible while keeping a face
        if tf.less_equal(max_val_y, min_val_y):
            new_img, new_boxes = random_crop(img, bboxes, lower_bound=lower_bound, upper_bound=upper_bound, clip=clip, bbox_normalized=False)
        
        random_x = tf.squeeze(tf.random.uniform([1], minval=min_val_x, maxval=max_val_x, dtype=tf.int32))
        random_y = tf.squeeze(tf.random.uniform([1], minval=min_val_y, maxval=max_val_y, dtype=tf.int32))
        
    random_x = tf.squeeze(tf.random.uniform([1], minval=0, maxval=tf.squeeze(tf.cast(img_width, dtype=tf.int32) - crop_size + 1), dtype=tf.int32))
    random_y = tf.squeeze(tf.random.uniform([1], minval=0, maxval=tf.squeeze(tf.cast(img_height, dtype=tf.int32) - crop_size + 1), dtype=tf.int32))
    
    new_img = img[random_y:(random_y+tf.squeeze(crop_size)), random_x:(random_x+tf.squeeze(crop_size)), :]
    
    random_x = tf.cast(random_x, dtype=tf.float32)
    random_y = tf.cast(random_y, dtype=tf.float32)
    crop_size = tf.cast(crop_size, dtype=tf.float32)

    #un-normalize and get new normalized coordinates
    x1 = (bboxes[:, 0] - random_x) / crop_size
    y1 = (bboxes[:, 1] - random_y) / crop_size
    x2 = (bboxes[:, 2] - random_x) / crop_size
    y2 = (bboxes[:, 3] - random_y) / crop_size
    new_boxes = tf.keras.backend.stack([x1, y1, x2, y2], axis=1)
    
    new_boxes = filter_boxes(new_boxes)
    
    if clip:
        new_boxes = clip_boxes(new_boxes)
        
    #print('Failed so trying again')
    # catch-all because it is possible that no box is in between the max box and min box
    if new_boxes.get_shape()[0]==0:
        # bboxes already normalized
        new_img, new_boxes = random_crop(img, bboxes, lower_bound=lower_bound, upper_bound=upper_bound, clip=clip, bbox_normalized=False)
    
    return new_img, new_boxes


def tf_random_horizontal_flip(img, bboxes, prob=0.5):
    """Random horizontal flip on image and bounding boxes"""
    
    p = np.random.uniform(low=0, high=1, size=1)
    if p <= prob:
        img = tf.image.flip_left_right(img)
        x1 = 1 - bboxes[:, 2]
        x2 =  1 - bboxes[:, 0]
        y1 = bboxes[:, 1]
        y2 = bboxes[:, 3]
        #print(x1, y1, x2, y2)
        bboxes = tf.keras.backend.stack([x1, y1, x2, y2], axis=1)
    
    return img, bboxes


def photometric_color_distortion(img):
    """Photometric color distortions
    # see https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/image.py for hyperparameters
    ## they dont align with AlexNet so I'm not sure where they got them
    """
    #this should be applied before normalizing the image I believe
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_hue(img, max_delta=.05)
    img = tf.image.random_saturation(img, lower=0.95, upper=1.05)
    img = tf.clip_by_value(img, 0, 255)
    
    return img
    