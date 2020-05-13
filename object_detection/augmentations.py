import numpy as np
import tensorflow as tf

#TO-DO: color and transformations (translation, rotation, shear, etc.)


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
                      tf.where(tf.math.greater(bboxes[:, 1],  max_x)),
                      tf.where(tf.math.greater(bboxes[:, 3],  max_y))]:
        #make sure not empty since will break concat
        if not tf.equal(tf.size(condition), 0):
            invalid_indices.append(condition)
        if invalid_indices:
            invalid_indices = tf.concat(invalid_indices, axis=1)
            invalid_indices = tf.keras.backend.flatten(invalid_indices)
            invalid_indices = tf.sort(invalid_indices, axis=0, direction='ASCENDING')
            invalid_indices, _ = tf.unique(invalid_indices)
            
            n_boxes, _ = bboxes.shape
            indices = tf.cast([idx for idx in range(0, n_boxes)], dtype=tf.int64)
            
            valid_indices = tf.gather(indices, tf.where(tf.math.not_equal(indices, invalid_indices)))
            bboxes = tf.gather(bboxes, valid_indices)
    return bboxes



def clip_boxes(bboxes, clip_min=0, clip_max=1):
    """Clip normalized boxes within dimensions of an image"""
    # RetinaFace keeps this as a bounding box; if not then use filter_bboxes
    x1 = tf.clip_by_value(bboxes[:, 0], clip_value_min=clip_min, clip_value_max=clip_max)
    y1 = tf.clip_by_value(bboxes[:, 1], clip_value_min=clip_min, clip_value_max=clip_max)
    x2 = tf.clip_by_value(bboxes[:, 2], clip_value_min=clip_min, clip_value_max=clip_max)
    y2 = tf.clip_by_value(bboxes[:, 3], clip_value_min=clip_min, clip_value_max=clip_max)
    
    return tf.keras.backend.stack([x1, y1, x2, y2], axis=1)



def random_crop(img, bboxes, lower_bound=0.3, upper_bound=1., clip=True):
    # TO-DO: want to make sure it at least have one face ?
    """Random Crop of image proportional to smallest side"""
    img_height, img_width, _ = img.shape
    
    #take random proportion from the smallest side
    prop_crop_size = np.random.uniform(low=lower_bound, high=upper_bound, size=1)
    # get smallest side
    min_size = np.amin(img.shape[:-1]) # not including batch_size, -1 ignoring channels
    crop_size = int(prop_crop_size*min_size)
    #print(f'crop_size: {crop_size}')
    
    #generate random pixels to crop on (within range of new crop)
    random_x = np.random.randint(0, img_width - crop_size + 1)
    random_y = np.random.randint(0, img_height - crop_size + 1)
    #print(f'random_x: {random_x}')
    #print(f'random_y: {random_y}')

    img = img[random_y:(random_y+crop_size), random_x:(random_x+crop_size), :]
    
    #un-normalize and get new normalized coordinates
    x1 = (tf.math.round(bboxes[:, 0]*img_width) - random_x) / crop_size
    y1 = (tf.math.round(bboxes[:, 1]*img_height) - random_y) / crop_size
    x2 = (tf.math.round(bboxes[:, 2]*img_width) - random_x) / crop_size
    y2 = (tf.math.round(bboxes[:, 3]*img_height) - random_y) / crop_size
    bboxes = tf.keras.backend.stack([x1, y1, x2, y2], axis=1)
    
    if clip:
        bboxes = clip_boxes(bboxes)
    
    return img, bboxes


def random_horizontal_flip(img, bboxes, prob=0.5):
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
    