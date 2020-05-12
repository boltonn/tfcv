# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IMDB Faces dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import numpy as np

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Since the publicly available face image datasets are often of small to medium size, rarely exceeding tens of thousands of images, and often without age information we decided to collect a large dataset of celebrities. For this purpose, we took the list of the most popular 100,000 actors as listed on the IMDb website and (automatically) crawled from their profiles date of birth, name, gender and all images related to that person. Additionally we crawled all profile images from pages of people from Wikipedia with the same meta information. We removed the images without timestamp (the date when the photo was taken). Assuming that the images with single faces are likely to show the actor and that the timestamp and date of birth are correct, we were able to assign to each such image the biological (real) age. Of course, we can not vouch for the accuracy of the assigned age information. Besides wrong timestamps, many images are stills from movies - movies that can have extended production times. In total we obtained 460,723 face images from 20,284 celebrities from IMDb and 62,328 from Wikipedia, thus 523,051 in total.

As some of the images (especially from IMDb) contain several people we only use the photos where the second strongest face detection is below a threshold. For the network to be equally discriminative for all ages, we equalize the age distribution for training. For more details please the see the paper.
"""

_PROJECT_URL = 'http://shuoyang1213.me/WIDERFACE/'

_WIDER_TRAIN_URL = ('https://drive.google.com/uc?export=download&'
                    'id=0B6eKvaijfFUDQUUwd21EckhUbWs')

_WIDER_VAL_URL = ('https://drive.google.com/uc?export=download&'
                  'id=0B6eKvaijfFUDd3dIRmpvSk8tLUk')

_WIDER_TEST_URL = ('https://drive.google.com/uc?export=download&'
                   'id=0B6eKvaijfFUDbW4tdGpaYjgzZkU')

_WIDER_ANNOT_URL = ('https://drive.google.com/uc?export=download&'
                    'id=1sAl2oml7hK6aZRdgRjqQJsjV5CEr7nl4')


# _WIDER_TRAIN_URL = ('https://drive.google.com/open?id=0B6eKvaijfFUDQUUwd21EckhUbWs')
# _WIDER_VAL_URL = ('https://drive.google.com/open?id=0B6eKvaijfFUDd3dIRmpvSk8tLUk')
# _WIDER_TEST_URL = ('https://drive.google.com/open?id=0B6eKvaijfFUDbW4tdGpaYjgzZkU')
# _WIDER_ANNOT_URL = ('http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip')


_CITATION = """
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}
"""

_DESCRIPTION = """
WIDER FACE dataset is a face detection benchmark dataset, of which images are 
selected from the publicly available WIDER dataset. We choose 32,203 images and 
label 393,703 faces with a high degree of variability in scale, pose and 
occlusion as depicted in the sample images. WIDER FACE dataset is organized 
based on 61 event classes. For each event class, we randomly select 40%/10%/50% 
data as training, validation and testing sets. We adopt the same evaluation 
metric employed in the PASCAL VOC dataset. Similar to MALF and Caltech datasets,
we do not release bounding box ground truth for the test images. Users are 
required to submit final prediction files, which we shall proceed to evaluate.
"""


class WiderFace(tfds.core.GeneratorBasedBuilder):
    """WIDER Face dataset."""

    VERSION = tfds.core.Version("0.1.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            # Describe the features of the dataset by following this url
            # https://www.tensorflow.org/datasets/api_docs/python/tfds/features
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(encoding_format='jpeg'),
                'image/filename': tfds.features.Text(),
                'faces': tfds.features.Sequence({
                    'bbox': tfds.features.BBoxFeature(),
                    'blur': tf.uint8,
                    'expression': tf.bool,
                    'illumination': tf.bool,
                    'occlusion': tf.uint8,
                    'pose': tf.bool,
                    'invalid': tf.bool
                })
            }), 
            #supervised_keys=("image", "category"),
            homepage=_PROJECT_URL,
            citation=_CITATION)

    def _split_generators(self, dl_manager):
        # Download the dataset and then extract it.
        extracted_dirs = dl_manager.download_and_extract({
            'wider_train': _WIDER_TRAIN_URL,
            'wider_val': _WIDER_VAL_URL,
            #'wider_test': _WIDER_TEST_URL,
            'wider_annot': _WIDER_ANNOT_URL
        })

        # Parsing the mat file which contains the list of train images

        return [
           tfds.core.SplitGenerator(
               name=tfds.Split.TRAIN,
               gen_kwargs={
                   'split': 'train',
                   'extracted_dirs': extracted_dirs
               }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'split': 'val',
                    'extracted_dirs': extracted_dirs
                })
            #excluding test data
#             tfds.core.SplitGenerator(
#                 name=tfds.Split.TEST,
#                 gen_kwargs={
#                     'split': 'test',
#                     'extracted_dirs': extracted_dirs
#                 })
        ]

    def _get_bounding_box_values(self, bbox, img_width, img_height):
        """Function to get normalized bounding box values.

        Args:
          bbox_annotations: list of bbox values in kitti format
          img_width: image width
          img_height: image height

        Returns:
          Normalized bounding box xmin, ymin, xmax, ymax values
        """
        xmin, ymin, wbox, hbox = np.array(bbox, dtype='float32')
        
        #make sure not bigger than the image
        xmin = np.clip(xmin, a_min=0, a_max=img_width)
        ymin = np.clip(ymin, a_min=0, a_max=img_height)
        xmax = np.clip(xmin + wbox, a_min=0, a_max=img_width)
        ymax = np.clip(ymin + hbox, a_min=0, a_max=img_height)
        
        xmin /= img_width
        xmax /= img_width
        ymin /= img_height
        ymax /= img_height
        
        return ymin, xmin, ymax, xmax
  
    def _get_image_shape(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        shape = image.shape[:2]
        return shape
    
    def _parse_annotation_file(self, filename):
        with tf.io.gfile.GFile(filename, 'r') as f:
            meta = f.read()
        meta = meta.split('\n')
        #chunkify
        split_meta=[]
        y=[]
        for line in meta:
            if '.jpg' in line:
                split_meta.append(y)
                y=[line]
            else:
                y.append(line)
        split_meta = split_meta[1:]
        
        formatted_annotations=[]
        for meta in split_meta:
            # filename includes the category 
            # (they don't treat this as a potential category in the original version but that could be interesting)
            category, fname = meta[0].split('/')
            out = {'image/filename': os.path.join(category, fname), 
                   'faces': []}
            annotations = meta[2:]
            for ann in annotations:
                ann = ann.split(' ', 12)
                face_annotation = {'bbox': ann[:4]} # (x1, y1, w, h)
                face_annotation['blur'] = ann[4]
                face_annotation['expression'] = ann[5]
                face_annotation['illumination'] = ann[6]
                face_annotation['invalid'] = ann[7]
                face_annotation['occlusion'] = ann[8]
                face_annotation['pose'] = ann[9]
                out['faces'].append(face_annotation)
            formatted_annotations.append(out)
        return formatted_annotations

    def _generate_examples(self, split, extracted_dirs):
        image_dir = os.path.join(extracted_dirs[f'wider_{split}'], f'WIDER_{split}', 'images')
        annotation_dir = os.path.join(extracted_dirs['wider_annot'], 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')
        
        annotations = self._parse_annotation_file(annotation_dir)
        
        for ann in annotations:
            # this includes a category subdirectory (ex: 0--Parade\0_Parade_marchingband_1_5.jpg)
            img_path = os.path.join(image_dir, ann['image/filename'])
            if os.path.isfile(img_path):
                img_height, img_width = self._get_image_shape(img_path)

                faces=[]
                for face in ann['faces']:
                    #noramlize bounding pox points
                    ymin, xmin, ymax, xmax = self._get_bounding_box_values(face['bbox'], img_width, img_height)

                    faces.append({
                        'bbox': tfds.features.BBox(xmin=xmin,
                                                   ymin=ymin,
                                                   xmax=xmax,
                                                   ymax=ymax),
                        'blur': face['blur'],
                        'expression': face['expression'],
                        'illumination': face['illumination'],
                        'invalid': face['invalid'], 
                        'occlusion': face['occlusion'],
                        'pose': face['pose']
                    })

                record = {
                    'image': img_path,
                    'image/filename': ann['image/filename'],
                    'faces': faces
                }
                # Yield a feature dictionary 
                yield ann['image/filename'], record
            else:
                print(f'Image file missing: {img_path}')
                continue
