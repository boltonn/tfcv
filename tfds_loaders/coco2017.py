"""COCO 2017 dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import json

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """COCO is a large-scale object detection, segmentation, and
captioning dataset. This version contains images, bounding boxes "
and labels for the 2017 version.
Note:
 * Some images from the train and validation sets don't have annotations.
 * Coco 2014 and 2017 uses the same images, but different train/val/test splits
 * The test split don't have any annotations (only images).
 * Coco defines 91 classes but the data only uses 80 classes.
 * Panotptic annotations defines defines 200 classes but only uses 133.

 <This is modified to only be 2017 object detection annotations>
"""

_URL = ("https://cocodataset.org/#download")

_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


class Coco2017(tfds.core.GeneratorBasedBuilder):
    """IMDB-Wiki Faces dataset."""

    VERSION = tfds.core.Version("1.0.0")
    
    MANUAL_DOWNLOAD_INSTRUCTIONS = """Place 'coco2017' in manual tensorflow_datasets path. \n
    It should contain 'train', 'val', 'test', and 'annotations'. Everything unzipped from urls 
    """

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            # Describe the features of the dataset by following this url
            # https://www.tensorflow.org/datasets/api_docs/python/tfds/features
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(encoding_format='jpeg'),
                "filename": tfds.features.Text(),
                "image_id": tf.int64,
                "objects": tfds.features.Sequence({
                    'id': tf.int64,
                    # Coco has unique id for each annotation. The id can be used for
                    # mapping panoptic image to semantic segmentation label.
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    # Coco has 91 categories but only 80 are present in the dataset
                    'label': tfds.features.ClassLabel(num_classes=80),
                    'is_crowd': tf.bool,
                })
            }),
            #supervised_keys=("image", ("gender", "age")), #doesnt look like they support multiple outputs
            homepage=_URL,
            citation=_CITATION)

    def _split_generators(self, dl_manager):
        
        # Download the dataset and then extract it.
        main_path = os.path.join(dl_manager.manual_dir, 'coco2017')

        def read_meta(main_path, mode):
            with tf.io.gfile.GFile(os.path.join(main_path, 'annotations', f'instances_{mode}2017.json')) as f:
                meta = json.load(f)
            return meta

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "image_dir": os.path.join(main_path, 'train'),
                    "metadata": read_meta(main_path, 'train'),
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "image_dir": os.path.join(main_path, 'val'),
                    "metadata": read_meta(main_path, 'val'),
                })
        ]

    def _generate_examples(self, image_dir, metadata):
        
        #format metadata

        # Each category is a dict:
        # {
        #    'id': 51,  # From 1-91, some entry missing
        #    'name': 'bowl',
        #    'supercategory': 'kitchen',
        # }
        categories = metadata['categories']
        categories_id2name = {c['id']: c['name'] for c in categories}
        
        #this will get the names
        self.info.features['objects']['label'].names = [c['name'] for c in categories]


        # Each image is a dict:
        # {
        #     'id': 262145,
        #     'file_name': 'COCO_train2017_000000262145.jpg'
        #     'flickr_url': 'http://farm8.staticflickr.com/7187/xyz.jpg',
        #     'coco_url': 'http://images.cocodataset.org/train2017/xyz.jpg',
        #     'license': 2,
        #     'date_captured': '2013-11-20 02:07:55',
        #     'height': 427,
        #     'width': 640,
        # }
        images = metadata['images']

        img_id2annotations = collections.defaultdict(list)
        for ann in metadata['annotations']:
            img_id2annotations[ann['image_id']].append(ann)

        img_id2annotations = {
            k: list(sorted(v, key=lambda a: a['id']))
                    for k, v in img_id2annotations.items()
        }

        def get_annotations(img_id):
            #return empty list if no annotations
            return img_id2annotations.get(img_id, [])


        # Iterate over all the rows in the dataframe and map each feature
        annotations_skipped=0
        for image_info in images:

            img_annotations = get_annotations(image_info['id'])
            
            if img_annotations:
                def build_bbox(x, y, width, height):
                    """Normalize bboxes and create tfds.features.BBox"""
                    return tfds.features.BBox(
                        ymin = y / image_info['height'],
                        xmin = x / image_info['width'],
                        ymax = (y + height) / image_info['height'],
                        xmax = (x + width) / image_info['width'],
                    )

                example = {
                    'image': os.path.join(image_dir, image_info['file_name']),
                    'filename': image_info['file_name'],
                    'image_id': image_info['id'],
                    'objects': [{
                        'id': detection['id'],
                        'area': detection['area'],
                        'bbox': build_bbox(*detection['bbox']),
                        #'label_id': detection['category_id'],
                        'label': categories_id2name[detection['category_id']],
                        'is_crowd': bool(detection['iscrowd']),
                    } for detection in img_annotations]
                }

                yield image_info['file_name'], example

            else:
                #print(image_info['id'])
                annotations_skipped+=1
                continue
                

            
    
        print(f'{annotations_skipped} images do not have annotations')