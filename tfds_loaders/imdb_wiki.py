"""IMDB Faces dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Since the publicly available face image datasets are often of small to medium size, rarely exceeding tens of thousands of images, and often without age information we decided to collect a large dataset of celebrities. For this purpose, we took the list of the most popular 100,000 actors as listed on the IMDb website and (automatically) crawled from their profiles date of birth, name, gender and all images related to that person. Additionally we crawled all profile images from pages of people from Wikipedia with the same meta information. We removed the images without timestamp (the date when the photo was taken). Assuming that the images with single faces are likely to show the actor and that the timestamp and date of birth are correct, we were able to assign to each such image the biological (real) age. Of course, we can not vouch for the accuracy of the assigned age information. Besides wrong timestamps, many images are stills from movies - movies that can have extended production times. In total we obtained 460,723 face images from 20,284 celebrities from IMDb and 62,328 from Wikipedia, thus 523,051 in total.

As some of the images (especially from IMDb) contain several people we only use the photos where the second strongest face detection is below a threshold. For the network to be equally discriminative for all ages, we equalize the age distribution for training. For more details please the see the paper.
"""

_URL = ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
# _IMDB_DATASET_ROOT_DIR = "imdb_crop"
# _IMDB_ANNOTATION_FILE = "imdb.mat"
# _WIKI_DATASET_ROOT_DIR = "wiki_crop"
# _WIKI_META_DATASET_ROOT_DIR = "wiki"
# _WIKI_ANNOTATION_FILE = "imdb.mat"


_CITATION = """\
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision},
  volume={126},
  number={2-4},
  pages={144--157},
  year={2018},
  publisher={Springer}
}
@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year = {2015},
  month = {December},
}
"""

# # Source URL of the IMDB faces dataset
# _IMDB_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
# _WIKI_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"
# _WIKI_META_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz"



class ImdbWikiFaces(tfds.core.GeneratorBasedBuilder):
    """IMDB-Wiki Faces dataset."""

    VERSION = tfds.core.Version("1.0.0")
    
    MANUAL_DOWNLOAD_INSTRUCTIONS = """Place 'imdb_wiki' in manual tensorflow_datasets path. \n
    It should contain 'imdb_crop' and 'wiki_crop' as well as custom train_ann.csv and test_ann.csv 
    """

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            # Describe the features of the dataset by following this url
            # https://www.tensorflow.org/datasets/api_docs/python/tfds/features
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "gender": tfds.features.ClassLabel(num_classes=2),
                "age": tf.int32,
                "name": tfds.features.Text()
            }),
            #supervised_keys=("image", ("gender", "age")), #doesnt look like they support multiple outputs
            homepage=_URL,
            citation=_CITATION)

    def _split_generators(self, dl_manager):
        pd = tfds.core.lazy_imports.pandas
        
        # Download the dataset and then extract it.
        main_path = os.path.join(dl_manager.manual_dir, 'imdb_wiki')
        
        def read_meta(main_path, split):
            df = pd.read_csv(os.path.join(main_path, f'{split}_ann.csv'))
            return df

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "image_dir": main_path,
                    "metadata": read_meta(main_path, 'train'),
                }),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "image_dir": main_path,
                    "metadata": read_meta(main_path, 'test'),
                })
        ]

    def _generate_examples(self, image_dir, metadata):
        
        # Iterate over all the rows in the dataframe and map each feature
        for _, row in metadata.iterrows():
            # Extract filename, gender, dob, photo_taken, 
            # face_score, second_face_score and celeb_id
            #some reason it is reading in as a string
            subdir = str(row['subdir'])
            if len(subdir)==1:
                subdir = '0' + subdir
            filename = os.path.join(image_dir, row['maindir'], subdir,  row['filename'])
            gender = row['gender']
            age = row['age']
            name = row['name']

            # Yield a feature dictionary 
            yield filename, {
              "image": filename,
              "gender": gender,
              "age": age,
              "name": name
            }