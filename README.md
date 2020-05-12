# TFCV- Tensorflow Computer Vision

TFCV is my repo for experimenting with Computer Vision models in Tensorflow 2. The goal is to have everything simple and searchable.
<br><br>

* **TO-DO**: 
    * [x] create WiderFace dataset / TF Records using tfds
    * [ ] pick apart and implement [RetinaNet repo](https://github.com/fizyr/keras-retinanet) for understanding 
        * [x] load backbone model from tf.keras.applications for transfer learning
        * [x] create FPN
        * [x] create context layers for classification and regression for final output
        * [x] work on anchor boxes
        * [ ] image augmentations (**currently working on**) and incorporate w/ anhors for preprocessing
        * [ ] figure out loss
        * [ ] understand inference differences (non-max supression, etc.)
        * [ ] modularize for tensorflow 2.2 (can now change **train_step** and **predict_step**)
        * [ ] add callbacks and distributed c/omputing
    * [ ] create notebooks for each piece w/ visualizations and understanding
    * [ ] run model w/ hyperparameters from RetinaFace modified for tf==2.2 and if good enough convert to tflite
    * [ ] take trained model and fine tune with ArcFace loss for face embeddding
    * [ ] convert that to tflite and get working on RaspberryPi
    * [ ] add new backbone or object detection models
    
<br>



| Section | Description |
|-|-|
| [Image Classification]() | Image classification models |
| [Object Detection]() | Object Detection models |
| [Face Verification]() | Face embedding methods |
| [Notebooks](https://github.com/sheyemkote42/tfcv/tree/master/notebooks) | Step-by-step guide to implementation of the models |
| [Models ](https://github.com/sheyemkote42/tfcv/tree/master/models) | Models trained usign this repo |
| [Literature Review](https://github.com/sheyemkote42/tfcv/blob/master/LitReview.md) | Literature review w/ links and notes |
<br>
*All other sections are code written in old version of Tensorflow that I will replace*
