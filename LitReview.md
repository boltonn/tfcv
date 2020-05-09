# Image Review

## Papers
1. **Image Classification**
    * VGG: [[paper]](https://arxiv.org/pdf/1409.1556.pdf), [[code]](https://github.com/dragen1860/TensorFlow-2.x-Tutorials/blob/master/06-CIFAR-VGG/network.py) *
        * good starting place for implementing a convnet (winner of Imagenet in 2014)
        * common **backbone** architecture (a lot of object detection models use the same standard image classification models as feature extractors)
        * known for its simplicity since it uses the same 3x3 conv filters w/ different channel size
        * since this is relatively early on it doesnt have the nice weigth initializations, still uses dropout and doesnt have batch normalization
        * as in rest of the models, the number inidcates the number of layers
    * ResNet: [[paper]](https://arxiv.org/pdf/1512.03385.pdf), [[blog]](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec), [[code]](https://github.com/calmisential/TensorFlow2.0_ResNet/tree/master/models) *
        * deeper networks (more layers) dont lead to a better model after a point; in fact it did worse which means that it the additional layers could not find an easy identity mapping of old layers to take advantage of lower level features, most likely due to **vanishing gradient** problem
        * this was resolved with **skip connections** where you concatenate the output of an older layer with on higher up (also known as residual connections, multiple layers make a **resiudal block**); worth noting that they use 1x1 convolutions to make the filter dimensions the same before concatentation
    * EfficientNet: [[paper]](https://arxiv.org/pdf/1905.11946v3.pdf)
        * scaling models or determing the right combination of width, depth, and image resolution/size is hard and largely manual. Google investigated the tradeoffs more systematically by doing a grid search of these parameters and found a combination that is a lot faster and little bit more accuracte (they also include minimizing flops for speed in the loss)
        * they release several versions of this model with different number of paramets (0-7 w/ 7 meaning more), which gets of course a trade-off of more accuracy but slower inference
    * Additional: AlexNet, InceptionV1, DenseNet, [ResNeXt](https://arxiv.org/pdf/1611.05431.pdf), [iResNet](https://arxiv.org/pdf/2004.04989.pdf), [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf), [DeConvolution](https://arxiv.org/pdf/1905.11926.pdf), [DCN layer](https://arxiv.org/pdf/1703.06211.pdf) [[code]](https://github.com/DHZS/tf-deformable-conv-layer/blob/master/nets/deformable_conv_layer.py)

2. **Object Detection**
    * R-CNN: [[paper]](https://arxiv.org/pdf/1311.2524.pdf)
        * Two-staged object model meaning you (1) propose the boxes and then (2) classify the boxes.
        * first SOTA object detection to use CNN's as the feature extractors for region proposals (RPN- Region Proposal Network) and then fit SVM on the top for classification and regression of bounding boxes
        * Best paper for understanding the regression. We are not outputting the points themselves. We are learning a transformation onto our anchor boxes. 
    * Faster R-CNN: [[paper]](https://arxiv.org/pdf/1506.01497.pdf)
        * Mostly same as R-CNN but use CNN for both proposals and outputs. They share the base of the model and then alternate when training
    * SSD: [[paper]](https://arxiv.org/pdf/1512.02325.pdf), [[code]](https://github.com/ChunML/ssd-tf2/blob/master/network.py) *
        * basic idea: skip the proposal stage (so one-stage/single-shot object detection) and have one model that produces bounding boxes and class predictions (single-stage)
        * use base/backbone VGG16
        * add *feature layers* with variations of filter size (and aspect ratios) to capture features of various sizes (*see anchor boxes below*)
        * use **non-maximum supression** to reduce to a reasonable size (take box w/ highest confidence and discard all boxes that have an Jaccard index or *IoU* above a threshold, .5)
        * incorporate regression of bounding box points (centers, width, and height) and categorical cross entropy of the class prediction (separate so that they can specialize on the two task)
        * add-ons: data augmentation, hard negative mining since a lot of boxes are negatives, and rule for various filter sizes and aspect ratios (various methods for this)
        * was a hell of a lot faster but loses a little accuracy
    * Feature Pyramid Network (FPN): [[paper]](https://arxiv.org/pdf/1612.03144.pdf), [[blog]](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
        * most convnets already have built in feature higher heirarchy that capture different spatial resolutions
        * so unlike SSD that fits on layers off different filter sizes onto the end of a backbone model, FPN runs predictions from a number of layers already in the model to capture low and high-resolution features (similar to skip connections except predictions at every layer), making it better for smaller objects
        * makes it also more computationally efficient
        * use ResNet as backbone, but this can be extended to other usually noted *\"X\"*-FPN
        * if going through convnet as normal is bottom-up where the spatial resolution is increased by a factor of 2 (# channels doubles), they also include a top-down *pathway* and incorporate 1x1 convolutions so each feature map is of the same size; this also allows higher resolution features built from semantically rich information (higher levels) 
    * RetinaNet: [[paper]](https://arxiv.org/pdf/1708.02002.pdf)
        * good paper to start and end with since it goes through history well and is intuitive
        * **focal loss** introduced. Since many of the boxes are negative examples (most often background) this accounts for the majority of the loss as well. Instead you can down weight negative examples (low probability $p_{t}$) by multiply loss by $(1-p_{t})^\gamma$ where $\gamma$ is a hyperparameter. They also mulitply by $\alpha$ as is typical in a normal class balance problem.
        * use a FPN on top of ResNet
    * EfficientDet: [[paper]](https://arxiv.org/pdf/1911.09070.pdf), [[Bi-FPN code]](https://github.com/yongqyu/BiFPN-tf2/blob/master/layer.py), [[production code]](https://github.com/google/automl/tree/master/efficientdet)
        * use EfficientNet as backbone and put new Bidirectional-FPN on top. (A FPN add this top-down pathway that basically had the output of every higher level features also connecting down. PANNet said lets add on another bottom up after the top-down, and Bi-FPN said let's add more skip connections so the information can flow more freely.)
    
    * Additional: [YOLO](https://arxiv.org/pdf/1506.02640.pdf)
    * **Important to Understanding**: [anchor boxes](https://d2l.ai/chapter_computer-vision/anchor.html) are where we apply a convolutional layer of a particular kernel size and aspect ratio onto the backbone rather than a dense layer so that, say, a 4x4xn output of a conv layer relates back to a particular *receptive field* in the original image, often visualized as dividing the original image into 4x4=16 boxes. you then run predictions for class and points for each of those *anchor boxes* and use IoU to pick the anchor box w/ the most overlap (this was hard for me to understand but Jeremy Howard shows in this [video](youtube.com/watch?v=0frKXR-2PBY) starting around the 30' mark). It helps me to think of it as first understanding the architecture and code without this concept, and then achor boxes give the intuition behind the carefully crafted layers and loss.
    
3. **Face Detection:**
    * PyramidBox: [[paper]](https://arxiv.org/pdf/1803.07737v2.pdf)
        * a
    * RetinaFace: [[paper]](https://arxiv.org/pdf/1905.00641v2.pdf), [[code]](https://github.com/peteryuX/retinaface-tf2/blob/master/modules/models.py)
        * mostly the same as RetinaNet but reduce the stride and sizes to capture smaller faces, only need one aspect ratio (1) since heads are mostly square
        * add in a second regression subnet similar to bounding box regression but for five facial landmarks
        * use a G-CNN (graph-based convolutional neural network)... *more soon*
    * Additional: [[MCNN]](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf), [[AINNOFace]](https://arxiv.org/pdf/1905.01585v3.pdf), [SSH](https://arxiv.org/pdf/1708.03979.pdf)

4. **Face Verification:**
    *recognition is 1:1 and verification is 1:k. What we're actually interested in is the face encoding so more of a 1:k problem*
    * FaceNet: [[paper]](https://arxiv.org/pdf/1503.03832.pdf), [[code 1]](https://github.com/sainimohit23/FaceNet-Real-Time-face-recognition/blob/master/train_triplet.py), [[code 2]](https://omoindrot.github.io/triplet-loss)
        * uses basic convnet that outputs 128 dimensional face embedding
        * inventors of triplet loss! where you want to maximize the euclidean distance between differences of an anchor image when compared to a positive image (same person) and a negative image (ideally similar to the anchor but a different person)
        * annoying part is creating the dataset generators to create the triplets both within a batch and for the entire dataset at the end of an epoch (training time therefore slow for a large amount of classes/people)
    * ArcFace: [[paper]](https://arxiv.org/pdf/1801.07698.pdf), [[code]](https://github.com.cnpmjs.org/peteryuX/arcface-tf2)
    
    
    
## Data:
* [WIDER Face](http://shuoyang1213.me/WIDERFACE/)- face detection
* [IMDB Face](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)- age and gender classification
* [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview)- TF record processed data