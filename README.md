This is an (re-)implementation of [Multipath refinement](https://arxiv.org/abs/1611.06612) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

## Requirements
Tensorflow 0.12  trained and tested on 0.12  ( should work on later versions as well with some small changes)
TF-slim api ( we use the tf slim libarary for many preprocessing and initailizaion networks)


## Model Description
Different varaints of recently proposed  Multipath refinement networks are trained and evaluated on Pascal VOC2012 Dataset.

## Instructions
Download  tensorflow slim models using (https://github.com/tensorflow/models) into a directory
```bash
git clone https://github.com/tensorflow/models
```
```bash
cd models/research/slim/
```
Now cp our folder segmentation_refinet to slim directory        
```bash
cp ~/segmentation_refinet ~/models/research/slim/
```
Download PASCAL VOC 2012 dataset to the root directory (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
To train the network, we use the augmented PASCAL VOC 2012 dataset with 10582 images for training and 1449 images for validation. Dowload Augmented classes
(using ['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0))
This will create a folder SegmentationClassAug in VOC2012 directory


##Training:
with a learning rate of 5e-4 untill model converges to some stabel loss.
```bash
ipython train_4_cascaded_refinet.py
```
You have to manually stop the training and change learning rate to 5e-5 and resume the training.
```bash
ipython train_4_cascaded_refinet.py
```
## evaluation
```bash
ipython four_cascaded_evaluate.py
```
Gives the mean iou and per category accuracy

## Visualization
To visualize some output images
run
```bash
ipython four_cascaded_inference.py
```

## to train and evaluate other vairaints
(a) Single refinet network
```bash
ipython train_single_refinet.py
```
```bash
ipython single_refinet_evaluate.py
```
(b) two cascaded refinet network
Note: we have used some image processing functions for following github repository
https://github.com/DrSleep/tensorflow-deeplab-resnet

Author
Fida Mohammad Thoker
