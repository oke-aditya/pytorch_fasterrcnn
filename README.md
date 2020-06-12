# Pytorch Faster RCNN

# Note: -
1. It is better to use https://github.com/lgvaz/mantisshrimp
2. All the features in this (plus extra features) have been added to mantisshrimp.
3. It is a complete framework for object detection and more scalable.

Faster RCNN Fine-Tune Implementation in Pytorch.

## How to use ?
1. git clone the repo
```
git clone https://github.com/oke-aditya/pytorch_fasterrcnn.git
```
2. install the requirements (will add)
```
pip install -r requirements.txt
```

Simply edit the config file to set your hyper parameters.


3. Keep the training and validation csv file as follows


NOTE

Do not use target as 0 class. It is reserved as background.



```
image_id xtl ytl xbr ybr      target
1        xmin ymin xmax ymax   1
1        xmin ymin xmax ymax   2

2		 xmin ymin xmax ymax   3
```

4. Simply edit the config file to set your hyper parameters

5. Run the train.py file

# Features: -

- It works for multiple class object detection.

## Backbones Supported: -


- Note that backbones are pretrained on imagenet. 

- Following backbones are supported

1. vgg11, vgg13, vgg16, vgg19
2. resnet18, resnet34, resnet50, resnet101, resnet152
3. renext101
4. mobilenet_v2



Sample Outputs

# Helmet Detector
![Helmet Detection](outputs/helmet.jpg)

# Mask Detector
![Mask Detection](outputs/mask.jpg)



If you like the implemenation or have taken an inspiration do give a star :-)





