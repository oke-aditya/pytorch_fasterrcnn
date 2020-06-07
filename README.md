# Pytorch Faster RCNN

Faster RCNN Fine-Tune Implementation in Pytorch.

Simply edit the config file to set your hyper parameters

Keep the training csv file as follows

image_id xtl ytl xbr ybr      target
1        xmin ymin xmax ymax   1
1        xmin ymin xmax ymax   2


It works for multiple class object detection.
Do not use target as 0 class. It is reserved as background.

I have used this to create a helmet detector and a mask detector as well.
Sample Outputs: -

# Helmet Detector
![Helmet Detection](outputs/helmet.jpg)

# Mask Detector
![Mask Detection](outputs/mask.jpg)
Currently it supports only mobilenet backbone, Will add functionality soon.

- Note the backbones are pretrained on imagenet. 

- Following backbones are supported

1. vgg11, vgg13, vgg16, vgg19
2. resnet18, resnet34, resnet50, resnet101, resnet152
3. renext101
4. mobilenet_v2

You can finetune the Anchors and roi align as well. Other hyperparemeters will be aded soon.



