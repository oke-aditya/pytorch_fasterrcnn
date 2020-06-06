# Pytorch Faster RCNN

Faster RCNN Fine-Tune Implementation in Pytorch.

Simply edit the config file to set your hyper parameters.

Keep the training csv file as follows

```
image_id xtl ytl xbr ybr      target
1        xmin ymin xmax ymax   1
1        xmin ymin xmax ymax   2
```

It works for multiple class object detection.
Do not use target as 0 class. It is reserved as background.

I have used this to create a helmet detector and a mask detector as well.

Sample Outputs

# Helmet Detector
![Helmet Detection](outputs/helmet.jpg)

# Mask Detector
![Mask Detection](outputs/mask.jpg)


Currently it supports only mobilenet backbone, Will add functionality soon.


