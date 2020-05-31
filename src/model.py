import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_model(num_classes, min_size=300, max_size=500, backbone="mobile_net"):
    # note num_classes = total_classes + 1 for background.
    # For helmet we have yes, no and background

    # This is the default backbone rcnn. We can change it.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = model.to(device)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # ft_min_size = min_size
    # ft_max_size = max_size

    ft_mean = [0.485, 0.456, 0.406]
    ft_std = [0.229, 0.224, 0.225]

    mobile_net = torchvision.models.mobilenet_v2(pretrained=True)
    # print(mobile_net.features)
    # From that I got the output channels for mobilenet

    ft_backbone = mobile_net.features
    ft_backbone.out_channels = 1280

    ft_anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # ft_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)


    ft_model = FasterRCNN(backbone=ft_backbone,
                        num_classes=num_classes, 
                        # min_size=ft_min_size, 
                        # max_size=ft_max_size, 
                        image_mean=ft_mean, 
                        image_std=ft_std) 
                        # rpn_anchor_generator=ft_anchor_generator, 
                        # box_roi_pool=ft_roi_pooler)

    # print(ft_model)

    return ft_model


