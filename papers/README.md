# Computer vision


# Improving object detection models
- [Dropblock](https://arxiv.org/pdf/1810.12890.pdf) on top of each convolution layer increased the overall performance on imagenet by 1.6%. Note that we should keep the probability high in the inital (close to 1) and reduce it to 0.85 over the period of training. check if `dropblock` is der in your network or not. 
- Use [CIOU>DIOU>GIOU>IOU>(L1&L2 loss)](bbox_regression_improvements.ipynb) for bbox regression. In anchor based models, we saw relative increase in mAP by 10% (3-5% mAP). 


## Updates 
- [30-12-2022] `ATSS` -  Adaptive Training Sample Selection ([ATSS](https://arxiv.org/pdf/1912.02424.pdf)) :construction_worker:
- [29-12-2022] `FCOS` - Object detection without Anchor boxes. :construction_worker:
- [19-09-2022] `Vision Transformer` - Understanding vision transfomer in 6 simple steps. 
- [01-09-2022] `Cutout` - A new generalization method, performs better than dropout. 
- [22-08-2022] `CSPNet` - Make resnet more efficient. 
- [15-08-2022] `structural reparameterization` - Get `VGG` style speed with `Resnet` kind of architectures. 
- [10-08-2022] CIOU>DIOU>GIOU>IOU>(L1&L2 loss) - bbox_regression_improvements.ipynb


