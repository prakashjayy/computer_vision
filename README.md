# Computer vision


# Improving object detection models
- [Dropblock](https://arxiv.org/pdf/1810.12890.pdf) on top of each convolution layer increased the overall performance on imagenet by 1.6%. Note that we should keep the probability high in the inital (close to 1) and reduce it to 0.85 over the period of training. check if `dropblock` is der in your network or not. 
- Use [CIOU>DIOU>GIOU>IOU>(L1&L2 loss)](bbox_regression_improvements.ipynb) for bbox regression. In anchor based models, we saw relative increase in mAP by 10% (3-5% mAP). 