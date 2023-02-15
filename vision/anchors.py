# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_anchors.ipynb.

# %% auto 0
__all__ = ['remove_boxes_outside_img', 'clip2img', 'box_grid_based']

# %% ../nbs/03_anchors.ipynb 3
import torch

# %% ../nbs/03_anchors.ipynb 20
def remove_boxes_outside_img(boxes, img_size):
    """img_size is PIL.size and boxes is xyxy"""
    lt = (boxes[:, :2] >=0).any(1)
    rt = (boxes[:, 2:] <=torch.Tensor(img_size)).any(1)
    return boxes[lt & rt]

# %% ../nbs/03_anchors.ipynb 25
def clip2img(boxes: torch.Tensor, img_size: tuple):
    """boxes are in xyxy format, use PIL.size for img_size"""
    out = boxes.clone()
    out[:, :2] = torch.maximum(torch.Tensor([0]), boxes[:, :2])
    out[:, 2:] = torch.minimum(torch.Tensor(img_size), boxes[:, 2:])
    return out

# %% ../nbs/03_anchors.ipynb 28
def box_grid_based(img_size: tuple, grid_size: int, clip= True):
    """Grid based anchor generation. img_size is PIL.size
    if image size is a multiple of stride, the generated boxes need not be clipped."""
    patch_size = (torch.Tensor(img_size)/grid_size).floor().type(torch.int)
    x, y = torch.arange(0, grid_size, step=1), torch.arange(0, grid_size, step=1)
    yx, xy = torch.meshgrid([x, y], indexing="xy")
    grid = torch.stack([yx, xy]).reshape(2, -1).T
    lxy = grid*patch_size
    hxy = lxy+ patch_size 
    boxes = torch.hstack([lxy, hxy])
    bc = clip2img(boxes, img_size) if clip else remove_boxes_outside_img(boxes, img_size)
    return bc 
