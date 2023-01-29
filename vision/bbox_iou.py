# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_bbox_iou.ipynb.

# %% auto 0
__all__ = ['bbox_dim', 'intersection_area_pair', 'bbox_area', 'bbox_pair_iou', 'intersection_area', 'bbox_iou',
           'min_enclosing_bbox_area_pair', 'bbox_pair_giou', 'min_enclosing_bbox_area', 'bbox_giou']

# %% ../nbs/02_bbox_iou.ipynb 2
import numpy as np 
import torch
from typing import Union

# %% ../nbs/02_bbox_iou.ipynb 4
def bbox_dim(bbox: Union[np.ndarray, torch.Tensor]):
    """bbox: N x [4/6]"""
    if bbox.shape[1] == 6: return 3
    if bbox.shape[1] == 4: return 2
    raise NotImplementedError("Only 2D and 3D bboxes are defined")

# %% ../nbs/02_bbox_iou.ipynb 21
def intersection_area_pair(b1: torch.Tensor, b2: torch.Tensor, dim: int=2):
    x1 = torch.max(b1[:, :dim], b2[:, :dim])
    x2 = torch.min(b1[:, dim:], b2[:, dim:])
    inter_hw = torch.clamp((x2 - x1), min=0)
    inter = torch.prod(inter_hw, dim=-1)
    return inter

# %% ../nbs/02_bbox_iou.ipynb 24
def bbox_area(b: torch.Tensor, dim: int=2):
    return torch.prod(b[:, dim:] - b[:, :dim], dim=-1)

# %% ../nbs/02_bbox_iou.ipynb 27
def bbox_pair_iou(b1: torch.Tensor, b2: torch.Tensor):
    """where b1 and b2 are of the same shape [N, 4/6]"""
    assert b1.shape == b2.shape , "b1 and b2 are of not the same shape"
    dim = bbox_dim(b1)
    inter = intersection_area_pair(b1, b2, dim)
    b1_area, b2_area = bbox_area(b1, dim), bbox_area(b2, dim)
    iou = inter/ (b1_area + b2_area - inter)
    return iou

# %% ../nbs/02_bbox_iou.ipynb 37
def intersection_area(b1: torch.Tensor, b2: torch.Tensor, dim: int=2):
    x1 = torch.max(b1[:, None, :dim], b2[:, :dim])
    x2 = torch.min(b1[:, None, dim:], b2[:, dim:])
    inter = torch.clamp(x2 - x1, min=0)
    inter_area = torch.prod(inter, dim=-1)
    return inter_area

# %% ../nbs/02_bbox_iou.ipynb 43
def bbox_iou(b1: torch.Tensor, b2: torch.Tensor):
    """calculate iou between b1 Nx(4/6) and b2 Mx(4/6)
    """
    dim = bbox_dim(b1)
    inter_area = intersection_area(b1, b2, dim)
    b1_area, b2_area = bbox_area(b1, dim), bbox_area(b2, dim)
    union = b1_area[:, None] + b2_area - inter_area
    iou = inter_area / union
    return iou.clamp(min=0, max=1)

# %% ../nbs/02_bbox_iou.ipynb 54
def min_enclosing_bbox_area_pair(b1: torch.Tensor, b2: torch.Tensor, dim: int=2):
    xc = torch.min(b1[:, :dim], b2[:, :dim])
    yc = torch.max(b1[:, dim:], b2[:, dim:])
    area = torch.prod(torch.clamp(yc-xc, min=0), dim=-1)
    return area 

# %% ../nbs/02_bbox_iou.ipynb 60
def bbox_pair_giou(b1: torch.Tensor, b2: torch.Tensor):
    """where b1 and b2 are of the same shape [N, 4/6]"""
    dim = bbox_dim(b1)
    C = min_enclosing_bbox_area_pair(b1, b2, dim)
    inter_iou = intersection_area_pair(b1, b2, dim)
    b1a, b2a = bbox_area(b1, dim), bbox_area(b2, dim)
    union = (b1a+b2a-inter_iou)
    penalty = (C-union)/C
    iou = inter_iou/union
    giou = iou - penalty
    return giou

# %% ../nbs/02_bbox_iou.ipynb 62
def min_enclosing_bbox_area(b1: torch.Tensor, b2: torch.Tensor, dim: int=2):
    xc = torch.min(b1[:, None, :dim], b2[:, :dim])
    yc = torch.max(b1[:, None, dim:], b2[:, dim:])
    area = torch.prod(torch.clamp(yc-xc, min=0), dim=-1)
    return area 

# %% ../nbs/02_bbox_iou.ipynb 64
def bbox_giou(b1: torch.Tensor, b2: torch.Tensor):
    """where b1 and b2 are of the same shape [N, 4/6]"""
    dim = bbox_dim(b1)
    C = min_enclosing_bbox_area(b1, b2, dim)
    inter_iou = intersection_area(b1, b2, dim)
    b1a, b2a = bbox_area(b1, dim), bbox_area(b2, dim)
    union = (b1a[:, None]+b2a-inter_iou)
    penalty = (C-union)/C
    iou = inter_iou/union
    giou = iou - penalty
    return giou
