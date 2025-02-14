# -*- coding: utf-8 -*-
"""
Code for the Pose-invariant Attention Network encoder architecture 
to learn Dual Category and Object Embeddings

Code adapted from: 
Dual Pose-invariant Embeddings: Learning Category and Object-specific
Discriminative Representations for Recognition and Retrieval, CVPR 2024

"""
import torch.nn as nn
import torch.nn.functional as F
import torch 
from torchvision import models
print(torch.__version__)
from einops import rearrange
from models.architecture_utils import MultiHeadAttention
        
class DualModel(nn.Module):
    def __init__(self, inChannel, embDim, nHeads, nLayers, dropout, nCls):
        super(DualModel, self).__init__()

        print("Using Pose-invariant Attention Network with VGG16 backbone")
        self.backbone = models.vgg16(pretrained=True)
        print(self.backbone)
        num_ftrs = 25088 # vgg_out (dimensionality of image features from VGG-16)
        self.nCls = nCls
        self.backbone.classifier = nn.Identity()
        self.obj_embedder = nn.Sequential(nn.Linear(num_ftrs, embDim)) 
        self.cls_embedder = nn.Sequential(nn.Linear(num_ftrs, embDim)) 
        self.embDim = embDim
        self.useAttention = False
        if nHeads > 0:
            print("Use self-attention in transformers")
            self.useAttention = True
            print("Using Self-Attention: nHeads =", nHeads)
            self.obj_mha = MultiHeadAttention(nHeads, embDim, embDim, embDim, dropout)
            self.cls_mha = MultiHeadAttention(nHeads, embDim, embDim, embDim, dropout)

    
    def forward(self, imgBatch):
        dim = imgBatch.shape
        b,v,c,h,w = dim
        # Rearrange multi-view image tensor to pass them all through the backbone as a batch
        # Batch (b), Views (v), Channel (c), Height (h), Width (w)--> Batch*Views, Channel, Height, Width
        imgBatch  = rearrange(imgBatch, 'b v c h w -> (b v) c h w').cuda()
        # Extract Visual Features using the VGG Backbone 
        imgFtrs = self.backbone(imgBatch) # [b*v, vgg_out]
        # Pass visual features to two FC layers 
        # Generates Single-View Object and Category Embeddings 
        objEmbs = self.obj_embedder(imgFtrs) # [b*v, emb_dim]
        clsEmbs = self.cls_embedder(imgFtrs) # [b*v, emb_dim]
        # Reorganizing input batch for self-attention layers 
        # [b*v, emb_dim] --> [b, v, emb_dim]
        objEmbs = rearrange(objEmbs, '(b v) e -> b v e',b=b) 
        clsEmbs = rearrange(clsEmbs, '(b v) e -> b v e', b=b) 
        # Extract Normalize Multi-View Object and Category Embeddings 
        if self.useAttention == True:
            clsContext, clsAttn = self.cls_mha(clsEmbs, clsEmbs, clsEmbs)
            MVCLSEmbs = F.normalize(torch.mean(clsContext,1), dim=1, p=2)
            objContext, objAttn = self.obj_mha(objEmbs, objEmbs, objEmbs)
            MVOBJEmbs = F.normalize(torch.mean(objContext,1), dim=1, p=2)
        else:
            MVCLSEmbs = F.normalize(torch.mean(clsEmbs,1), dim=1, p=2)
            MVOBJEmbs = F.normalize(torch.mean(objEmbs,1), dim=1, p=2)
            objAttn = []
            clsAttn = []
        
        # Normalize Single-View Object and Category Embeddings
        SVCLSEmbs = F.normalize(clsEmbs, dim=2, p=2)
        SVOBJEmbs = F.normalize(objEmbs, dim=2, p=2)
        
        return SVOBJEmbs, SVCLSEmbs, MVOBJEmbs, MVCLSEmbs, objAttn, clsAttn