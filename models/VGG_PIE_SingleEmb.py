"""
Code for different Pose-invariant architectures to learn category and object 
embeddings in the Single Embedding Space 

Code adapted from: 
PIEs: Pose Invariant Embeddings, CVPR 2019

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.vgg import model_urls as model_url_vgg
import torchvision.models as models
"""
PI-TC model with VGG backbone
"""
    
class VGG_avg_pitc(nn.Module):
    def __init__(self, input_view, output_class):
        super(VGG_avg_pitc, self).__init__()
        self.input_view = input_view
        VGG = models.vgg16(pretrained=True)
        VGG.classifier._modules['6']= nn.Linear(4096, out_features=output_class)
        
        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class                
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.class_centers = nn.Parameter(torch.randn(1,1,self.output_class, 4096))
        
    def forward(self, x):
        image_features = None
        shape_feature = None
        _, views, _, _, _ = x.shape
        for view in range(views):
            view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
            view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)

            if image_features is None:
                image_features = view_feature
            else:
                image_features = torch.cat([image_features, view_feature],1)
        
        class_feature = F.normalize(self.class_centers, p=2, dim=3, eps=1e-12)
        class_feature = class_feature.view(self.output_class,4096)
        
        shape_feature = torch.mean(image_features,1)
        shape_feature = F.normalize(shape_feature, p=2, dim=1, eps=1e-12)
        
        image_features = image_features.view(x.shape[0]*views,-1)
        image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)
        
        return image_features, shape_feature, class_feature 

"""
PI-Proxy model with VGG backbone
"""
class VGG_avg_piproxy(nn.Module):
    def __init__(self, input_view, output_class):
        super(VGG_avg_piproxy, self).__init__()
        self.input_view = input_view
        model_url_vgg['vgg16'] = model_url_vgg['vgg16'].replace('https://', 'http://')
        VGG = models.vgg16(pretrained=True)
        VGG.classifier._modules['6']= nn.Linear(4096, out_features=output_class)
        
        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class                
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.class_centers = nn.Parameter(torch.randn(1,1,self.output_class, 4096))
        
    def forward(self, x):      
        image_features = None
        shape_feature = None
        _, views, _, _, _ = x.shape
        for view in range(views):
            view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
            view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)

            if image_features is None:
                image_features = view_feature
            else:
                image_features = torch.cat([image_features, view_feature],1)
        
        class_feature = F.normalize(self.class_centers, p=2, dim=3, eps=1e-12)
        class_feature = class_feature.view(self.output_class,4096)
        
        shape_feature = torch.mean(image_features,1)
        shape_feature = F.normalize(shape_feature, p=2, dim=1, eps=1e-12)
        
        image_features = image_features.view(x.shape[0]*views,-1)
        image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)
        
        return image_features, shape_feature, class_feature 

"""
PI-CNN model with VGG backbone
"""
class VGG_avg_picnn(nn.Module):
    def __init__(self, input_view, output_class):
        super(VGG_avg_picnn, self).__init__()
        self.input_view = input_view
        model_url_vgg['vgg16'] = model_url_vgg['vgg16'].replace('https://', 'http://')
        VGG = models.vgg16(pretrained=True)
        VGG.classifier._modules['6']= nn.Linear(4096, out_features=output_class)
        
        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class                
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.class_centers = nn.Parameter(torch.randn(1,1,self.output_class, 4096))
        
    def forward(self, x):        
        image_features = None
        shape_feature = None
        _, views, _, _, _ = x.shape
        for view in range(views):
            view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
            view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)

            if image_features is None:
                image_features = view_feature
            else:
                image_features = torch.cat([image_features, view_feature],1)
        
        class_feature = F.normalize(self.class_centers, p=2, dim=3, eps=1e-12)
        class_feature = class_feature.view(self.output_class,4096)
        
        shape_feature = torch.mean(image_features,1)
        shape_feature = F.normalize(shape_feature, p=2, dim=1, eps=1e-12)
        
        image_features = image_features.view(x.shape[0]*views,-1)
        image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)
        
        return image_features, shape_feature, class_feature        