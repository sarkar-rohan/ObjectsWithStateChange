# -*- coding: utf-8 -*-
"""
Helper functions for the evaluation framework
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from PIL import Image
import os
import torch
import numpy as np
import faiss

"""
===============================================================================
    Class for Fast Similarity Search for classification and retrieval 
    based on k-Nearest Neighbors. This is based on Faiss library 
    Reference: https://github.com/facebookresearch/faiss
===============================================================================
"""

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k
        
    """ Build Faiss index """
    def fit(self, X, y):
        d = X.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X.astype(np.float32))
        self.y = y
    
    """ Find k-nearest neighbors """
    def neighbors(self, X):
        _, indices = self.index.search(X.astype(np.float16), k=self.k)
        votes = self.y[indices].astype(np.int32)
        return votes
    
    """Predict label based on majority voting"""
    def predict(self, X):
        _, indices = self.index.search(X.astype(np.float16), k=self.k)
        votes = self.y[indices].astype(np.int32)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    
    """ Used for SV Object Recognition."""
    def predict_sv(self, X):
        _, indices = self.index.search(X.astype(np.float16), k=self.k+1)
        votes = self.y[indices].astype(np.int32)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes[:,1:]])
        return predictions

"""
===============================================================================
          CUSTOM DATALOADER FOR TESTING, VALIDATION, AND EVALUATION
===============================================================================
Custom Dataset class for loading multiple images of each object
"""     
    
class loadDataset(Dataset):
    
    def __init__(self,Config, split, dataset_name):
        self.transform = transforms.Compose([
                                       transforms.Resize((Config.imgDim, Config.imgDim)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
        self.split = split
        if self.split == "train" or self.split == "val":
            self.N_class = Config.Ntrain
            self.datadir = Config.gallery_dir
            self.obj2cls = Config.o2ctrain
        elif self.split == "test":
            self.N_class = Config.Ntest
            self.datadir = Config.probe_dir
            self.obj2cls = Config.o2ctest
        self.dataset = dataset_name
        self.N_G = Config.N_G
        self.Config = Config
        
    def applyTransform(self,img_path):
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
            
    def __getitem__(self,index):
        obj_ind = index
        cls_ind = self.obj2cls[obj_ind]
        images = list()
        obj_labels = list()
        cls_labels = list()
        
        if self.split == 'train':
            vp = self.Config.gal_vp
            if self.dataset == "OWSC":
                vp = self.Config.gal_vp[index+1]
        elif self.split == 'val':
            vp = self.Config.gal_vp
            if self.dataset == "OWSC":
                if len(self.Config.gal_vp[index+1])<=self.N_G:
                    vp = self.Config.gal_vp[index+1]
                else:
                    vp = random.sample(self.Config.gal_vp[index+1], self.N_G)
        elif self.split == 'test':
            vp = self.Config.probe_vp 
            if self.dataset == "OWSC":
                vp = self.Config.probe_vp[index+1]
                
        for j in vp:
            if self.dataset == "OOWL":
                data_path = self.datadir+str(index+1)+"/"+str(j)+".jpg"
            elif self.dataset == "MNet40":
                data_path = self.datadir+str(index+1)+"/"+str(j).zfill(3)+".jpg"
            elif self.dataset == "OWSC":
                data_path = j
            else:
                print("Dataset not recognized.")
                
            images.append(self.applyTransform(data_path))
            obj_labels.append(torch.from_numpy(np.array(obj_ind)))
            cls_labels.append(torch.from_numpy(np.array(cls_ind)))
        return torch.stack(images), torch.stack(obj_labels), torch.stack(cls_labels)
        
    def __len__(self):
        return self.N_class
    
def load_class_data(i, dataset, datadir, flag, Config):
    temp = list()
    if flag == 0:
        vp = Config.gal_vp
        if dataset == "OWSC":
            vp = Config.gal_vp[i+1]
    elif flag == 1:
        vp = Config.probe_vp 
        if dataset == "OWSC":
            vp = Config.probe_vp[i+1]
    elif flag == 2:
        vp = Config.val_vp
    else:
            print("Error. Wrong Flag.")
    for j in vp:
        if dataset == "OOWL":
            data_path = datadir+str(i+1)+"/"+str(j)+".jpg"
        elif dataset == "MNet40":
            data_path = datadir+str(i+1)+"/"+str(j).zfill(3)+".jpg"
        elif dataset == "OWSC":
            data_path = j
        else:
            print("Dataset not recognized.")
        if os.path.isfile(data_path):
            img = Image.open(data_path)
            if dataset == "OOWL":
                timg = transforms.Compose([
                                       transforms.Resize(Config.imgDim),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
            elif dataset == "MNet40" or dataset == "OWSC":
                timg = transforms.Compose([
                                       transforms.Resize(Config.imgDim),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ])
            t  = timg(img)
            t.resize_((1,Config.inpChannel,Config.imgDim,Config.imgDim))
            temp.append(t)
    return temp
