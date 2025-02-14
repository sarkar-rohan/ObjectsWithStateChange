#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import torchvision.transforms as transforms
import numpy as np

"""
Hyperparameters used for training 
"""
class HyperParams():
    def __init__(self, dataset, expname="default", seed_no = 1):
        self.dataset = dataset # OOWL or MNet40
        self.case = 1
        self.embDim = 2048 # Embedding dimension
        """ Set margins for Object Object Embedding Space """
        self.alpha = 0.25 # Intra-class margin for object
        self.beta = 1.0 # Inter-class margin for object
        """ Set margins for Category Embedding Space """
        self.theta = 0.25 
        self.gamma = 4.0 # Controls angular separattion between categories 
        # for large-margin softmax loss
        # Number of randomly sampled images/object and batch size 
        # use 8 for ObjectPI and 12 for ModelNet40 and OWSC
        if dataset == 'MNet40':
            self.batchSize = 3
            self.n_randsamp_class = 12
            self.lamda = 1.0
        elif dataset == 'OOWL':
            self.batchSize = 4
            self.n_randsamp_class = 8 
            self.lamda = 1.0
        elif dataset == 'OWSC':
            self.batchSize = 3
            self.n_randsamp_class = 12
            self.lamda = 2.0
        else:
            print("Dataset not specified")
        self.seed_inp = seed_no # Seed input for an experiment
        self.nHeads = 1 # Number of Heads for transformer encoder
        self.nLayers = 1 # Number of Layers for transformer encoder
        self.dropout = 0.25 # Dropout for the self-attention layers 
        self.expname = expname # experiment name for saving model weights
        self.task = 'JNT' # change to JNT
        self.lamda = 1.0 # controls weight for intra-class and inter-class separation # 2.0
        # 1 indicates equal weightage
        self.ecc_ratio = 3.0 # Early Convergence Ratio

class ConfigOWSC():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigOWSC, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/OWSC/" # medres images stored, may need to switch to original
        self.save_path = "results/OWSC/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/OWSC_'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/OWSC_Best'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(bs)+'_OWSCbenchmark.csv'    
        self.save_plot_dist_path = self.save_path+str(bs)+'_OWSCdistance_curr.png'
        self.save_plot_learn_path = self.save_path+str(bs)+'_OWSClearn_curr.png'
        self.BS = bs
        self.Nepochs = 150
        self.Ncls = 21
        self.Niter = 1
        self.Ntrain = 331
        self.Ntest = 331
        self.part_const = 2
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy')
        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy')
        print("O2CTrain", self.o2ctrain)
        print("O2CTest", self.o2ctest)
        clist = []
        for c in range(self.Ncls):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp   
        self.class_list = clist
        print(clist)
        self.LR = 0.00005 # 0.00001
        self.alpha = a
        self.inpChannel = 3
        self.imgDim = 224
        self.embedDim = edim
        self.vData = False
        self.Ncomp = 3 # 3
        self.gal_vp = []
        self.probe_vp = []
        self.train_dataAug = transforms.Compose([ 
                                  transforms.Resize((self.imgDim,self.imgDim)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomAffine(5, translate=None, scale=(0.9,1.1), shear=[-1,1,-1,1], resample=False, fillcolor=0),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
        self.gal_vp = np.load(self.gallery_dir+'train_o2views.npy', allow_pickle=True).item()
        self.probe_vp = np.load(self.probe_dir+'test_o2views.npy', allow_pickle=True).item()
        self.N_G = n_s 
        print(self.N_G)


class ConfigOOWL():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigOOWL, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/ObjectPI/"
        self.save_path = "results/ObjectPI/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/OOWL_'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/OOWL_Best'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(case)+'_OOWLbenchmark.csv'    
        self.save_plot_dist_path = self.save_path+str(case)+'_OOWLdistance.png'
        self.save_plot_learn_path = self.save_path+str(case)+'_OOWLlearn.png'
        self.part_const = 2
        self.BS = bs
        self.Nepochs = 25
        self.Ncls = 25
        self.Niter = 1
        self.Ntrain = 382
        self.Ntest = 98
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy').astype('int')
        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy').astype('int')
        clist = []
        for c in range(25):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp   
        self.class_list = clist
        print(clist)
        
        self.LR = 0.00001 # 0.0001
        self.alpha = a
        self.inpChannel = 3
        self.imgDim = 224
        self.embedDim = edim
        self.vData = False
        self.Ncomp = 5
        self.gal_vp = []
        self.probe_vp = []
        self.train_dataAug = transforms.Compose([ 
                                  transforms.Resize((self.imgDim,self.imgDim)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomAffine(5, translate=None, scale=(0.9,1.1), shear=[-1,1,-1,1], resample=False, fillcolor=0),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
        #=====================================================================
        N_G_B = 8
        self.metaCategories = [range(x-15, x+1) for x in range(16, 383, 16)]
        self.gal_vp = [1, 2, 3, 4, 5, 6, 7, 8]
        self.probe_vp = [1, 2, 3, 4, 5, 6, 7, 8]
        self.N_G = min(len(self.gal_vp),N_G_B) 
        print(self.N_G)
        
class ConfigMNet40():
    def __init__(self, case, edim, bs, a, n_s, seed_no):
        super(ConfigMNet40, self).__init__()
        random.seed(seed_no)
        self.root_path = "data/ModelNet40/"
        self.save_path = "results/ModelNet40/"
        self.case = case
        self.data_dir = self.root_path
        self.gallery_dir = self.data_dir+"train/"
        self.probe_dir = self.data_dir+"test/"
        self.save_model_path = self.save_path+'models/MNet40_'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.best_model_path = self.save_path+'models/MNet40_Best'+str(a)+'_case'+str(case)+'_b'+str(bs)+str(edim)
        self.save_result_path = self.save_path+str(case)+'_MNet40benchmark.csv'    
        self.save_plot_dist_path = self.save_path+str(case)+'_AP_MNet40distance.png'
        self.save_plot_learn_path = self.save_path+str(case)+'_AP_MNet40learn.png'
        self.BS = bs
        self.Nepochs = 25
        self.Ncls = 40
        self.Niter = 1
        self.Ntrain = 3183
        self.Ntest = 800
        self.o2ctrain = np.load(self.gallery_dir+'train_o2c.npy')
        self.part_const = self.Ntrain/(4*self.Nepochs)
        self.o2ctest = np.load(self.probe_dir+'test_o2c.npy')
        print("O2CTrain", self.o2ctrain)
        print("O2CTest", self.o2ctest)
        clist = []
        for c in range(self.Ncls):
            clist.append([])
        for i, x in enumerate(self.o2ctrain):
            temp = clist[x]
            temp.append(i)
            clist[x] = temp   
        self.class_list = clist
        print(clist)
        self.LR = 0.00001 # 0.0001
        self.alpha = a
        self.inpChannel = 3
        self.imgDim = 224
        self.embedDim = edim
        self.vData = False
        self.Ncomp = 10
        self.gal_vp = []
        self.probe_vp = []
        self.train_dataAug = transforms.Compose([ 
                                  transforms.Resize((self.imgDim,self.imgDim)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomAffine(5, translate=None, scale=(0.9,1.1), shear=[-1,1,-1,1], resample=False, fillcolor=0),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
        #=====================================================================
        N_G_B = 12
        self.gal_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.probe_vp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.N_G = min(len(self.gal_vp),N_G_B) 
        print(self.N_G)
        
