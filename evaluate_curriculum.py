#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Code for evaluating trained models on single and multi-image 
pose or state-invariant classification and retrieval tasks
on ObjectsWithStateChange (OWSC SI), ModelNet-40, and ObjectPoseInvariance (OOWL) 
datasets.
===============================================================================
"""
"""
Load Libraries
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.datasets as dset
import torch
import sys 
print(torch.__version__)
from utils.InferenceUtility_SI import evaluate_SI_performance_dual
from utils.InferenceUtility_PI import evaluate_PI_performance_dual
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from ConfigLearn import ConfigOOWL, ConfigMNet40, ConfigOWSC_SI, ConfigFG3D, HyperParams
from models.VGG_PAN_DualEmb import DualModel

"""
Input information and hyper-parameters from user
"""
dataset = sys.argv[1] # OOWL, MNet40, OWSC, FG3D
model_path = sys.argv[2] 
nHeads = int(sys.argv[3])
nLayers = int(sys.argv[4])
hp = HyperParams(dataset)
"""
Load configuration files for the dataset
"""
# Datasets for testing invariant classification and retrieval tasks
if dataset == 'OOWL':
    Config = ConfigOOWL(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'MNet40':
    Config = ConfigMNet40(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'OWSC':
    Config = ConfigOWSC_SI(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
elif dataset == 'FG3D':
    Config = ConfigFG3D(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
else:
    print("Wrong Dataset")


"""
    Main script for evaluation
"""


trained_model = DualModel(Config.inpChannel, Config.embedDim, nHeads, nLayers, hp.dropout, Config.Ncls).to(device)
print("Evaluating model trained in the dual embedding space")
print("Number of Heads: ", nHeads)
print("Number of Layers: ", nLayers)
print(trained_model)
trained_model.load_state_dict(torch.load(model_path))    
trained_model.eval()

if dataset == 'OOWL'or dataset == 'MNet40' or dataset == 'FG3D':
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc = evaluate_PI_performance_dual(dataset, Config, trained_model, 1)
    MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = evaluate_PI_performance_dual(dataset, Config, trained_model, Config.N_G)
elif dataset == 'OWSC':
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = evaluate_SI_performance_dual(dataset, Config, trained_model, 1)


print("Loaded model weights")

print("-------------------------------------------------------------------")    
print("Test Results: \n")   
print("-------------------------- Classification (Accuracy) ----------------------------")
print("SV Category {} % | MV Category {} %|\n".format(SV_C_acc, MV_C_acc))  
print("SV Object {} %| MV Object {} %| ".format(SV_O_acc, MV_O_acc)) 
print("Average Classification {} %| ".format((SV_C_acc+MV_C_acc+SV_O_acc+MV_O_acc)/4)) 
print("-------------------------- Retrieval (mAP) ----------------------------")    
print("SV Category {} %| MV Category {} %|\n".format(SV_C_mAP, MV_C_mAP))
print("SV Object {} %| MV Object {} %| ".format(SV_O_mAP, MV_O_mAP))
print("Average retrieval {} %| ".format((SV_C_mAP+MV_C_mAP+SV_O_mAP+MV_O_mAP)/4))
print("-------------------------------------------------------------------") 
