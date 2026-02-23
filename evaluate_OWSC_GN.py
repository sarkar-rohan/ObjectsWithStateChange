#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Code for evaluating trained models on single- and multi-image
state-invariant classification and retrieval tasks on the 
ObjectsWithStateChange (OWSC) GN dataset. 
===============================================================================
"""
"""
Load Libraries
"""
import torch
import sys 
print(torch.__version__)
from utils.InferenceUtility_GN import evaluate_SI_performance_single, evaluate_SI_performance_dual
from models.VGG_PIE_SingleEmb import VGG_avg_pitc, VGG_avg_piproxy, VGG_avg_picnn
from models.VGG_PAN_DualEmb import DualModel
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from ConfigLearn import ConfigOWSC_GN, HyperParams

"""
Input information and hyper-parameters from user
"""
dataset = 'OWSC' # OWSC
model = sys.argv[1] # pitc, piprx, picnn, piro, ours
model_path = sys.argv[2] 
nHeads = int(sys.argv[3])
nLayers = int(sys.argv[4])
hp = HyperParams(dataset)

print("Evaluation on State-invarant tasks on the ObjectsWithStateChange (OWSC) Dataset")

"""
Load configuration files for the OWSC dataset
""" 
Config = ConfigOWSC_GN(hp.case, hp.embDim, hp.batchSize, hp.alpha, hp.n_randsamp_class, hp.seed_inp)
"""
    Main script for evaluation
"""
print(device)
if model == 'piro' or model == 'ours':
    trained_model = DualModel(Config.inpChannel, Config.embedDim, nHeads, nLayers, hp.dropout, Config.Ncls).to(device)
elif model == 'pitc':
    trained_model = VGG_avg_pitc(Config.N_G, Config.Ncls).to(device)
elif model == 'piprx':
    trained_model = VGG_avg_piproxy(Config.N_G, Config.Ncls).to(device)
elif model == 'picnn':
    trained_model = VGG_avg_picnn(Config.N_G, Config.Ncls).to(device)
else:
    print("Wrong model.")
print("Evaluating "+model+" model")
print(trained_model)
trained_model.load_state_dict(torch.load(model_path, map_location=device))
trained_model.eval()
print("Loaded model weights")
if model == 'pitc' or model == 'piprx' or model == 'picnn':
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = evaluate_SI_performance_single(dataset, Config, trained_model, 1)
elif model == 'piro' or model == 'ours':
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc = evaluate_SI_performance_dual(dataset, Config, trained_model, 1)


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