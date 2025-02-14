#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for inference on State-invariant Classification and Retrieval 
category and object-level tasks for the ObjectsWithStateChange (OWSC) dataset
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pickle
import torch
import numpy as np
import faiss
from functools import reduce
import operator
from utils.rank_metrics import calculate_mAP
from utils.helper_eval import load_class_data, loadDataset, FaissKNeighbors
import time



"""
===============================================================================
                             Inference framework
===============================================================================
"""


def predict_object_label(Query, Database, labels):
    CLS_KNN = FaissKNeighbors(1)
    CLS_KNN.fit(np.array(Database), np.array(labels))
    obj_predict = CLS_KNN.predict(np.array(Query))
    return obj_predict


def NNC_OWSC_SV(refC_emb, refO_emb, testC_emb, testO_emb, Config):
    cacc = 0
    oacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    
    for i, x in enumerate(refC_emb):
        if x.ndim == 1:
            refC_emb[i]= x.reshape(1, x.shape[0])
    XCTrain = np.concatenate(refC_emb,axis=0)
    XOTrain = np.concatenate(refO_emb,axis=0)
    oTrain = [[i]*refO_emb[i].shape[0] for i in range(Config.Ntrain)] 
    oTrain = torch.tensor(reduce(operator.concat, oTrain))
    cTrain = [[o2cTrain[i]]*refC_emb[i].shape[0] for i in range(Config.Ntrain)] 
    cTrain = torch.tensor(reduce(operator.concat, cTrain))
    
    XCTest = np.concatenate(testC_emb,axis=0)
    XOTest = np.concatenate(testO_emb,axis=0)
    oTest = [[i]*testO_emb[i].shape[0] for i in range(Config.Ntest)] 
    oTest = torch.tensor(reduce(operator.concat, oTest))
    cTest = [[o2cTest[i]]*testC_emb[i].shape[0] for i in range(Config.Ntest)] 
    cTest = torch.tensor(reduce(operator.concat, cTest))  
    
    """=====================================================================
                                SV Object Retrieval
       ====================================================================="""
    OR_KNN = FaissKNeighbors(XOTrain.shape[0])
    OR_KNN.fit(np.array(XOTrain), np.array(oTrain))
    objranks = OR_KNN.neighbors(np.array(XOTest))
    obj_mAP = calculate_mAP(objranks, oTest)
    print("SV Object Retrieval mAP: ", obj_mAP)
    del OR_KNN
    del objranks
    """=====================================================================
                                SV Object Recognition
       ====================================================================="""
    OC_KNN = FaissKNeighbors(1)
    OC_KNN.fit(np.array(XOTrain), np.array(oTrain))
    objpredict = OC_KNN.predict(np.array(XOTest))
    for x, gt in enumerate(oTest):
        if objpredict[x] == gt:
            oacc +=1
    print("SV Object Recognition: ", oacc/len(oTest))
    del OC_KNN
    del objpredict
    """=====================================================================
                                SV Category Recognition
       ====================================================================="""
    CLS_KNN = FaissKNeighbors(10) 
    CLS_KNN.fit(np.array(XCTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
    print("SV Category Recognition: ", cacc/len(cTest))
        
    del CLS_KNN
    del cls_predict
    """=====================================================================
                                SV Category Retrieval
       ====================================================================="""
    CR_KNN = FaissKNeighbors(XCTrain.shape[0])
    CR_KNN.fit(np.array(XCTrain), np.array(cTrain))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)
    print("SV Category Retrieval mAP: ", cr_mAP)
    
    del XCTrain
    del XOTrain
    del cTrain
    del oTrain
    del CR_KNN
    del crranks
    
    return cr_mAP*100, cacc/len(cTest)*100, obj_mAP*100, oacc/len(oTest)*100
    
def NNC_OWSC_MV(refC_emb, refO_emb, testC_emb, testO_emb, Config):
    cacc = 0
    oacc = 0
    o2cTrain = Config.o2ctrain
    o2cTest = Config.o2ctest
    XCTrain = np.concatenate(refC_emb,axis=0)
    cTrain = torch.tensor([o2cTrain[i] for i in range(Config.Ntrain)])
    XOTrain = np.concatenate(refO_emb,axis=0)
    oTrain = torch.tensor([i for i in range(Config.Ntrain)])
    
    XCTest = np.concatenate(testC_emb,axis=0)
    cTest = torch.tensor([o2cTest[i] for i in range(Config.Ntest)])
    XOTest = np.concatenate(testO_emb,axis=0)
    oTest = torch.tensor([i for i in range(Config.Ntest)])

        
    """=====================================================================
                                MV Category Retrieval
       ====================================================================="""
    CR_KNN = FaissKNeighbors(Config.Ntest)
    CR_KNN.fit(np.array(XCTrain), np.array(cTrain))
    crranks = CR_KNN.neighbors(np.array(XCTest))
    cr_mAP = calculate_mAP(crranks, cTest)
    
    print("MV Category Retrieval mAP: ", cr_mAP)
        
    del CR_KNN
    del crranks
    
    """=====================================================================
                                MV Category Recognition
       ====================================================================="""
    CLS_KNN = FaissKNeighbors(5) 
    CLS_KNN.fit(np.array(XCTrain), np.array(cTrain))
    cls_predict = CLS_KNN.predict(np.array(XCTest))
    
    for t in range(0,len(cTest)):
        if cls_predict[t]==cTest[t]:
            cacc += 1
            
    print("MV Category Recognition Accuracy: ", cacc/len(cTest))
        
    del CLS_KNN
    del cls_predict
    del XCTrain
    del XCTest
    del cTrain
    
    """=====================================================================
                                MV Object Retrieval
       ====================================================================="""
    OR_KNN = FaissKNeighbors(Config.Ntest)
    OR_KNN.fit(np.array(XOTrain), np.array(oTrain))
    orranks = OR_KNN.neighbors(np.array(XOTest))
    or_mAP = calculate_mAP(orranks, oTest)
    
    print("MV Object Retrieval mAP: ", or_mAP)
        
    del OR_KNN
    del orranks
    
    """=====================================================================
                                MV Object Recognition
       ====================================================================="""
    OBJ_KNN = FaissKNeighbors(1) 
    OBJ_KNN.fit(np.array(XOTrain), np.array(oTrain))
    obj_predict = OBJ_KNN.predict(np.array(XOTest))
    
    for t in range(0,len(oTest)):
        if obj_predict[t]==oTest[t]:
            oacc += 1
            
    print("MV Object Recognition mAP: ", oacc/len(oTest))
        
    del OBJ_KNN
    del obj_predict
    
    return cr_mAP*100, cacc/len(cTest)*100, or_mAP*100, oacc/len(oTest)*100


"""
===============================================================================
Evaluate State-invariant category and object-level classification and retrieval
                performance for Single embedding space methods
===============================================================================
"""

def evaluate_SI_performance_single(dataset, Config, trcv_model, nview):
    ref_OE={}
    ref_CE={}
    ref_mv_CE={}
    test_mv_CE={}
    test_OE={}
    test_CE={}
    label_cls = {}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}
    """
    Load the Single Pose-invariant embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData,shuffle=False,num_workers=8,batch_size=1)
    trcv_model.eval()
    with torch.no_grad():    
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)): 
            rOE, mvrOE, _ = trcv_model(ref_data.cuda())
            ref_OE[i]=rOE.squeeze().detach().cpu().numpy()
            ref_CE[i]=rOE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i]=mvrOE.detach().cpu().numpy()
            ref_mv_OE[i]=mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData,shuffle=False,num_workers=8,batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)): 
            tOE, mvtOE,_ = trcv_model(test_data.cuda())
            test_OE[j]=tOE.squeeze().detach().cpu().numpy()                
            test_CE[j]=tOE.squeeze().detach().cpu().numpy()
            label_obj[j]= obj_labels
            test_mv_CE[j]=mvtOE.detach().cpu().numpy()
            test_mv_OE[j]=mvtOE.detach().cpu().numpy()
            label_mv_OE[j]=j
    """ Compute performance from single-image query """      
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc  = NNC_OWSC_SV(list(ref_CE.values()), list(ref_OE.values()), list(test_CE.values()), list(test_OE.values()), Config) 
    """ Compute performance from multi-image query """
    MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc  = NNC_OWSC_MV(list(ref_mv_CE.values()), list(ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config) 
    return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc

"""
===============================================================================
Evaluate State-invariant category and object-level classification and retrieval
                performance for Dual embedding space methods
===============================================================================
"""

def evaluate_SI_performance_dual(dataset, Config, trcv_model, nview):
    ref_OE={}
    ref_CE={}
    ref_mv_CE={}
    test_mv_CE={}
    test_OE={}
    test_CE={}
    label_cls = {}
    label_obj = {}
    ref_mv_OE = {}
    test_mv_OE = {}
    label_mv_OE = {}
    """
    Load the Dual Pose-invariant embedding space model and 
    extract the gallery (train) embeddings and probe (test) embeddings
    """
    trainData = loadDataset(Config, 'train', dataset)
    trainLoader = DataLoader(trainData,shuffle=False,num_workers=8,batch_size=1)
    trcv_model.eval()
    with torch.no_grad():
        for i, (ref_data, obj_labels, cls_labels) in enumerate(tqdm(trainLoader)): 
            rOE, rCE, mvrOE, mvrCE, _,_ = trcv_model(ref_data)
            ref_OE[i]=rOE.squeeze().detach().cpu().numpy()
            ref_CE[i]=rCE.squeeze().detach().cpu().numpy()
            ref_mv_CE[i]=mvrCE.detach().cpu().numpy()
            ref_mv_OE[i]=mvrOE.detach().cpu().numpy()
    del trainData
    del trainLoader
    testData = loadDataset(Config, 'test', dataset)
    testLoader = DataLoader(testData,shuffle=False,num_workers=8,batch_size=1)
    with torch.no_grad():
        for j, (test_data, obj_labels, cls_labels) in enumerate(tqdm(testLoader)): 
                tOE, tCE, mvtOE, mvtCE, _,_ = trcv_model(test_data)
                test_OE[j]=tOE.squeeze().detach().cpu().numpy()                
                test_CE[j]=tCE.squeeze().detach().cpu().numpy()
                label_obj[j]= obj_labels
                test_mv_CE[j]=mvtCE.detach().cpu().numpy()
                test_mv_OE[j]=mvtOE.detach().cpu().numpy()
                label_mv_OE[j]=j
    """ Compute performance from single-image query """
    SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc  = NNC_OWSC_SV(list(ref_CE.values()), list(ref_OE.values()), list(test_CE.values()), list(test_OE.values()), Config) 
    """ Compute performance from multi-image query """
    MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc  = NNC_OWSC_MV(list(ref_mv_CE.values()), list(ref_mv_OE.values()), list(test_mv_CE.values()), list(test_mv_OE.values()), Config) 
    return SV_C_mAP, SV_C_acc, SV_O_mAP, SV_O_acc, MV_C_mAP, MV_C_acc, MV_O_mAP, MV_O_acc
