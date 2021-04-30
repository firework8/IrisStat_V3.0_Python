import os
import sys
import numpy as np
from PIL import Image 
import time
from tqdm import tqdm
import scipy.io as scio
import math
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols

def  evalSeg(gtIrisMask, segResIrisMask):
    # evalSeg - evaluate the performance of iris segmentation between one GT
    # iris mask and one Seg Result, return precision(P), recall(R), F1-score(F),
    # error rate(E), E2, IoU (from Caiyong TIFS2020)
    #
    # Usage: 
    # [P, R, F, E1, E2, Hdist, IoU, errImg] = evalSeg(gt, res)
    # Arguments:
    # gt                    - Ground Truth Iris Mask (binary map)
    # res                   - Seg Result Iris Mask (binary map)

    # Output:
    # Precision
    # Recall
    # F-score
    # E-error rate
    # Hausdorff distance
    # Interaction of Union
    # Error Image (RGB)

    # Author: 
    # Yunlong Wang
    # yunlong.wang@cripac.ia.ac.cn
    # CRIPAC, NLPR, CASIA
    # May 2020
    # Updated 2020.09.27
    gt = gtIrisMask.astype(np.bool_)
    res = segResIrisMask.astype(np.bool_)
    
    errormap = gt ^ res
    thetpr = gt & res
    thefpr = ~gt & res
    thefnr = ~res&gt
    theor = gt | res

    E1 = len(errormap.nonzero()[0]) / (gt.shape[0]*gt.shape[1])
    # true positive
    tpr = len(thetpr.nonzero()[0]) / (gt.shape[0]*gt.shape[1])
    # false postive
    fpr = len(thefpr.nonzero()[0]) / (gt.shape[0]*gt.shape[1])
    # false negative
    fnr = len(thefnr.nonzero()[0]) / (gt.shape[0]*gt.shape[1])
    E2 = (fpr+fnr)/2.0
    P = tpr/(tpr+fpr+1e-6)
    R = tpr/(tpr+fnr+1e-6)
    F = 2*P*R/(R+P+1e-6)
    # Interaction-over-Union (IOU)
    IoU = len(thetpr.nonzero()[0]) / (len(theor.nonzero()[0])+1e-6)

    # Hausdorff distance for shape similarity, parameterization of iris localizaiton
    Hdist = Hausdorff(gt, res)

    # Error Image (RGB)
    # Blue: iris regions correctly classified
    blueImg = (gt == 1) & (res == 1)
    # Red: iris regions misclassified as background
    redImg = (gt == 1) & (res == 0)
    # Green: background misclassified as iris regions
    greenImg = (gt == 0) & (res == 1)
    # Black: background correctly classified
    blackImg = (gt == 0) & (res == 0)

    # RGB color
    black = np.zeros([1,1,3])
    blue = np.zeros([1,1,3])
    blue[:,:,2] = 1.0
    green = np.zeros([1,1,3])
    green[:,:,1] = 1.0
    red = np.zeros([1,1,3])
    red[:,:,0] = 1.0
    
    blueC = np.transpose(np.tile(blueImg, (3,1,1)),(1,2,0)) * np.tile(blue, (gt.shape[0],gt.shape[1],1))
    greenC = np.transpose(np.tile(greenImg, (3,1,1)),(1,2,0)) * np.tile(green, (gt.shape[0],gt.shape[1],1))
    redC = np.transpose(np.tile(redImg, (3,1,1)),(1,2,0)) * np.tile(red, (gt.shape[0],gt.shape[1],1))
    blackC = np.transpose(np.tile(blackImg, (3,1,1)),(1,2,0)) * np.tile(black, (gt.shape[0],gt.shape[1],1))

    errImg = blueC + greenC + redC + blackC

    return P, R, F, E1, E2, Hdist , IoU, errImg

def Hausdorff(S,G):
    # Hausdorff calculates hausdorff distance between segmented objects in S
    # and ground truth objects in G
    #
    # Inputs:
    #   S: a label image contains segmented objects
    #   G: a label image contains ground truth objects
    #
    # Outputs:
    #   hausdorffDistance: as the name indicated
    #
    # Korsuk Sirinukunwattana
    # BIAlab, Department of Computer Science, University of Warwick
    # 2015

    # convert S and G to the same format
    # check if S or G is non-empty
    S1 = S.reshape(1,S.shape[0]*S.shape[1]).tolist()
    listS1 = list(set(S1[0]))
    listS = []
    for i in range(len(listS1)) :
        if listS1[i] != 0 :
            listS.append(listS1[i])
    numS = len(listS)
    G1 = G.reshape(1,G.shape[0]*G.shape[1]).tolist()
    listG1 = list(set(G1[0]))
    listG = []
    for i in range(len(listG1)) :
        if listG1[i] != 0 :
            listG.append(listG1[i])
    numG = len(listG)

    if numS == 0 and numG == 0 :
        hausdorffDistance = 0
        return hausdorffDistance
    elif numS == 0 or numG == 0 :
        hausdorffDistance = np.inf
        return hausdorffDistance
    
    # Calculate Hausdorff distance
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for j in range(S.shape[1]):
        for i in range(S.shape[0]):
            if S[i,j] == 1 :
                y1.append(i+1)
                y2.append(j+1)
            if G[i,j] == 1 :
                x1.append(i+1)
                x2.append(j+1)        
    x = np.append(np.array(x1).reshape(len(x1),1),np.array(x2).reshape(len(x2),1),axis=1)
    y = np.append(np.array(y1).reshape(len(y1),1),np.array(y2).reshape(len(y2),1),axis=1)
    # sup_{x \in G} inf_{y \in S} \|x-y\|
    neigh1 = NearestNeighbors(n_neighbors=1)
    neigh1.fit(y)
    knnsearch1 = neigh1.kneighbors(x)
    dist1 = max(knnsearch1[0])
    # sup_{x \in S} inf_{y \in G} \|x-y\|
    neigh2 = NearestNeighbors(n_neighbors=1)
    neigh1.fit(x)
    knnsearch2 = neigh1.kneighbors(y)
    dist2 = max(knnsearch2[0])

    hausdorffDistance = max(dist1,dist2)
    
    return hausdorffDistance
