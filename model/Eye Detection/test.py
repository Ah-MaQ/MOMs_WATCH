#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import os
from time import time
from datetime import datetime
import torch
from ritnet.dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
from ritnet.utils import mIoU
from ritnet.dataset import transform
from ritnet.opt import parse_args
from ritnet.models import model_dict
from ritnet.utils import get_predictions

if __name__ == '__main__':
    
    args = parse_args()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Starting at:", dt_string)

    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model_dict[args.model]
    model = model.to(device)
    filename = "trained_rit.pkl"
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()
    from torchsummary import summary
    summary(model, input_size=(1, 224, 224))

    test_set = IrisDataset(filepath = '/media/di2/T7/RIT_net/test25k/',\
    #/home/hoseung/Work/NIA/data/rit_data/test25k/',\
                                 split = 'valid',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    counter=0
    
    ious = []   
    t00 = time() 
    with torch.no_grad():
        for i, batchdata in enumerate(testloader):
            if i < 1100:
                continue
            img,labels,index,x,y= batchdata
            data = img.to(device)       
            output = model(data)            
            predict = get_predictions(output)
            #print(labels.shape)
            iou = mIoU(predict,labels)*3.7
            ious.append(iou)
    
            if i%100 == 0:
                print(f'[{i}/{len(testloader)}] IoU: {iou:.3f}, took {time()-t00:.2f} sec')
                t00 = time()
        
    print(f'\nTest complete...  mIoU {np.average(ious):3f}')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Finished at:", dt_string)
    
