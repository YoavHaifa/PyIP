# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:10:24 2024
Torch NN Model for creating Poly Tables
@author: yoavb
"""

import time
import torch
import torch.nn as nn

nImages = 280
maxRadius = 128
nRows = 192
nDets = 688

"""
class CExDiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return input / 100

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        #return grad_output * 1.5 * (5 * input ** 2 - 1)
        return 0.01
"""

class CPolyModel:
    """
    """
    def __init__(self, nIn, nOut):
        self.CreateModel(nIn, nOut)
    
    def CreateModel(self, nIn, nOut):
        self.model = nn.Sequential(
        nn.Linear(nIn, nOut),
        nn.Sigmoid()
        )


def main():
    nIn = nImages * maxRadius
    nOut = nRows * nDets
    print(f'{nIn=}, {nOut=}')
    startTime0 = time.time()
    model = CPolyModel(nIn, nOut)
    deltaSec = time.time() - startTime0
    print(f'Model creaation consumed {deltaSec=} seconds')
    #PDiv = CExDiv.apply
    
    iVec = torch.randn(nIn)
    print('Apply Model')
    startTime = time.time()
    oVec = model.model(iVec)
    deltaSec = time.time() - startTime
    print(f'Model application consumed {deltaSec=} seconds')
    
    startTime = time.time()
    oVec = oVec / 100
    deltaSec = time.time() - startTime
    print(f'PDiv application consumed {deltaSec=} seconds')

    deltaSec = time.time() - startTime0
    print(f'All consumed {deltaSec=} seconds')
    
    
    
    print(f'{iVec=}')
    print(f'{oVec=}')
    

if __name__ == '__main__':
    main()

