# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:02:04 2021

@author: yoavb
"""

import torch

class CVoxelEnv():
    """
    Voxel environment for CNN input and training
    Includes 2D or 3D input matrix + optional target
    """
    def __init__(self, iImage, iLine, iCol, x, y, yInterp, yPrev, yNext, bNormalize, bNormalizeAll, nInLayers):
        """
        Args:
            x: Input for computation - 2D or 3D matrix
            y: Optional target - if not defined, should be "None"
        """
        
        self.iVolume = 0
        self.iImage = iImage
        self.iLine = iLine
        self.iCol = iCol
        self.xOrig = x
        self.x = x.clone()
        self.y = y
        self.yInterp = yInterp
        self.yPrev = yPrev
        self.yNext = yNext
        self.yVert = [self.yPrev, self.yInterp, self.yNext]
        self.loss = 1000
        self.nUsed = 0
        self.bNormalize = bNormalize
        self.bNormalizeAll = bNormalizeAll
        self.bScale = False
        self.scaleFactor = 1
        
        # Subtruct average of environment
        if bNormalize:
            if bNormalizeAll:
                self.avg = torch.mean(self.x)
                self.x = self.x - self.avg
                
                # Divide by amplitude
                absX = torch.abs(self.x)
                self.amp = torch.mean(absX)
                if self.amp > 0:
                    self.x = self.x / self.amp
            else:
                self.avg = torch.mean(self.x[0:nInLayers])
                self.x[0:nInLayers] = self.x[0:nInLayers] - self.avg
                
                # Divide by amplitude
                absX = torch.abs(self.x[0:nInLayers])
                self.amp = torch.mean(absX)
                if self.amp > 0:
                    self.x[0:nInLayers] = self.x[0:nInLayers] / self.amp
                    
                if len(self.x) > nInLayers:
                    self.x[nInLayers:] = (self.x[nInLayers:] - 0.5) / 100

            if self.y is not None:
                self.y = self.y - self.avg
                if self.amp > 0:
                    self.y = self.y / self.amp
                    
            for yv in self.yVert:
                #if self.yOrig is not None:
                yv = yv - self.avg
                if self.amp > 0:
                    yv = yv / self.amp
                    
        else:
            self.avg = 0
            self.amp = 1
            
            if self.bScale:
                self.x = self.x / self.scaleFactor
                self.y = self.y / self.scaleFactor

    def Print(self):
        print('<CVoxelEnv::Print>')
        print('X Original:', self.xOrig)
        print('X:', self.x)
        print('Y:', self.y)
        for i, yv in enumerate (self.yVert):
            print(f'yVert{i}: {yv}')
        if self.bNormalize:
            print('avg:', self.avg)
            print('amp:', self.amp)
        else:
            print('Not Normalized!')
            
        
