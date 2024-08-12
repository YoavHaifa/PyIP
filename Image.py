# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:09:56 2019
Handle real images
@author: yoavb
"""

import torch
import numpy as np
import sys
from os import path
import matplotlib.pyplot as plt

from PyString import GetValue, SetValue, RemoveValue, SetType, AddDesc
from Utils import GetDumpDir

class CImage:
    """
    Class for holding a single image
    Image may be read from file - short or float
    Image is held as 32-bit float
    """
    def __init__(self, nLines=0, nCols=0, pData=None, bInit=False, fName=None, i = 0):
        """
        Args:
            nLines
            nCols
            pData
            bInit
            fName - name of file with binary data
            i - index of image in volume
        """
        if nLines:
            self.nLines = nLines
        elif fName:
            self.nLines = GetValue(fName,'_height')
        else:
            self.nLines = 0
            
        if nCols:
            self.nCols = nCols
        elif fName:
            self.nCols = GetValue(fName,'_width')
        else:
            self.nCols = 0
            
        self.pData = pData
        self.fName = fName
        self.i = i
        if fName and (pData is None):
            self.ReadImage(fName)
        if bInit:
            self.pData = torch.zeros((nLines,nCols))
        self.nvPad = 0
        self.nhPad = 0

    #def clone(self):
    #    cloned = CImage(self.nLines, self.nCols, self.pData.clone())
    #    return cloned
    
    def IsPadded(self):
        return self.nvPad > 0 or self.nhPad > 0

    def ReadImage(self, fName):
        self.fName = fName
        print(f'Reading image {self.fName}...')
        bSrcIsFloat = self.fName.find('.float.') > 0
        if bSrcIsFloat:
            image = np.memmap(self.fName, dtype='float32', mode='r').__array__()
        else:
            image = np.memmap(self.fName, dtype='int16', mode='r').__array__()
        
        print(f'{image.ndim=}')
        print(f'{image.size=}')
        if len(image) != self.nLines * self.nCols:
            print('<ReadImage> size error', fName)
            print(f'<ReadImage> len(image) {len(image)} != {self.nLines} self.nLines * {self.nCols} self.nCols')
            sys.exit()
        
        image = torch.from_numpy(image.copy())
        image = image.view(self.nLines,self.nCols)
        if bSrcIsFloat:
            self.pData = image
        else:
            self.pData = image.float()
        
    def SetNameForDump(self):
        sName = self.fName
        sName = SetValue(sName, '_width', self.nCols)
        sName = SetValue(sName, '_height', self.nLines)
        sName = SetType(sName, '.float.rimg')
        self.fName = sName
    
    def WriteToFileInternal(self):
        self.SetNameForDump()
        npimage = self.pData.numpy()
        with open (self.fName, 'wb') as file:
            file.write(npimage.tobytes())
        print('Image saved:', self.fName)
        
    def WriteToFile(self, fileName=None):
        if not fileName:
            fileName = self.fName
        self.fName = path.join(GetDumpDir(), fileName)
        self.WriteToFileInternal()

    def WriteToFilePath(self, filePath = None, sDesc = None):
        if sDesc:
            print(f'<WriteToFilePath> sDesc: {sDesc}')
        else:
            print('<WriteToFilePath>')
        if not filePath is None:
             self.fName = filePath
        if sDesc:
            self.fName = AddDesc(self.fName, sDesc)
        self.WriteToFileInternal()
        return self.fName

    def Show(self):
        plt.matshow(self.pData, cmap='gray')
        plt.show()
        
    def CreateCopy(self):
        myCopy = CImage(self.nLines, self.nCols, self.pData.clone())
        myCopy.fName = self.fName
        return myCopy
    
    def CreateUnPaddedCopy(self):
        if not self.IsPadded():
            return self.CreateCopy()
        
        nNewLines = self.nLines - 2 * self.nvPad
        nNewCols = self.nCols - 2 * self.nhPad
        newData = self.pData[self.nvPad:self.nLines-self.nvPad, self.nhPad:self.nCols-self.nhPad].clone()
        myCopy = CImage(nNewLines, nNewCols, newData)
        sImageName = self.fName
        sImageName = RemoveValue(sImageName, '_hPad')
        sImageName = RemoveValue(sImageName, '_vPad')
        sImageName = SetValue(sImageName, '_width', nNewCols)
        sImageName = SetValue(sImageName, '_height', nNewLines)
        myCopy.fName = sImageName
        return myCopy
    
    def Multiply(self, multiplier):
        self.pData = self.pData * multiplier
        
    def Subtract(self, other):
        self.pData = self.pData - other.pData
        
    def SetConstant(self, value):
        self.pData.fill_(value)
        
    def AddConstantVertically(self, iFromCol, value):
        for iLine in range(self.nLines):
            self.pData[iLine,iFromCol:].add_(value)
            
    def AddConstantToLine(self, iLine, value):
        self.pData[iLine,:].add_(value)
        
    def ZeroFirstLines(self, nLines):
        self.pData[0:nLines,:] = 0
            
    def ZeroLastLines(self, iFrom):
        self.pData[iFrom:,:] = 0
            
    def Print(self):
        print(f'<CImage:Print> name: {self.fName}, i: {self.i}')
        print(f'pData.size {self.pData.size()}')
        if self.nLines * self.nCols < 100:
            print('pData', self.pData)
        print('---')

def WriteImage(image, fPath):
    npimage = image.numpy()
    with open (fPath, 'wb') as file:
        file.write(npimage.tobytes())
    print('Image saved:', fPath)
