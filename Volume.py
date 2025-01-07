# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:18:10 2021

@author: yoavb
"""

from os import path
import numpy as np
import torch
import sys
import math

import Config
from Utils import FPrivateName
from PyString import GetValue, GetValue2, SetValue, RemoveValue, AddDesc
from Image import CImage

defaultMatrix = 512

debug = 0
verbosity = 1


def Clip(minValue, value, maxValue):
    if value <= minValue:
        return minValue
    if value >= maxValue:
        return maxValue
    return value
    

class CVolume():
    """
    Class holding one volume of images
    Can read files with short integers or float values
    Width, height, hPad and vPad are inferred from file name
    Default shape of data in file is 512*512 short
    nLines and nCols include Padded lines and columns
    Internally data is always held as 32-bit float
    """
    def __init__(self, name, sfName):
        """
        Args:
            fileName: name of file to read - values may be short or float nImages
        """
        if path.exists(sfName):
            sfFullName = sfName
        else:
            sfFullName = path.join(Config.sVolDir, sfName)
            if not path.exists(sfFullName):
                print(f'{Config.sRootVolDir=}')
                print(f'{sfName=}')
                sfTry = path.join(Config.sRootVolDir, sfName)
                print(f'{sfTry=}')
                if path.exists(sfTry):
                    sfFullName = sfTry
                
            if not path.exists(sfFullName):
                print(f'<CVolume::__init__> {name} MISSING file: {sfFullName}')
                sys.exit()
        if verbosity > 1:
            print(f'<CVolume::__init__> {name} file: {sfFullName}')
            
        self.name = name;
        self.fName = sfFullName
        self.fPrivateName = FPrivateName(sfFullName)
        mat = GetValue2(sfFullName, '_mat', '_matrix')
        if mat > 0:    
            self.nLines = mat
            self.nCols = mat
        else:
            self.nLines = defaultMatrix
            self.nCols = defaultMatrix
            nCols1 =  GetValue(sfFullName, '_width')
            if nCols1 > 0:
                self.nCols = nCols1
            nLines1 =  GetValue(sfFullName, '_height')
            if nLines1 > 0:
                self.nLines = nLines1
        
        self.nhPad = GetValue(sfFullName, '_hPad')
        self.nvPad = GetValue(sfFullName, '_vPad')
        self.ReadImages()
        self.i = 0

    def ReadImages(self):
        if verbosity > 1:
            print(f'<CVolume::ReadImages> Reading volume {self.fName}...')
        bSrcIsFloat = self.fName.find('.float.') > 0
        if bSrcIsFloat:
            images = np.memmap(self.fName, dtype='float32', mode='r').__array__()
        else:
            images = np.memmap(self.fName, dtype='int16', mode='r').__array__()
            
        pImages = torch.from_numpy(images.copy())
        pImages = pImages.view(-1,self.nLines,self.nCols)
        if bSrcIsFloat:
            self.pImages = pImages
        else:
            self.pImages = pImages.float()
            if self.fName.find('.raw') > 0:
                self.fName = self.fName.replace('.raw', '.float.rimg')

        self.nImages = self.pImages.size()[0]
        if debug:
            print('<CVol::ReadImages> nImages', self.nImages)
        
    def IsPadded(self):
        return self.nvPad > 0 or self.nhPad > 0

    def GetImage(self, iImage, sDesc):
        imageData = self.pImages[iImage].clone()
        image = CImage(self.nLines, self.nCols, pData=imageData, fName=self.fName, i=iImage)
        sImageName = self.fName
        sImageName = SetValue(sImageName, '_width', self.nCols)
        sImageName = SetValue(sImageName, '_height', self.nLines)
        sImageName = SetValue(sImageName, '_im', iImage+1);
        if not sDesc is None:
            sImageName = AddDesc(sImageName, sDesc)
        image.fName = sImageName
        if debug:
            print(f'<Volume::GetImage> {iImage} ==> {sImageName}')
        return image

    def GetImageUnPad(self, iImage, sDesc):
        if not self.IsPadded():
            return self.GetImage(iImage, sDesc)
        
        imageData = self.pImages[iImage,self.nvPad:self.nLines-self.nvPad,self.nhPad:self.nCols-self.nhPad].clone()
        image = CImage(self.nLines-2*self.nvPad, self.nCols-2*self.nhPad, pData=imageData, fName=self.fName, i=iImage)
        
        #print('<GetImageUnPad> self.fName', self.fName)
        sImageName = self.fName
        sImageName = RemoveValue(sImageName, '_hPad')
        sImageName = RemoveValue(sImageName, '_vPad')
        sImageName = SetValue(sImageName, '_width', self.nCols - 2 * self.nhPad)
        sImageName = SetValue(sImageName, '_height', self.nLines - 2 * self.nvPad)
        image.fName = SetValue(image.fName, '_im', iImage);
        if not sDesc is None:
            sImageName = AddDesc(sImageName, sDesc)
        image.fName = sImageName

        return image

    def GetImagePadded(self, iImage, sDesc):
        imageData = self.pImages[iImage].clone()
        image = CImage(self.nLines, self.nCols, pData=imageData, fName=self.fName, i=iImage)
        
        image.fName = self.fName
        image.fName = SetValue(image.fName, '_im', iImage);
        image.fName = AddDesc(image.fName, sDesc);
        
        image.nhPad = self.nhPad
        image.nvPad = self.nvPad
        return image
    
    def GetPaddedImages(self, iImage, n, sDesc):
        iFirst = iImage - int(n/2)
        aImages = []
        for i in range(n):
            iCur = Clip(0, iFirst + i, self.nImages - 1)
            aImages.append(self.GetImagePadded(iCur, sDesc))
        return aImages
    
    def OnInCompatible(self, other, sText):
        print('*** Input ERROR: Incompatible volumes ***')
        self.Print()
        other.Print()
        print(sText)
        print('Exiting on error...')
        sys.exit()
    
    def AssertCompatible(self, other):
        if self.nImages != other.nImages and other.nImages != 1:
            self.OnInCompatible(other, 'Different number of images')
        if self.nLines != other.nLines or self.nCols != other.nCols:
            self.OnInCompatible(other, 'Different image shape')

    def Print(self):
        print(f'CVolume {self.name} with {self.nImages} images')
        print(f'{self.pImages.size()=}')
        print(f'{self.pImages.dtype=}')
        print(f'File name: {self.fName}')

    def IsEqual(self, other):
        return torch.equal(self.pImages, other.pImages)

    def DumpSimilarVolume(self, pNewVolume, sNewName):
        size = pNewVolume.shape;
        sfName = f'd:/Dump/{sNewName}_width{size[2]}_height{size[1]}.float.rvol'
        npVolume = pNewVolume.numpy()
        with open (sfName, 'wb') as file:
            file.write(npVolume.tobytes())
        print(f'<DumpSimilarVolume> saved: {sfName}')

    def FindMaxPosition(self):
        iMax = self.pImages.argmax()
        imSize = self.nCols * self.nLines
        iImage = int(iMax / imSize)
        iInImage = iMax % imSize
        iLine = int(iInImage / self.nCols)
        iCol = iInImage % self.nCols
        return iImage, iLine, iCol

    def findRadius(self, iLine, iCol):
        dy = (iLine - self.nLines / 2)
        dx = (iCol - self.nCols / 2)
        rad = math.sqrt(dy*dy + dx*dx)
        return rad
        