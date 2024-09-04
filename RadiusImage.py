# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:07:07 2024
Radius image - used instead of computing radius repeatedly
Convert radius to int - to get iRadius as index of entry in table
@author: yoavb
"""

import time
import torch

import Config
from Image import CImage
from Utils import WriteMatrixToFile



bTiming = True
debug = 0
verbosity = 5

class CRadiusImage:
    """
    """
    def __init__(self):
        """
        """
        start = time.monotonic()
        self.ComputeRadiusPerPixel()
        self.CountPixelsPerRadius()
        self.MapRadiusToPixels()
        
        if bTiming:
            elapsed = time.monotonic() - start
            print(f"<CRadiusImage::__init__(> Elapsed time: {elapsed:.3f} seconds")

    def ComputeRadiusPerPixel(self):
        self.image = CImage(512,512,bInit=True)
        centerX = 255.5
        centerY = 255.5
        
        xVector = torch.arange(0,512)
        xDiff = xVector - centerX
        xDiff2 = torch.pow(xDiff,2)
        
        for y in range(512):
            yDiff2 = (y-centerY)**2
            sumVec = xDiff2 + yDiff2
            self.image.pData[y] = torch.sqrt(sumVec)
            
            #for x in range(512):
            #    radius = math.sqrt(yDiff2 + xDiff2[x])
            #    image.pData[y,x] = radius
            
        #self.image.WriteToFile("RadiusImage")
        self.image.Float2Short()
        self.image.WriteToFile("RadiusImage")
 
    def CountPixelsPerRadius(self):
        self.maxRadius = self.image.pData.max()
        self.nRadiuses = self.maxRadius + 1
        if verbosity > 1:
            print(f'Max Radius {self.maxRadius}')
        self.countPerRadius = torch.zeros(self.nRadiuses, dtype=torch.int16)
        for iLine in range(512):
            for iCol in range(512):
                self.countPerRadius[self.image.pData[iLine,iCol]] += 1 
        
        f = Config.OpenLog('CountPerRadius.csv')
        for i in range (self.nRadiuses):
            f.write(f'{self.countPerRadius[i]}\n')
        f.close()
        
        self.maxPixelsPerRadius = self.countPerRadius.max()
        if verbosity > 1:
            print(f'Max pixels per radius {self.maxPixelsPerRadius}')
     
    def MapRadiusToPixels(self):
        if verbosity > 1:
            print('<MapRadiusToPixels>')
        self.rad2PixLine = torch.zeros([self.nRadiuses, self.maxPixelsPerRadius+1], dtype=torch.int16)
        self.rad2PixCol = torch.zeros([self.nRadiuses, self.maxPixelsPerRadius+1], dtype=torch.int16)
        countPerRad = torch.zeros(self.nRadiuses, dtype=torch.int16)
        for iLine in range(512):
            for iCol in range(512):
                radius = self.image.pData[iLine,iCol]
                iIn = countPerRad[radius]
                self.rad2PixLine[radius, iIn] = iLine
                self.rad2PixCol[radius, iIn] = iCol
                countPerRad[radius] += 1
                
        if debug:
            WriteMatrixToFile(self.rad2PixLine, 'rad2PixLine', 'short')
            WriteMatrixToFile(self.rad2PixCol, 'rad2PixCol', 'short')
                

def main():
    print('*** Test Radius Image')
    radIm = CRadiusImage()
    print(f'{radIm=}')
    

if __name__ == '__main__':
    main()
