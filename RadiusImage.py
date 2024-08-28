# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:07:07 2024
Radius image - used instead of computing radius repeatedly
Convert radius to int - to get iRadius as index of entry in table
@author: yoavb
"""

import time
import torch

from Image import CImage


bTiming = False

class CRadiusImage:
    """
    """
    def __init__(self):
        """
        """
        start = time.monotonic()
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
        
        if bTiming:
            end = time.monotonic()
            elapsed = end - start
            print(f"<CreateRadiusImage> Elapsed time: {elapsed:.3f} seconds")
            
        self.image.WriteToFile("RadiusImage")
        
        
        
def main():
    print('*** Test Radius Image')
    radIm = CRadiusImage()
    print(f'{radIm=}')
    

if __name__ == '__main__':
    main()
