# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:30:07 2024
Random sample for scoring flatness
@author: yoavb
"""

import torch
import random
import time

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from RunRecon import RunOriginalRecon

maxRadius = 260
minPerRadius = 300
maxPerRadius = 300
nImages = 280

bTiming = True

class CSample:
    """
    Hold 
    """
    def __init__(self, maskVolume, radiusImage):
        """
        Parameters
        ----------
        maskVolume : TYPE
            DESCRIPTION.

        """
        self.n = 0
        self.nPerRadius = torch.zeros(maxRadius, dtype=torch.int16)
        self.samples = []
        self.AddSamples(maskVolume.mask, radiusImage)
        
    def AddSamples(self, mask, radIm):
        start = time.monotonic()
        
        maxNSample = maxPerRadius * maxRadius
        self.images = torch.zeros(maxNSample, dtype=torch.int16)
        self.lines = torch.zeros(maxNSample, dtype=torch.int16)
        self.cols = torch.zeros(maxNSample, dtype=torch.int16)
        self.radiuses = torch.zeros(maxNSample, dtype=torch.int16)
        n = 0 
        
        relMin = 0
        relMinLast = 0
        bCenter = False
        while relMin < minPerRadius:
            if bCenter:
                iLine = random.randint(200,311)
                iCol = random.randint(200,311)
            else:
                iLine = random.randint(0,511)
                iCol = random.randint(0,511)
            bCenter = not bCenter
                
            iRadius = radIm.image.pData[iLine, iCol]
            if iRadius < maxRadius and self.nPerRadius[iRadius] < maxPerRadius:
                iImage = random.randint(0,nImages-1)
                if mask[iImage, iLine, iCol] > 0:
                    # Add sample
                    self.images[n] = iImage
                    self.lines[n] = iLine
                    self.cols[n] = iCol
                    self.radiuses[n] = iRadius
                    
                    #count samples
                    self.nPerRadius[iRadius] += 1
                    n += 1
                    
                    relMin = self.nPerRadius[0:250].min().item()
                    if n % 15000 == 0 or relMin > relMinLast + 9:
                        print(f'<AddSamles> {n=}, {relMin=}')
                        relMinLast = relMin
        self.n = n
        if bTiming:
            elapsed = time.monotonic() - start
            print(f"<CSample::AddSamples(> Elapsed time: {elapsed:.3f} seconds")
               
    def LogToFile(self):
        sfName = Config.LogFileName('Samples.csv')
        with open(sfName, 'w') as file:
            file.write('image, line, col, radius\n')
            for i in range(self.n):
                file.write(f'{self.images[i]}, {self.lines[i]}, {self.cols[i]}, {self.radiuses[i]}\n')
        print('Log file written: ', sfName)
               
        sfName = Config.LogFileName('NSamplesPerRadius.csv')
        with open(sfName, 'w') as file:
            file.write('radius, n\n')
            for i in range(maxRadius):
                file.write(f'{i}, {self.nPerRadius[i]}\n')
        print('Log file written: ', sfName)
        
        
def main():
    print('*** Test Sample')
    RunOriginalRecon()
    radIm = CRadiusImage()
    vol = CVolume('flatnessVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(vol)
    sample = CSample(maskVol, radIm)
    sample.LogToFile()
    
    

if __name__ == '__main__':
    main()
