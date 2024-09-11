# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:30:07 2024
Random sample for scoring flatness for each image of a volume
Select same number of voxels from any ring of any image
@author: yoavb
"""

import torch
import random
import time
import sys

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from RunRecon import RunOriginalRecon

maxRadius = 260
minPerRadius = 300
maxPerRadius = 300
nImages = 280

nSamplesPerRadius = 12

bTiming = True
verbosity = 1

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
        self.nImages = nImages
        self.nRadiusesPerImage = 255
        self.nSamplesPerRadius = nSamplesPerRadius
        self.samplesLines = torch.zeros([self.nImages,self.nRadiusesPerImage,self.nSamplesPerRadius], dtype=torch.int)
        self.samplesCols = torch.zeros([self.nImages,self.nRadiusesPerImage,self.nSamplesPerRadius], dtype=torch.int)
        self.nValidSamples = torch.zeros([self.nImages,self.nRadiusesPerImage], dtype=torch.int16)
        self.nInvalidSamples = torch.zeros([self.nImages,self.nRadiusesPerImage], dtype=torch.int16)
        self.AddSamples(maskVolume.mask, radiusImage)
        
        self.radIm = radiusImage
    
    def AddSamples(self, mask, radIm):
        if verbosity > 0:
            print(f'<CSample::AddSamples> {self.nImages=} {self.nRadiusesPerImage=}')
        start = time.monotonic()
            
        for iImage in range(self.nImages):
            #print('<AddSamples> iImage= ', iImage)
            for iRadius in range(self.nRadiusesPerImage):
                #print('<AddSamples> iRadius= ', iRadius)
                nVoxelsPerRadius = radIm.countPerRadius[iRadius].item()
                if nVoxelsPerRadius < 1:
                    continue
                    
                nToSelect = min(self.nSamplesPerRadius, nVoxelsPerRadius)
                #print(f'<> {nVoxelsPerRadius=}, {self.nSamplesPerRadius=}, {nToSelect=}')
                
                if nToSelect < 0:
                    print(f'{nToSelect=}')
                    sys.exit()
                if nToSelect < 0:
                    print(f'{nToSelect=} < {nVoxelsPerRadius}')
                    sys.exit()

                selected = random.sample(range(0, nVoxelsPerRadius), nToSelect)
                #print(selected)
                nValid = 0
                for iSample in range(nToSelect):
                    iInRad = selected[iSample]
                    iLine = radIm.rad2PixLine[iRadius,iInRad]
                    iCol = radIm.rad2PixCol[iRadius,iInRad]
                    if mask[iImage,iLine,iCol] > 0:
                        
                        nValid += 1
                        self.samplesLines[iImage,iRadius,iSample] = iLine
                        self.samplesCols[iImage,iRadius,iSample] = iCol
                    else:
                        self.nInvalidSamples[iImage,iRadius] += 1
                        
                self.nValidSamples[iImage,iRadius] = nValid
        if bTiming:
            elapsed = time.monotonic() - start
            print(f"<CSample::AddSamples(> Elapsed time: {elapsed:.3f} seconds")

        
    """
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
            """
               
    def LogToFile(self):
        sfName = Config.LogFileName('Samples.csv')
        with open(sfName, 'w') as file:
            file.write('image, radius, index, line, col, n\n')
            for iImage in range(self.nImages):
                for iRadius in range(self.nRadiusesPerImage):
                    nValid = self.nValidSamples[iImage,iRadius]
                    for iInRad in range(nValid):
                        file.write(f'{iImage}, {iRadius}, {iInRad}')
                        iLine =self.samplesLines[iImage,iRadius,iInRad]
                        iCol = self.samplesCols[iImage,iRadius,iInRad]
                        file.write(f', {iLine}, {iCol}, ')
                        file.write(f', {iInRad}, {self.nInvalidSamples[iImage,iRadius]}\n')
                        
                            
        print('Log file written: ', sfName)
        
        """
        sfName = Config.LogFileName('NSamplesPerRadius.csv')
        with open(sfName, 'w') as file:
            file.write('radius, n\n')
            for i in range(maxRadius):
                file.write(f'{i}, {self.nPerRadius[i]}\n')
        print('Log file written: ', sfName)
        """
        
        
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
