# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:46:22 2024
Deviation Raster - Deviation from target for each image and radius
@author: yoavb
"""

#import torch

#import Config
#from Volume import CVolume
#from RadiusImage import CRadiusImage
#from MaskVolume import CMaskVolume
#from Sample import CSample
#from PolyScorer import CPolyScorer
from Log import Log

maxRadius = 260
maxImages = 280

verbosity = 1
count = 0

class CDevRaster:
    """
    """
    def __init__(self, targetLevel, averagePerImRad):
        """
        """
        global count
        count += 1
        self.dev = averagePerImRad - targetLevel
        size = self.dev.shape;
        self.nImages = size[0]
        self.nRadiuses = size[1]
        self.ComputeScore()
        self.FindMaxDeviation()

    def ComputeScore(self):
        self.avgDev = self.dev.mean().item()
        self.PeelDeviation()
        if count <= 100:
            self.Dump()
        self.absDev = self.dev.abs()
        self.score = self.absDev.mean().item()
        

    def PeelDeviation(self):
        #self.nPeeled += 1
        self.dev[0] = 0
        self.dev[-1] = 0
        oldDelta = self.dev.clone().detach()
        for iIm in range (self.nImages-2):
            iImage = iIm + 1
            for iR in range(self.nRadiuses-1):
                if oldDelta[iImage-1,iR] == 0:
                    self.dev[iImage,iR] = 0
                if oldDelta[iImage+1,iR] == 0:
                    self.dev[iImage,iR] = 0
                if oldDelta[iImage,iR+1] == 0:
                    self.dev[iImage,iR] = 0

    def Dump(self):
        size = self.dev.shape;
        sfName = f'd:/Dump/ScoreDEV_count{count}_{count}_width{size[1]}_height{size[0]}_dzoom2.float.rmat'
        npMat = self.dev.numpy()
        with open (sfName, 'wb') as file:
            file.write(npMat.tobytes())
        if verbosity > 1:
            print(f'<DumpDeviation> saved: {sfName}')
        Log(f'<DumpDeviation> saved: {sfName}')
        

    def FindMaxDeviation(self):
        #threshold = float(max(1, self.count - 1000))
        #absNew = torch.where(self.iRingTargeted < threshold, self.absDeltaAveragePerRing, 0)
        #iMax = absNew.argmax().item()
        iMax = self.absDev.argmax().item()
        self.iImageMaxDev = int(iMax / maxRadius)
        self.iRadMaxDev = iMax % maxRadius
        self.maxDev = self.dev[self.iImageMaxDev,self.iRadMaxDev].item()
        #self.iRingTargeted[self.iImageMaxDev,self.iRadMaxDev] = self.count
        
        #find width of current ring
        self.maxDevWidth = 1 
        threshold = self.maxDev / 10
        if self.maxDev > 0:
            for iTry in range(9):
                delta = iTry + 1
                if self.iImageMaxDev+delta < self.nImages:
                    if self.dev[self.iImageMaxDev+delta,self.iRadMaxDev] < threshold:
                        break
                if self.iImageMaxDev-delta >= 0:
                    if self.dev[self.iImageMaxDev-delta,self.iRadMaxDev] < threshold:
                        break
                if self.iRadMaxDev+delta < maxRadius:
                    if self.dev[self.iImageMaxDev,self.iRadMaxDev+delta] < threshold:
                        break
                if self.iRadMaxDev-delta >= 0:
                    if self.dev[self.iImageMaxDev,self.iRadMaxDev-delta] < threshold:
                        break
                self.maxDevWidth += 1 
        else:
            for iTry in range(9):
                delta = iTry + 1
                if self.iImageMaxDev+delta < self.nImages:
                    if self.dev[self.iImageMaxDev+delta,self.iRadMaxDev] > threshold:
                        break
                if self.iImageMaxDev-delta >= 0:
                    if self.dev[self.iImageMaxDev-delta,self.iRadMaxDev] > threshold:
                        break
                if self.iRadMaxDev+delta < maxRadius:
                    if self.dev[self.iImageMaxDev,self.iRadMaxDev+delta] > threshold:
                        break
                if self.iRadMaxDev-delta >= 0:
                    if self.dev[self.iImageMaxDev,self.iRadMaxDev-delta] > threshold:
                        break
                self.maxDevWidth += 1 

        #if verbosity > 1 or self.count < 10 or self.count % 100 == 0:
        s = f'<FindMaxDeviation> Average Deviation {self.avgDev:.3f} Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {self.maxDev:.3f} width {self.maxDevWidth}'
        if verbosity > 1:
            print(s)
        Log(s)

    
    def CheckChangeInMaxPos(self):
        if self.iImageMaxDev >= 0 and self.iRadMaxDev >= 0:
            changed = self.dev[self.iImageMaxDev,self.iRadMaxDev].item()
            if changed == self.maxDev:
                s = f'<CheckChangeInMaxPos> Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {self.maxDev:.3f} SAME'
            elif changed < self.maxDev:
                self.bMaxImproved = True
                self.prevMaxDev = self.maxDev
                self.maxDev = changed
                s = f'<CheckChangeInMaxPos> Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {changed:.3f} BETTER < {self.maxDev:.3f}'
            else:
                s = f'<CheckChangeInMaxPos> Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {changed:.3f} WORSE > {self.maxDev:.3f}'
            Log(s)

def main():
    global verbosity
    verbosity = 5
    
    print('*** Test Dev Raster - TBD')
    """
    Config.SetSpecialVolDir('D:/PolyCalib/Volumns')
    Config.OnInitRun()
    radIm = CRadiusImage()
    vol = CVolume('nominalVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(vol)
    sample = CSample(maskVol, radIm)
    scorer = CPolyScorer()
    
    oldScore = scorer.OldScore(Config.sfVolumeNominal, sample)
    scorer.Log()
    print(f'{oldScore=}')
    
    oldScore = scorer.OldScore(Config.sfVolumeAi, sample)
    scorer.Log()
    print(f'{oldScore=}')

    score = scorer.ComputeNewScoreOfVolume(Config.sfVolumeAi, sample)
    scorer.Log()
    print(f'{score=}')
    #devRaster = CDevRaster()
    """

if __name__ == '__main__':
    main()
