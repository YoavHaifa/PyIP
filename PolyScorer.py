# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:06:52 2024
Poly Scorer - Compute the score from volume and sample
@author: yoavb
"""

import torch
#import time
#import sys

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Sample import CSample
from Log import Start, End, Log

maxRadius = 260
maxImages = 280
HIGH_SCORE = 1000000

verbosity = 1

#sSavedVolume = 'BP_PolyAI_Output_width512_height512_save4000.float.rvol'

class CPolyScorer:
    """
    """
    def __init__(self):
        """
        """
        self.originalAverage = 0
        self.oldScore = HIGH_SCORE
        self.newScore = HIGH_SCORE
        self.count = 0
        self.nImages = 0
        self.iImage = 0
        self.iImageMaxDev = -1
        self.iRadMaxDev = -1
        self.targetAverage = 0
        self.bTargetDefined = False
        self.iRingTargeted = torch.zeros([maxImages, maxRadius])

    def ScoreImage(self, image, samplesLines, samplesCols, nValid, iImage):
        nRadiuses = 0
        avgs = torch.zeros(maxRadius)
        for iRad in range(self.nRadiusesPerImage):
            #print(f'{iRad=}')
            nValidPerRad = nValid[iRad]
            """
            if iImage == 92 and iRad == 0:
                print(f'image {iImage} iRad {iRad} {nValidPerRad=}')
                print(f'{samplesLines[0,0:nValidPerRad]=}')
                print(f'{samplesCols[0,0:nValidPerRad]=}')
                """
            if nValidPerRad < 1:
                break
                
            #print(f'{samplesLines[iRad]=}')
            #print(f'{samplesCols[iRad]=}')
            fastGet = image[samplesLines[iRad,0:nValidPerRad],samplesCols[iRad,0:nValidPerRad]]
            """
            if iImage == 92 and iRad == 0:
                print(f'{fastGet=}')
                """
            fastAvg = fastGet.mean()
           
            avgs[iRad] = fastAvg
            nRadiuses += 1
            self.sum += fastGet.sum()
            self.nSummed += nValidPerRad
            
        if nRadiuses < 8:
            return
        
        relevant = avgs[0:nRadiuses]
        if self.bTargetDefined:
            self.deltaAveragePerRing[iImage,0:nRadiuses] = relevant - self.targetAverage
            """
            if iImage >= 90 and iImage <= 100:
                print(f'<ScoreImage> {iImage} - {relevant[0]}')
                """
        #self.stdPerImage[self.nImagesScored] = torch.std(relevant).item()
        self.radsPerImage[self.nImagesScored] = nRadiuses
        
        #Compute Range
        relevantSorted, _  = relevant.sort()
        iHalf = int(nRadiuses / 2)
        low = relevantSorted[0:iHalf].mean()
        high = relevantSorted[iHalf:nRadiuses].mean()
        self.rangePerImageH[self.nImagesScored] = high - low
        
        iQuarter = int(nRadiuses / 4)
        lowQ = relevantSorted[0:iQuarter].mean()
        highQ = relevantSorted[nRadiuses-iQuarter:nRadiuses].mean()
        self.rangePerImageQ[self.nImagesScored] = highQ - lowQ
        
        #Compute average along radiuses - to avoid level change
        self.averagePerImage[self.nImagesScored] = relevant.mean()
            
        self.nImagesScored += 1

    def ScoreAllImages(self, vol, sample):
        Start('ScoreAllImages')
        self.nPeeled = 0
        self.sum = 0
        self.nSummed = 0
        self.nImagesScored = 0
        #self.stdPerImage = torch.zeros(self.nImages)
        self.radsPerImage = torch.zeros(self.nImages)
        self.rangePerImageH = torch.zeros(self.nImages)
        self.rangePerImageQ = torch.zeros(self.nImages)
        self.averagePerImage = torch.zeros(self.nImages)
        self.deltaAveragePerImage = torch.zeros(self.nImages) # Relative to original Volume Average
        self.deltaAveragePerRing = torch.zeros([self.nImages, maxRadius])
        self.absDeltaAveragePerRing = torch.zeros([self.nImages, maxRadius])

        nToScore = min(maxImages, self.nImages)
        for iIm in range(nToScore):
            #print(f'{iIm=}')
            self.iImage = iIm
            self.ScoreImage(vol.pImages[iIm], sample.samplesLines[iIm], sample.samplesCols[iIm], sample.nValidSamples[iIm], iIm)
            
        #relevant = self.stdPerImage[0:self.nImagesScored]
        #self.flatnessScore = relevant.mean()
        #relevantRange = self.rangePerImage[0:self.nImagesScored]
        self.rangeScoreH = self.rangePerImageH[0:self.nImagesScored].mean()
        self.rangeScoreQ = self.rangePerImageQ[0:self.nImagesScored].mean()
        End('ScoreAllImages')
        #stdSqr = torch.square(relevant)
        #self.flatnessScore = stdSqr.mean()
        
    def OldComputeAverage(self):
        if self.nSummed > 0:
            self.average = self.sum / self.nSummed
            #print(f'Average {self.average} = sum {self.sum} / n {self.nSummed}')
        else:
            self.average = 0

        self.averageDiff = 0
        self.averageScore = 0
        
        if self.originalAverage == 0:
            self.originalAverage = self.average
            print(f'<Score> Original Average is {self.originalAverage}')
        elif not self.bTargetDefined:
            self.targetAverage = self.average
            self.bTargetDefined = True
            print(f'<Score> Flat Average is {self.targetAverage}')
        else:
            self.averageDiff = self.targetAverage - self.average

        if self.bTargetDefined:
            for i in range(self.nImagesScored):
                self.deltaAveragePerImage[i] = abs(self.averagePerImage[i] - self.targetAverage)

        relevanDelatAvg = self.deltaAveragePerImage[0:self.nImagesScored]
        self.averageScore = relevanDelatAvg.mean()
        
    def NewScore(self, sfVolume, sample):
        if not self.bTargetDefined:
            self.newScore = HIGH_SCORE
            return
            
        Start('NewScore')
        if verbosity > 1:
            print('<CPolyScorer::NewScore> ', sfVolume)
        Log(f'<CPolyScorer::NewScore> {sfVolume}')
        vol = CVolume('scoredVol', sfVolume)
        self.radIm = sample.radIm
        
        self.nImages = sample.nImages
        self.nRadiusesPerImage = sample.nRadiusesPerImage
        self.ScoreAllImages(vol, sample)
        
        if self.count > 0:
            self.CheckChangeInMaxPos()
        self.FindMaxDeviation()
        self.count += 1
        if verbosity > 1:
            print(f'<CPolyScorer::NewScore> {self.count} Score {self.newScore}')
        End('NewScore')
        return self.newScore
        
    def OldScore(self, sfVolume, sample):
        Start('OldScore')
        if verbosity > 1:
            print('<CPolyScorer::OldScore> ', sfVolume)
        Log(f'<CPolyScorer::Score> {sfVolume}')
        vol = CVolume('scoredVol', sfVolume)
            
        self.radIm = sample.radIm
        
        self.nImages = sample.nImages
        self.nRadiusesPerImage = sample.nRadiusesPerImage
        self.ScoreAllImages(vol, sample)
        
        self.OldComputeAverage()
        
        #self.oldScore = self.flatnessScore + self.rangeScore + self.averageScore
        self.oldScore = self.rangeScoreH + self.rangeScoreQ + self.averageScore
        self.oldScore = self.oldScore.item()

        self.count += 1
        s = f'<CPolyScorer::OldScore> {self.count} rangeH {self.rangeScoreH}, rangeQ {self.rangeScoreQ}, Avg {self.averageScore} ==> Score {self.oldScore}'
        if verbosity > 1:
            print(s)
        Log(s)
        End('OldScore')
        return self.oldScore
    
    def CheckChangeInMaxPos(self):
        if self.iImageMaxDev >= 0 and self.iRadMaxDev >= 0:
            changed = self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev].item()
            if changed == self.maxDev:
                s = f'<FindMaxDeviation> Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {self.maxDev:.3f} SAME'
            else:
                s = f'<FindMaxDeviation> Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {self.maxDev:.3f} ==> {changed:.3f}'
            Log(s)

    def FindMaxDeviation(self):
        self.avgDev = self.deltaAveragePerRing.mean().item()
        #self.DumpDeviation()
        self.PeelDeviation()
        if self.count <= 100:
            self.DumpDeviation()
        self.absDeltaAveragePerRing = self.deltaAveragePerRing.abs()
        prev = self.newScore
        self.newScore = self.absDeltaAveragePerRing.mean().item()
        diff = self.newScore - prev
        
        Log(f'<<CheckChangeInMaxPos>> score {prev} ==> {self.newScore} (diff {diff})')

        threshold = float(max(1, self.count - 1000))
        absNew = torch.where(self.iRingTargeted < threshold, self.absDeltaAveragePerRing, 0)

        iMax = absNew.argmax().item()
        self.iImageMaxDev = int(iMax / maxRadius)
        self.iRadMaxDev = iMax % maxRadius
        self.maxDev = self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev].item()
        self.iRingTargeted[self.iImageMaxDev,self.iRadMaxDev] = self.count
        
        #find width of current ring
        self.maxDevWidth = 1 
        thresh = self.maxDev / 10
        if self.maxDev > 0:
            for iTry in range(9):
                delta = iTry + 1
                if self.iImageMaxDev+delta < self.nImages:
                    if self.deltaAveragePerRing[self.iImageMaxDev+delta,self.iRadMaxDev] < threshold:
                        break
                if self.iImageMaxDev-delta >= 0:
                    if self.deltaAveragePerRing[self.iImageMaxDev-delta,self.iRadMaxDev] < threshold:
                        break
                if self.iRadMaxDev+delta < maxRadius:
                    if self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev+delta] < threshold:
                        break
                if self.iRadMaxDev-delta >= 0:
                    if self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev-delta] < threshold:
                        break
                self.maxDevWidth += 1 
        else:
            for iTry in range(9):
                delta = iTry + 1
                if self.iImageMaxDev+delta < self.nImages:
                    if self.deltaAveragePerRing[self.iImageMaxDev+delta,self.iRadMaxDev] > threshold:
                        break
                if self.iImageMaxDev-delta >= 0:
                    if self.deltaAveragePerRing[self.iImageMaxDev-delta,self.iRadMaxDev] > threshold:
                        break
                if self.iRadMaxDev+delta < maxRadius:
                    if self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev+delta] > threshold:
                        break
                if self.iRadMaxDev-delta >= 0:
                    if self.deltaAveragePerRing[self.iImageMaxDev,self.iRadMaxDev-delta] > threshold:
                        break
                self.maxDevWidth += 1 

        #if verbosity > 1 or self.count < 10 or self.count % 100 == 0:
        s = f'<FindMaxDeviation> Average Deviation {self.avgDev:.3f} Max at [{self.iImageMaxDev},{self.iRadMaxDev}] = {self.maxDev:.3f} width {self.maxDevWidth}'
        if verbosity > 1:
            print(s)
        Log(s)

    def PeelDeviation(self):
        self.nPeeled += 1
        self.deltaAveragePerRing[0] = 0
        self.deltaAveragePerRing[-1] = 0
        size = self.deltaAveragePerRing.shape;
        nImages = size[0]
        nRadiuses = size[1]
        oldDelta = self.deltaAveragePerRing.clone().detach()
        for iIm in range (nImages-2):
            iImage = iIm + 1
            for iR in range(nRadiuses-1):
                if oldDelta[iImage-1,iR] == 0:
                    self.deltaAveragePerRing[iImage,iR] = 0
                if oldDelta[iImage+1,iR] == 0:
                    self.deltaAveragePerRing[iImage,iR] = 0
                if oldDelta[iImage,iR+1] == 0:
                    self.deltaAveragePerRing[iImage,iR] = 0
        

    def DumpDeviation(self):
        size = self.deltaAveragePerRing.shape;
        sfName = f'd:/Dump/ScoreDEV_count{self.count}_width{size[1]}_height{size[0]}_p{self.nPeeled}_dzoom2.float.rmat'
        npMat = self.deltaAveragePerRing.numpy()
        with open (sfName, 'wb') as file:
            file.write(npMat.tobytes())
        if verbosity > 1:
            print(f'<DumpDeviation> saved: {sfName}')
        Log(f'<DumpDeviation> saved: {sfName}')
    
    def WriteTitle(self, f):
        f.write(', rangeH, rangeQ, maxDev, score')
    
    def WriteScore(self, f):
        sScore = f', {self.rangeScoreH}, {self.rangeScoreQ}, {self.maxDev}, {self.newScore}'
        f.write(sScore)
        return sScore

    def Log(self):
        f, sfName = Config.OpenLogGetName(f'ScorerValuePerImage_{self.count}.csv')
        f.write("image, radiuses, std, range, average, delta avg\n")
        for i in range(self.nImagesScored):
            f.write(f'{i}, {self.radsPerImage[i]}')
            #f.write(f', {self.stdPerImage[i]}')
            f.write(f', {self.rangePerImageH[i]}')
            f.write(f', {self.rangePerImageQ[i]}')
            f.write(f', {self.averagePerImage[i]}')
            f.write(f', {self.deltaAveragePerImage[i]}')
            f.write('\n')
            
        f.write(f',,,,,Original Avg, {self.originalAverage}\n')
        f.write(f',,,,,Flat Avg, {self.targetAverage}\n')
        f.write(f',,,,,Last Avg, {self.average}\n')
        f.write(f',,,,,rangeH, {self.rangeScoreH}, rangeQ, {self.rangeScoreQ}, maxDev, {self.maxDev}\n')
        f.write(f',,,,,score, {self.newScore}\n')
        f.close()
        print(f'File written: {sfName}')
        
        
def main():
    global verbosity
    print('*** Test Poly Scorer')
    verbosity = 5
    Config.OnInitRun(sSpecialVolDir='D:/PolyCalib/Volumns')
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
    
    score = scorer.NewScore(Config.sfVolumeAi, sample)
    scorer.Log()
    print(f'{score=}')
    
    """
    start = time.monotonic()
    score = scorer.Score(sSavedVolume, sample)
    scorer.Log()
    print(f'{score=}')
    elapsed = time.monotonic() - start
    print(f"<scorer.Score> Elapsed time for last scoring: {elapsed:.3f} seconds")
    """
    
     

if __name__ == '__main__':
    main()
