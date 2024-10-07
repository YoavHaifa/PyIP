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

maxRadius = 260
maxImages = 1000

verbosity = 1

#sSavedVolume = 'BP_PolyAI_Output_width512_height512_save4000.float.rvol'

class CPolyScorer:
    """
    """
    def __init__(self):
        """
        """
        self.originalAverage = 0
        self.flatAverage = 0
        self.score = 1000000
        self.count = 0
        self.nImages = 0
        self.iImage = 0

    def ScoreImage(self, image, samplesLines, samplesCols, nValid):
        nRadiuses = 0
        avgs = torch.zeros(maxRadius)
        for iRad in range(self.nRadiusesPerImage):
            #print(f'{iRad=}')
            nValidPerRad = nValid[iRad]
            if nValidPerRad < 1:
                break
                
            #print(f'{samplesLines[iRad]=}')
            #print(f'{samplesCols[iRad]=}')
            fastGet = image[samplesLines[iRad,0:nValidPerRad],samplesCols[iRad,0:nValidPerRad]]
            fastAvg = fastGet.mean()
           
            avgs[iRad] = fastAvg
            nRadiuses += 1
            self.sum += fastGet.sum()
            self.nSummed += nValidPerRad
            
        if nRadiuses < 8:
            return
        
        relevant = avgs[0:nRadiuses]
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
        Config.Start('ScoreAllImages')
        self.sum = 0
        self.nSummed = 0
        self.nImagesScored = 0
        #self.stdPerImage = torch.zeros(self.nImages)
        self.radsPerImage = torch.zeros(self.nImages)
        self.rangePerImageH = torch.zeros(self.nImages)
        self.rangePerImageQ = torch.zeros(self.nImages)
        self.averagePerImage = torch.zeros(self.nImages)
        self.deltaAveragePerImage = torch.zeros(self.nImages) # Relative to original Volume Average

        nToScore = min(maxImages, self.nImages)
        for iIm in range(nToScore):
            #print(f'{iIm=}')
            self.iImage = iIm
            self.ScoreImage(vol.pImages[iIm], sample.samplesLines[iIm], sample.samplesCols[iIm], sample.nValidSamples[iIm])
            
        #relevant = self.stdPerImage[0:self.nImagesScored]
        #self.flatnessScore = relevant.mean()
        #relevantRange = self.rangePerImage[0:self.nImagesScored]
        self.rangeScoreH = self.rangePerImageH[0:self.nImagesScored].mean()
        self.rangeScoreQ = self.rangePerImageQ[0:self.nImagesScored].mean()
        Config.End('ScoreAllImages')
        #stdSqr = torch.square(relevant)
        #self.flatnessScore = stdSqr.mean()
        
    def ComputeAverage(self):
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
        elif self.flatAverage == 0:
            self.flatAverage = self.average
            print(f'<Score> Flat Average is {self.flatAverage}')
        else:
            self.averageDiff = self.flatAverage - self.average

        if self.flatAverage != 0:
            for i in range(self.nImagesScored):
                self.deltaAveragePerImage[i] = abs(self.averagePerImage[i] - self.flatAverage)
        relevanDelatAvg = self.deltaAveragePerImage[0:self.nImagesScored]
        self.averageScore = relevanDelatAvg.mean()

        
    def Score(self, sfVolume, sample):
        Config.Start('Score')
        if verbosity > 1 or self.count < 3:
            print('<CPolyScorer::Score> ', sfVolume)
        vol = CVolume('scoredVol', sfVolume)
        self.radIm = sample.radIm
        
        self.nImages = sample.nImages
        self.nRadiusesPerImage = sample.nRadiusesPerImage
        self.ScoreAllImages(vol, sample)
        
        self.ComputeAverage()
        
        #self.score = self.flatnessScore + self.rangeScore + self.averageScore
        self.score = self.rangeScoreH + self.rangeScoreQ + self.averageScore
        self.score = self.score.item()
            
        self.count += 1
        if verbosity > 1:
            print(f'<CPolyScorer::Score> {self.count} range {self.rangeScoreH}, range {self.rangeScoreQ}, Avg {self.averageScore} ==> Score {self.score}')
        Config.End('Score')
        return self.score
    
    def WriteTitle(self, f):
        f.write(', rangeH, rangeQ, avg, score')
    
    def WriteScore(self, f):
        sScore = f', {self.rangeScoreH}, {self.rangeScoreQ}, {self.averageScore}, {self.score}'
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
        f.write(f',,,,,Flat Avg, {self.flatAverage}\n')
        f.write(f',,,,,Last Avg, {self.average}\n')
        f.write(f',,,,,rangeH, {self.rangeScoreH}, rangeQ, {self.rangeScoreQ}, Avg, {self.averageScore}\n')
        f.write(f',,,,,score, {self.score}\n')
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
    
    score = scorer.Score(Config.sfVolumeNominal, sample)
    scorer.Log()
    print(f'{score=}')
    
    score = scorer.Score(Config.sfVolumeAi, sample)
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
