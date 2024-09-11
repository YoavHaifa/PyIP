# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:06:52 2024
Poly Scorer - Compute the score from volume and sample
@author: yoavb
"""

import torch
import time
import sys

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Sample import CSample

maxRadius = 260

verbosity = 1

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
            
            """
            valSum = 0
            for i in range(nValidPerRad):
                iLine = samplesLines[iRad,i]
                iCol = samplesCols[iRad,i]
                valSum += image[iLine, iCol]
            avg = valSum / nValidPerRad
            
            if avg != fastAvg:
                diff = abs(avg - fastAvg)
                if diff > 0.001:
                    print(f'{avg=:.9f} != {fastAvg=:.9f}')
                    sys.exit()
             """
           
            avgs[iRad] = fastAvg
            nRadiuses += 1
            self.sum += fastGet.sum()
            self.nSummed += nValidPerRad
            
        if nRadiuses < 2:
            return
        
        relevant = avgs[0:nRadiuses]
        self.stdPerImage[self.nImagesScored] = torch.std(relevant).item()
        self.radsPerImage[self.nImagesScored] = nRadiuses
        
        #Compute Range
        self.RangePerImage[self.nImagesScored] = relevant.max() - relevant.min()
        
        """
        #Compute moment
        nMoment = nRadiuses - 5
        if nMoment > 0:
            base = relevant[2:5].mean()
            sumMoment = 0
            for i in range(nMoment):
                iInVec = i + 5
                sumMoment += abs(relevant[iInVec] - base) * iInVec
            self.momentPerImage[self.nImagesScored] = (sumMoment / nMoment) * 0.01
            """
            
        self.nImagesScored += 1

    def ScoreAllImages(self, vol, sample):
        Config.Start('ScoreAllImages')
        self.sum = 0
        self.nSummed = 0
        self.nImagesScored = 0
        self.stdPerImage = torch.zeros(self.nImages)
        self.radsPerImage = torch.zeros(self.nImages)
        self.RangePerImage = torch.zeros(self.nImages)

        for iIm in range(self.nImages):
            #print(f'{iIm=}')
            self.ScoreImage(vol.pImages[iIm], sample.samplesLines[iIm], sample.samplesCols[iIm], sample.nValidSamples[iIm])
            
        relevant = self.stdPerImage[0:self.nImagesScored]
        self.flatnessScore = relevant.mean()
        relevantRange = self.RangePerImage[0:self.nImagesScored]
        self.rangeScore = relevantRange.mean()
        Config.End('ScoreAllImages')
        #stdSqr = torch.square(relevant)
        #self.flatnessScore = stdSqr.mean()
        
    def ComputeAverage(self):
        if self.nSummed > 0:
            self.average = self.sum / self.nSummed
        else:
            self.average = 0

        self.averageDiff = 0
        if self.originalAverage == 0:
            self.originalAverage = self.average
            print(f'<Score> Original Average is {self.originalAverage}')
        elif self.flatAverage == 0:
            self.flatAverage = self.average
            print(f'<Score> Flat Average is {self.flatAverage}')
        else:
            self.averageDiff = self.flatAverage - self.average
        
    def Score(self, sfVolume, sample):
        Config.Start('Score')
        if verbosity > 1 or self.count < 5:
            print('<CPolyScorer::Score> ', sfVolume)
        vol = CVolume('scoredVol', sfVolume)
        self.radIm = sample.radIm
        
        self.nImages = sample.nImages
        self.nRadiusesPerImage = sample.nRadiusesPerImage
        self.ScoreAllImages(vol, sample)
        
        self.ComputeAverage()
        
        self.score = self.flatnessScore + self.momentScore + (self.averageDiff / 2) ** 2
        self.score = self.score.item()
            
        self.count += 1
        if verbosity > 1:
            print(f'<CPolyScorer::Score> {self.count} STD {self.flatnessScore}, Mom {self.momentScore}, Average Diff {self.averageDiff} ==> Score {self.score}')
        Config.End('Score')
        return self.score
            
        

    """
    def Score(self, sfVolume, sample):
        if verbosity > 1:
            print('<CPolyScorer::Score> ', sfVolume)
        self.sums = torch.zeros(maxRadius)
        self.counts = torch.zeros(maxRadius)
        vol = CVolume('scoredVol', sfVolume)
        
        # Get all samples
        for i in range(sample.n):
            iRadius = sample.radiuses[i]
            value = vol.pImages[sample.images[i],sample.lines[i],sample.cols[i]]
            self.sums[iRadius] += value
            self.counts[iRadius] += 1
        
        onesVector = torch.ones(maxRadius)
        self.counts = torch.max(self.counts,onesVector)
        self.averages = self.sums.div(self.counts)
        
        # Log
        
        self.average = self.averages[5:245].mean()
        
        relevant = self.averages[0:254]
        self.std = torch.std(relevant).item()
        self.averageDiff = 0
        if self.originalAverage == 0:
            self.originalAverage = self.average
            print(f'<Score> Original Average is {self.originalAverage}')
        elif self.flatAverage == 0:
            self.flatAverage = self.average
            print(f'<Score> Flat Average is {self.flatAverage}')
        else:
            self.averageDiff = self.flatAverage - self.average
        self.score = self.std + (self.averageDiff / 10) ** 2
            
        self.count += 1
        if verbosity > 1:
            print(f'<CPolyScorer::Score> {self.count} STD {self.std}, Average Diff {self.averageDiff} ==> Score {self.score}')
        return self.score
    """
        
    
    def Log(self):
        sfName = f'd:/Pylog/ScorerValuePerImage_{self.count}.csv'
        f = open(sfName,'w')
        f.write("image, radiuses, std\n")
        for i in range(self.nImagesScored):
            f.write(f"{i}, {self.radsPerImage[i]}, {self.stdPerImage[i]}\n")
        f.close()
        print(f'File written: {sfName}')
        
        
def main():
    global verbosity
    print('*** Test Poly Scorer')
    verbosity = 5
    Config.OnInitRun()
    radIm = CRadiusImage()
    vol = CVolume('nominalVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(vol)
    sample = CSample(maskVol, radIm)
    scorer = CPolyScorer()
    start = time.monotonic()
    score = scorer.Score(Config.sfVolumeNominal, sample)
    print(f'{score=}')
    elapsed = time.monotonic() - start
    print(f"<CSample::AddSamples(> Elapsed time: {elapsed:.3f} seconds")
    scorer.Log()
    score = scorer.Score(Config.sfVolumeAi, sample)
    print(f'{score=}')
    
     

if __name__ == '__main__':
    main()
