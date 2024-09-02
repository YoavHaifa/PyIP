# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:06:52 2024
Poly Scorer - Compute the score from volume and sample
@author: yoavb
"""

import torch
import time

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
        self.sums = torch.zeros(maxRadius)
        self.counts = torch.zeros(maxRadius)
        self.originalAverage = 0
        self.flatAverage = 0
        self.score = 1000000
        self.count = 0
        
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
        
    
    def Log(self):
        sfName = f'd:/Pylog/ValuePerRadius_{self.count}.csv'
        f = open(sfName,'w')
        f.write("radius, count, average\n")
        for i in range(256):
            f.write(f"{i}, {self.counts[i]}, {self.averages[i]}\n")
        f.close()
        print(f'File written: {sfName}')
        
        
def main():
    print('*** Test Poly Scorer')
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
