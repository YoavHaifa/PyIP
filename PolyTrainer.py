# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 02:12:48 2024
Poly Trainer Class - responsible for the complete process of training a poly table
@author: yoavb
"""

#import os
from os import path
import random
#import torch
#import time
import sys

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Sample import CSample
from PolyScorer import CPolyScorer
from RunRecon import RunAiRecon, RunOriginalRecon
from PolyTable import CPolyTables
from Utils import GetAbortFileName
from Log import CLog

nDetectors = 688
nRows = 192
nLayers = 3
deltaSave = 100
deltaSample = 100

BIG_SCORE = 1000000.0

class CPolyTrainer:
    """
    """
    def __init__(self):
        """
        """
        Config.gLog = CLog('Trainer')
        self.originalScore = BIG_SCORE
        self.firstScore = BIG_SCORE
        self.bestScore = BIG_SCORE
        self.nImprovements = 0
        self.radIm = CRadiusImage()
        
        RunOriginalRecon()
        
        self.originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
        self.maskVol = CMaskVolume(self.originalVol)
        self.scorer = CPolyScorer()
        self.iSample = 0
        self.sample = CSample(self.maskVol, self.radIm)

        self.iTry = 0
        self.nBetter = 0
        self.nBetter0 = 0
        self.nBetter1 = 0
        
        self.nBetterPerSample = 0;
        self.nNotBetterPerSample = 0
        self.nFailedSamples = 0
        
        self.nNotBetterConsecutive = 0
        self.nMaxBetter = 10000
        
        self.sfAbort = GetAbortFileName()
        self.originalScore = self.scorer.Score(Config.sfVolumeNominal, self.sample)
        self.sNext = 'random'
        self.nMultiply = 0

    def OpenCsv(self, sfName):
        f, sfName = Config.OpenLogGetName(sfName)
        #f.write('xrt, fRow, lRow, fCol, lCol, delta, score, diff\n')
        #f.write(f'01, 0, {nRows}, 0, {nDetectors}, 0, {self.bestScore}, 0\n')
        
        #Write Title for all columns
        f.write('xrt, sample, delta, step')
        self.scorer.WriteTitle(f)
        f.write(', diff\n')
       
        f.write('0_1, 0, 0')
        self.scorer.WriteScore(f)
        f.write(', 0\n')
        f.close()
        return sfName
                
    def RunFlatTable(self):
        self.tableGenerator = CPolyTables() # Prepares flat table
        RunAiRecon()
        self.firstScore = self.scorer.Score(Config.sfVolumeAi, self.sample)
        self.bestScore = self.firstScore
        
        self.sfAllCsv = self.OpenCsv('Training_All.csv')
        self.sfImproveCsv = self.OpenCsv('Training_Improve.csv')
        
        Config.SaveAiVolume(0)
        #sfSave = 'd:\Dump\BP_PolyAI_Output_width512_height512_save0.float.dat'
        #TryRename(Config.sfVolumeAi, sfSave)
        
    def CreateNewSampe(self):
        nAll = self.nBetterPerSample + self.nNotBetterPerSample
        print(f'<CreateNewSampe> better {self.nBetterPerSample} / {nAll}')
        self.iSample += 1
        if self.nBetterPerSample > 0:
            self.nFailedSamples = 0
        else:
            self.nFailedSamples += 1
            if self.nFailedSamples > 1:
                print(f'<CreateNewSampe> {self.nFailedSamples=} - aborting...')
                sys.exit()
                
        self.sample = CSample(self.maskVol, self.radIm)
        prevBest = self.bestScore
        self.firstScore = self.scorer.Score(Config.sfVolumeAi, self.sample)
        self.bestScore = self.firstScore
            
        self.nBetterPerSample = 0;
        self.nNotBetterPerSample = 0

        print(f'<CreateNewSampe> {self.iSample} best {prevBest} --> {self.bestScore}')

    def SelectNextStep(self):
        #print(f'<SelectNextStep> {self.sNext=}')
        if self.sNext == 'random':
            self.iTable = random.randint(0, 1)
            delta = self.tableGenerator.TryRandomTableStep(self.iTable)
            print(f'<TryStep> {self.iTry} RANDOM {self.tableGenerator.sLast} {delta=}')
        elif self.sNext == 'OnFailure':
            delta = self.tableGenerator.TryOnFailure()
            print(f'<TryStep> {self.iTry} ON_FAIL {self.tableGenerator.sLast} {delta=}')
        elif self.sNext == 'OnSuccess':
            delta = self.tableGenerator.TryOnSuccess()
            print(f'<TryStep> {self.iTry} ON_GOOD {self.tableGenerator.sLast} {delta=}')
        else:
            print(f'<SelectNextStep> Illegal step {self.sNext}')
            self.sNext = 'random'
            sys.exit()
            return self.SelectNextStep()
    
        return delta

    def TryStep(self):
        Config.Start('TryStep')
        self.iTry += 1 
        delta = self.SelectNextStep()
        RunAiRecon()
        score = self.scorer.Score(Config.sfVolumeAi, self.sample)
        diff = self.bestScore - score
        fAll = open(self.sfAllCsv,'a')
        fAll.write(f'{self.iTable}, {self.iSample}, {delta}, {self.tableGenerator.sLast}')
        self.scorer.WriteScore(fAll)
        fAll.write(f', {diff}')

        if score < self.bestScore:
            Config.Start('OnBetter')
            if self.iTable == 0:
                self.nBetter0 += 1
                self.tableGenerator.SaveBetter(0, self.nBetter0)
            else:
                self.nBetter1 += 1
                self.tableGenerator.SaveBetter(1, self.nBetter1)
                
            self.nBetter += 1
            print(f'+++ === >>> New table better {self.nBetter} {score=} < {self.bestScore}')
            self.bestScore = score
            #centralImage.fName = centralImage.fName.replace('Central_Image', f'Central_Image_step{self.nBetter}')
            #centralImage.WriteToFile()
            fAll.write(', *\n')
            fImp = open(self.sfImproveCsv, 'a')
            #fImp.write(f'{iTable}, {iFirstRow}, {iRowAfter}, {iFirstCol}, {iColAfter}, {delta}, {score}, {diff}\n')
            fImp.write(f'{self.iTable}, {self.iSample}, {delta}')
            #fImp.write(f'{self.iTable}, {self.iSample}, {delta*100:.6f}')
            sScore = self.scorer.WriteScore(fImp)
            #fImp.write(f', {diff}')
            fImp.write('\n')
            fImp.close()
            Config.Log(sScore)
            
            if self.nBetter % deltaSample == 0:
                self.CreateNewSampe()
            if self.nBetter % deltaSave == 0:
                Config.SaveAiVolume(self.nBetter)
                #sfSave = f'd:\Dump\BP_PolyAI_Output_width512_height512_save{self.nBetter}.float.dat'
                #TryRename(sVolumeFileNameAi, sfSave)
            self.nNotBetterConsecutive = 0
            if self.nMultiply < 2:
                self.sNext = 'OnSuccess'
                self.nMultiply += 1 
            else:
                self.sNext = 'random'
                self.nMultiply = 0 
                
            self.nBetterPerSample += 1
            Config.End('OnBetter')
            Config.End('TryStep')
            return True
        
        fAll.write('\n')
        fAll.close()
        self.nNotBetterConsecutive += 1
        self.nNotBetterPerSample += 1
        self.tableGenerator.RestoreTable(self.iTable)
        print(f'--- <<< New table NOT better {self.nNotBetterConsecutive}')
        if self.nNotBetterPerSample % 10 == 0:
            print(f'--- <<< New table NOT better {score=} >= {self.bestScore} ({diff=})')
        #if self.sNext == 'random':
        #    self.sNext = 'OnFailure'
        #else:
        #    self.sNext = 'random'
        self.sNext = 'random'
        self.nMultiply = 0 
        Config.End('TryStep')
        if self.nNotBetterPerSample % deltaSample == 0:
            self.CreateNewSampe()
        return False
    
    def Train(self, nTrials):
        Config.Start('Train')
        for i in range(nTrials):
            self.TryStep()
            if self.nBetter >= self.nMaxBetter:
                break
            if path.exists(self.sfAbort):
                print('Aborting...')
                break
            
        self.tableGenerator.OnEndTraining()
        Config.End('Train')

     

        
def main():
    bShort = True
    bShort = False
    print('*** Train Poly Table')
    Config.OnInitRun()
    Config.Clean()
    #vol = CVolume('nominalVol', sVolumeFileNameNominal)
    trainer = CPolyTrainer()
    #trainer.RunOriginalReconAndScore()
    trainer.RunFlatTable()
    
    if bShort:
        trainer.Train(10)
    else:
        trainer.Train(20000)

if __name__ == '__main__':
    main()
