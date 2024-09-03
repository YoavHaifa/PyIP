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

import Config
from Volume import CVolume
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Sample import CSample
from PolyScorer import CPolyScorer
from RunRecon import RunAiRecon, RunOriginalRecon
from PolyTable import CPolyTables
from Utils import GetAbortFileName

nDetectors = 688
nRows = 192
nLayers = 3
deltaSave = 100
deltaSample = 50

BIG_SCORE = 1000000.0

class CPolyTrainer:
    """
    """
    def __init__(self):
        """
        """
        self.originalScore = BIG_SCORE
        self.firstScore = BIG_SCORE
        self.bestScore = BIG_SCORE
        self.nImprovements = 0
        self.radIm = CRadiusImage()
        
        RunOriginalRecon()
        
        self.originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
        self.maskVol = CMaskVolume(self.originalVol)
        self.scorer = CPolyScorer()
        self.sample = CSample(self.maskVol, self.radIm)
        self.iSample = 0

        self.iTry = 0
        self.nBetter = 0
        self.nBetter0 = 0
        self.nBetter1 = 0
        self.nBetterPerSample = 0;
        self.nNotBetterPerSample = 0
        self.nNotBetterConsecutive = 0
        self.nMaxBetter = 10000
        
        self.sfAbort = GetAbortFileName()
        self.originalScore = self.scorer.Score(Config.sfVolumeNominal, self.sample)

    def OpenCsv(self, sfName):
        f, sfName = Config.OpenLogGetName(sfName)
        #f.write('xrt, fRow, lRow, fCol, lCol, delta, score, diff\n')
        #f.write(f'01, 0, {nRows}, 0, {nDetectors}, 0, {self.bestScore}, 0\n')
        f.write('xrt, sample, delta, std, average, score, diff\n')
        f.write(f'0_1, 0, 0, {self.scorer.std}, {self.scorer.average}, {self.bestScore}, 0\n')
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
        self.sample = CSample(self.maskVol, self.radIm)
        prevBest = self.bestScore
        self.firstScore = self.scorer.Score(Config.sfVolumeAi, self.sample)
        self.bestScore = self.firstScore
        self.iSample += 1
        print(f'<CreateNewSampe> {self.iSample} best {prevBest} --> {self.bestScore}')

    def TryStep(self):
        self.iTry += 1 
        print('<TryStep> ', self.iTry)
        iTable = random.randint(0, 1)
        delta = self.tableGenerator.TryTableStep(iTable)
        RunAiRecon()
        score = self.scorer.Score(Config.sfVolumeAi, self.sample).item()
        diff = self.bestScore - score
        fAll = open(self.sfAllCsv,'a')
        #fAll.write(f'{iTable}, {iFirstRow}, {iRowAfter}, {iFirstCol}, {iColAfter}, {delta}, {score}, {diff}')
        fAll.write(f'{iTable}, {self.iSample}, {delta}, {self.scorer.std}, {self.scorer.average}, {score}, {diff}')

        if score < self.bestScore:
            if iTable == 0:
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
            fImp.write(f'{iTable}, {self.iSample}, {delta}, {self.scorer.std}, {self.scorer.average}, {score}, {diff}\n')
            fImp.close()
            
            if self.nBetter % deltaSample == 0:
                self.CreateNewSampe()
            if self.nBetter % deltaSave == 0:
                Config.SaveAiVolume(self.nBetter)
                #sfSave = f'd:\Dump\BP_PolyAI_Output_width512_height512_save{self.nBetter}.float.dat'
                #TryRename(sVolumeFileNameAi, sfSave)
            self.nNotBetterConsecutive = 0
            return True
        
        fAll.write('\n')
        fAll.close()
        self.nNotBetterConsecutive += 1
        self.nNotBetterPerSample += 1
        self.tableGenerator.RestoreTable(iTable)
        print(f'--- <<< New table NOT better {self.nNotBetterConsecutive}')
        #print(f'--- <<< New table NOT better {score=} >= {self.bestScore} ({diff=})')
        return False
    
    def Train(self, nTrials):
        for i in range(nTrials):
            self.TryStep()
            if self.nBetter >= self.nMaxBetter:
                break
            if path.exists(self.sfAbort):
                print('Aborting...')
                break
            
        self.tableGenerator.OnEndTraining()

     

        
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
        trainer.Train(20)
    else:
        trainer.Train(20000)

if __name__ == '__main__':
    main()
