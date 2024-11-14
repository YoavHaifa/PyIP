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
from RunRecon import RunAiRecon, RunOriginalRecon, VeifyReconRunning
from PolyTable import CPolyTables
from Utils import GetAbortFileName
from Log import Log, Start, End
from InverseIR import GInverseByTube

nDetectors = 688
nRows = 192
nLayers = 3
deltaSave = 200
deltaSample = 400

BIG_SCORE = 1000000.0

nStepsToReport = 10

verbosity = 1

def consolePrint(sLine, sChar):
    if verbosity > 1:
        print(sLine)
    else:
        print(sChar,end='')
    Log(sLine)

class CPolyTrainer:
    """
    """
    def __init__(self):
        """
        """
        #Config.gLog = CLog('Trainer')
        self.originalScore = BIG_SCORE
        self.firstOldScore = BIG_SCORE
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

        self.nTry = 0
        self.nBetter = 0
        self.nBetterNotReported = 0
        
        self.nBetterPerSample = 0;
        self.nNotBetterPerSample = 0
        self.nFailedSamples = 0
        
        self.nNotBetterConsecutive = 0
        self.nMaxBetter = 10000
        
        self.sfAbort = GetAbortFileName()
        self.originalScore = self.scorer.OldScore(Config.sfVolumeNominal, self.sample)
        self.sNext = 'targeted'
        self.bUseRandom = False
        #self.IIR = CInverseIRTablesPerTubes

    def OpenCsv(self, sfName):
        f, sfName = Config.OpenLogGetName(sfName)
        
        #Write Title for all columns
        f.write('xrt, sample, delta, step')
        self.scorer.WriteTitle(f)
        f.write(', diff\n')
       
        f.write(f'0_1, 0, {Config.firstDelta}, R')
        self.scorer.WriteScore(f)
        f.write(', 0\n')
        f.close()
        return sfName
                
    def RunInitialTable(self):
        self.tableGenerator = CPolyTables() # Prepares initial table
        RunAiRecon()
        self.firstOldScore = self.scorer.OldScore(Config.sfVolumeAi, self.sample)
        self.firstScore = self.scorer.NewScore(Config.sfVolumeAi, self.sample)
        s = f'<RunInitialTable> first score {self.firstScore}'
        print(s)
        Log(s)
        self.bestScore = self.firstScore
        self.tableGenerator.InitPatches(self.bestScore)
        
        self.sfAllCsv = self.OpenCsv('Training_All.csv')
        self.sfImproveCsv = self.OpenCsv('Training_Improve.csv')
        
        Config.SaveAiVolume(0)
        
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
        self.firstScore = self.scorer.NewScore(Config.sfVolumeAi, self.sample)
        self.bestScore = self.firstScore
            
        self.nBetterPerSample = 0;
        self.nNotBetterPerSample = 0

        print(f'<CreateNewSampe> {self.iSample} best {prevBest} --> {self.bestScore}')

    def SelectNextStep(self):
        #print(f'<SelectNextStep> {self.sNext=}')
        if self.sNext == 'targeted':
            self.iImage = self.scorer.iImageMaxDev
            self.iRad = self.scorer.iRadMaxDev
            self.deviation = self.scorer.maxDev
            self.width = self.scorer.maxDevWidth
            #print(f'<SelectNextStep> {iImage=}, {iRad=}')
            #InverseIR = self.IIR
            #print(f'{InverseIR=}')
            self.iTube = random.randint(0,1)
            iRow, iDet = GInverseByTube(self.iTube, self.iImage, self.iRad+1)
            #iTube, iRow, iDet = InverseIR.InverseByRandomTube(iImage=iImage,iRad=iRad)
            self.delta = self.tableGenerator.TryTargetedTableStep(self.iTube, iRow, iDet, self.deviation, self.width)
            consolePrint(f'<TryStep> {self.nTry} TARGETED {self.tableGenerator.sLast} {self.delta=}', 't')
            
        elif self.sNext == 'random':
            self.delta = self.tableGenerator.TryRandomTableStep()
            consolePrint(f'<TryStep> {self.nTry} RANDOM {self.tableGenerator.sLast} {self.delta=}', 'r')
                
        elif self.sNext == 'OnFailure':
            self.iTube = 1 - self.iTube
            iRow, iDet = GInverseByTube(self.iTube, self.iImage, self.iRad+1)
            #iTube, iRow, iDet = InverseIR.InverseByRandomTube(iImage=iImage,iRad=iRad)
            self.delta = self.tableGenerator.TryTargetedTableStep(self.iTube, iRow, iDet, self.deviation, self.width)
            consolePrint(f'<TryStep> {self.nTry} TARG-Other {self.tableGenerator.sLast} {self.delta=}', 'o')
            # Old "On Failure" changed sign
            #self.delta = self.tableGenerator.TryOnFailure()
            #consolePrint(f'<TryStep> {self.nTry} ON_FAIL {self.tableGenerator.sLast} {self.delta=}', 'f')
        #elif self.sNext == 'OnSuccess':
        #    self.delta = self.tableGenerator.TryOnSuccess()
        #    print(f'<TryStep> {self.nTry} ON_GOOD {self.tableGenerator.sLast} {self.delta=}')
        else:
            print(f'<SelectNextStep> Illegal step {self.sNext}')
            sys.exit()

    def OnBetter(self, score):
            Start('OnBetter')
                
            self.nBetter += 1
            self.nBetterNotReported += 1
            consolePrint(f'+++ === >>> New table better {self.nBetter} {score=} < {self.bestScore}', '+')
            self.bestScore = score
            
            if self.nBetter % deltaSample == 0:
                self.CreateNewSampe()
            if self.nBetter % deltaSave == 0:
                Config.SaveAiVolume(self.nBetter)
                
            self.nNotBetterConsecutive = 0
                
            if not self.bUseRandom:
                #self.sNext = 'random'
                self.sNext = 'targeted'
            self.nBetterPerSample += 1
            End('OnBetter')
            
    def LogStep(self, sf, gain, bAll = False):
        with open(sf,'a') as f:
            f.write(f'{self.tableGenerator.iCurTab}, {self.iSample}, {self.delta}, {self.tableGenerator.sLast}')
            sScore = self.scorer.WriteScore(f)
            f.write(f', {gain}')
            if bAll and gain > 0:
                f.write(', *')
            f.write('\n')
        return sScore
        
    def OnNotbetter(self):
        self.nNotBetterConsecutive += 1
        self.nNotBetterPerSample += 1
        consolePrint(f'--- <<< New table NOT better {self.nNotBetterConsecutive}', '-')
        if self.bUseRandom:
            if self.sNext == 'targeted':
                self.sNext = 'random'
            if self.sNext == 'random':
                self.sNext = 'OnFailure'
            else:
                self.sNext = 'targeted'
        else:
            # All targeted - but check other tube on failure
            if self.sNext == 'targeted':
                self.sNext = 'OnFailure'
            else:
                self.sNext = 'targeted'
            
        
    def TryStep(self):
        Start('TryStep')
        self.nTry += 1 
        self.SelectNextStep()
        RunAiRecon()
        score = self.scorer.NewScore(Config.sfVolumeAi, self.sample)
        gain = self.bestScore - score
        self.LogStep(self.sfAllCsv, gain, bAll = True)
        
        self.tableGenerator.OnNewScore(self.bestScore, score)

        if score < self.bestScore:
            self.OnBetter(score)
            sScore = self.LogStep(self.sfImproveCsv, gain)
            Log(sScore)
        else:
            self.OnNotbetter()
            
        End('TryStep')
        if self.nTry % nStepsToReport == 0:
            print(f' Better {self.nBetter}/{self.nTry}, new {self.nBetterNotReported}/{nStepsToReport}, best {self.bestScore:.6f}')
            self.nBetterNotReported = 0
    
    def Train(self, nTrials):
        Start('Train')
        print(f'<Train> n={nTrials}')
        for i in range(nTrials):
            self.TryStep()
            if self.nBetter >= self.nMaxBetter:
                break
            if path.exists(self.sfAbort):
                print('Aborting...')
                break
            
        self.tableGenerator.OnEndTraining()
        End('Train')

     

        
def main():
    VeifyReconRunning()
    bShort = False
    #bShort = True
    if Config.sExp == 'try':
        bShort = True
    print('*** Train Poly Table')
    Config.OnInitRun()
    Config.Clean()
    #vol = CVolume('nominalVol', sVolumeFileNameNominal)
    trainer = CPolyTrainer()
    #trainer.RunOriginalReconAndScore()
    trainer.RunInitialTable()
    
    if bShort:
        trainer.Train(20)
    else:
        trainer.Train(20000)

if __name__ == '__main__':
    main()
