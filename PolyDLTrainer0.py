# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:31:03 2024
Train Poly Table for a single or few points
@author: yoavb
"""

import sys
from os import path
import torch
import torch.optim as optim

import Config
from RunRecon import RunAiRecon, VeifyReconRunning
from PolyTable import CPolyTables
from Log import Log, Start, End
from PolyScorer import CPolyScorer
from Sample import CSample
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Volume import CVolume
from PolyModel import CPolyModel
from ExRecon import CExRecon
from Utils import GetAbortFileName
from CsvLog import gCsvLog

iTube = 0
iRow = 70
iDet = 300


nDetectors = 688
nRows = 192
nLayers = 3

nSmallInput = 10

sfOriginalVolume = ''

class CPolyDLTrainer0:
    """
    """
    def __init__(self, n):
        """
        """
        self.nTrials = n
        self.scorer = CPolyScorer()
        self.radIm = CRadiusImage()
        self.originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
        self.maskVol = CMaskVolume(self.originalVol)
        self.sample = CSample(self.maskVol, self.radIm)
        self.sfAbort = GetAbortFileName()
        
    def RunInitialTable(self):
        print('*** <RunInitialTable>')
        self.tableGenerator = CPolyTables() # Prepares initial table
        RunAiRecon('InitialTable')
        self.firstOldScore = self.scorer.OldScore(Config.sfVolumeAi, self.sample, bSikpFirst=True)
        self.scorer.ComputeNewScoreOfVolume1(Config.sfVolumeAi, self.sample)
        self.firstScore = self.scorer.ComputeNewScoreOfVolume2()
        s = f'<RunInitialTable> first score {self.firstScore}'
        print(s)
        Log(s)
        self.bestScore = self.firstScore
        self.tableGenerator.InitPatches(self.bestScore)
        
        #self.sfAllCsv = self.OpenCsv('Training_All.csv')
        #self.sfImproveCsv = self.OpenCsv('Training_Improve.csv')
        
        self.initialDevMap = self.scorer.devRaster.dev.clone()
        Config.WriteMatrixToFile(self.initialDevMap, 'DevMap_flatTab_initial')
        
        Config.SaveAiVolume(0)
        

    def TrySingleTableSpot(self, iTube, iRow, iDet):
        print(f'<TrySingleTableSpot> {iTube=},  {iRow=}, {iDet=}')
        self.tableGenerator.SetValue(iTube, iRow, iDet, 2)
        RunAiRecon('SecondTable')
        self.scorer.ComputeNewScoreOfVolume1(Config.sfVolumeAi, self.sample)
        self.secondScore = self.scorer.ComputeNewScoreOfVolume2()
        print(f'<RunInitialTable> second score {self.secondScore}')
        self.secondDevMap = self.scorer.devRaster.dev.clone()
        Config.WriteMatrixToFile(self.secondDevMap, f'DevMap_second_t{iTube}_r{iRow}_d{iDet}')
        
    def NextLR(self):
        self.curLR = self.curLR / 2 
        self.optimizer.param_groups[0]['lr'] = self.curLR
        print(f'LR reduced to {self.curLR}')
        
    def TrainDL0(self, nTrials):
        Start('TrainDL')
        nIn = 1 #nSmallInput # nImages * maxRadius
        tInput = torch.ones(nIn)
        #nOut = nRows * nDetectors * 2
        nOut = 1
        print(f'{nIn=}, {nOut=}')
        model = CPolyModel(nIn, nOut)
        
        ApplyRecon = CExRecon.apply
        self.curLR = 0.1
        self.optimizer = optim.SGD(model.model.parameters(), lr=self.curLR)
        bestLoss = 1000000
        prevLoss = bestLoss
        
        for i in range(nTrials):
            gCsvLog.StartNewLine()
            gCsvLog.AddItem(self.curLR)
            tabs1 = model.model(tInput)
            tabs1 = tabs1 / 100 - 0.005 + 1.0
            gCsvLog.AddItem(tabs1[0])
            #tabs1 = tabs.view(-1,nRows,nDetectors)
            if Config.debug & 8:
                print(f'{tabs1=}')
            #print(f'{tabs.size()=}')
            with torch.no_grad():
                #self.tableGenerator.Set1(tabs1)
                self.tableGenerator.SetValue(iTube, iRow, iDet, tabs1[0])
                
            loss = ApplyRecon(tabs1, self.scorer, self.sample)
            gCsvLog.AddLastItem(loss)
            if i == 0:
                bestLoss = loss
            elif loss < bestLoss:
                bestLoss = loss
            elif loss > prevLoss and self.curLR > 0.001:
                self.NextLR()                
            elif loss > 2*bestLoss:
                print('Quitting on bad direction...')
                break
            prevLoss - loss
                
            self.optimizer.zero_grad()
            #print(f'CALL loss.backward(), {loss=}')
            loss.backward()
            #print('optimizer.step()')
            self.optimizer.step()
            
            if loss < 1.0 and self.curLR == 0.1:
                self.NextLR()                
            
            if path.exists(self.sfAbort):
                print('Aborting...')
                break
            
        End('TrainDL')

def main():
    VeifyReconRunning()
    bDummy = False
    bShort = False
    nShort = 100
    #bShort = True
    if Config.sExp == 'try':
        bShort = True
    print('*** Train Poly Table')
    Config.OnInitRun()
    Config.Clean()
    trainer = CPolyDLTrainer0(1)
    #trainer.RunOriginalReconAndScore()
    trainer.RunInitialTable()
    
    if bDummy:
        trainer.TrySingleTableSpot(iTube, iRow, iDet)
        sys.exit()
    
    if bShort:
        trainer.TrainDL0(nShort)
    else:
        trainer.TrainDL0(20000)

if __name__ == '__main__':
    main()

