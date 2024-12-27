# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:16:28 2024
Information for a single point in the table
@author: yoavb
"""


import sys
from os import path
#import csv
import pandas as pd
import torch

import Config
from PolyTable import CPolyTables
from RunRecon import RunAiRecon, VeifyReconRunning
from PolyScorer import CPolyScorer
from Sample import CSample
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Volume import CVolume
from CsvLog import CCsvLog
from Utils import GetAbortFileName


sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_70_c170_1_346'
sfIR = 'Tube0_IR_grid_r67_d170.csv'
sfAbort = GetAbortFileName()

firstStepAmplitude = 0.001
LRFraction = 0.04


sTitle = 'i, delta00, delta01, delta10, delta11, g00, g01, g10, g11'
sTitle = sTitle + ', iMaxGrad, fullDelta, tabDelta'
sTitle = sTitle + ', tab, dev00, dev01, dev10, dev11, score'

verbosity = 1

class CTabValue:
    """
    Single value in the poly table
    """
    def __init__(self, iTube, csvIRLine):
        """
        """
        if verbosity > 1:
            print(f'{csvIRLine=}')
        self.iTube = iTube
        self.iRow = int(csvIRLine['row'].item())
        self.iDet = int(csvIRLine[' det'].item())
        self.iIm = int(csvIRLine[' im'].item())
        self.iRad = int(csvIRLine[' rad'].item())
        
        self.signature = f't{iTube}_r{self.iRow}_d{self.iDet}'
        sfName = f'Train_tab_value_{self.signature}.csv'
        self.csv = CCsvLog(sfName, sTitle)
        
        self.prevDev = None
        self.dev = None
        self.deltaDev = None
        self.deltaTable = None
        self.grad = None
        self.score = 1000
        self.prevTabValue = 1.0
        self.nBad = 0
        self.count = -1
        self.tabValue = 1
        
    def SetDevFromEnv(self, env):
        self.count += 1
        self.prevScore = self.score
        self.dev = env.devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.score = self.dev.abs().mean().item()
        
        self.LogGrad()
        for d in self.dev.view(4):
            self.csv.AddItem(d.item())
        self.csv.AddLastItem(self.score)
       
    def SelectLastOrPrev(self):
        if self.score > self.prevScore:
            self.prevDev, self.dev = self.dev, self.prevDev
            self.prevTabValue, self.tabValue = self.tabValue, self.prevTabValue
            self.prevScore, self.score = self.score, self.prevScore
            self.ComputeGradFinal()
       
    def SetDev(self, devMap):
        self.count += 1
        #Save prev
        self.prevDev = self.dev
        self.prevScore = self.score
        # Set new
        self.dev = devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        for d in self.dev.view(4):
            self.csv.AddItem(d.item())
        self.score = self.dev.abs().mean().item()
        self.csv.AddLastItem(self.score)
            
        if self.score > self.prevScore:
            self.nBad += 1 
            self.Print()
            print(f'<ComputeGrad> with WORSE SCORE {self.nBad} ({self.score:.6f} > {self.prevScore:.3f})')
            if self.nBad > 2:
                self.ShortPrint()
                sys.exit()

    def SetInitialDev(self, devMap):
        self.LogNoGrad(tabPos = 1.0)

        self.prevTabValue = 1.0
        self.dev = devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.SetDev(devMap)
        
    def ComputeGradFinal(self):
        self.deltaTable = self.tabValue - self.prevTabValue
        self.deltaDev = self.dev - self.prevDev
        if self.deltaTable != 0:
            self.grad = self.deltaDev / self.deltaTable
        else:
            print('<ComputeGrad2> WARNING: deltaTable was 0!')
            self.grad = torch.ones([2,2])
        
    def ComputeGrad2(self, env):
        self.SetDevFromEnv(env)
        self.prevDev = env.prevDev[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.ComputeGradFinal()
        
    def ComputeGrad(self):
        self.deltaTable = self.tabValue - self.prevTabValue
        self.deltaDev = self.dev - self.prevDev
        if self.deltaTable != 0:
            self.grad = self.deltaDev / self.deltaTable
        else:
            print('<ComputeGrad> WARNING: deltaTable was 0!')
            self.grad = torch.ones([2,2])
        
    def LogGrad(self, tabPos = None):
        self.csv.StartNewLine()
        if self.grad is None:
            self.LogNoGrad()
        else:
            for delta in self.deltaDev.view(4):
                self.csv.AddItem(delta.item())
            for g in self.grad.view(4):
                self.csv.AddItem(g.item())
        
            self.csv.AddItem(self.iMaxGrad)
            self.csv.AddItem(self.fullDelta)
            self.csv.AddItem(self.delta)
            
        self.csv.AddItem(self.tabValue*1000)
        
    def LogNoGrad(self):
        for i in range(11): # delta and grad
            self.csv.AddItem(0)

    def AdjustTable(self, tabGen, delta):
        self.prevTabValue = self.tabValue
        self.tabValue = self.prevTabValue + delta
        tabGen.SetValue(self.iTube, self.iRow, self.iDet, self.tabValue)

    def AdjustTableLocally(self, tabGen, delta):
        self.prevTabValue = self.tabValue
        self.tabValue = self.prevTabValue + delta
        tabGen.SetValueLocally(self.iTube, self.iRow, self.iDet, self.tabValue)

    def SetNextStep(self, tabGen):
        flatGrad = self.grad.view(4)
        flatDev = self.dev.view(4)
        absGrad = flatGrad.abs()
        self.iMaxGrad = absGrad.argmax()
        maxGrad = flatGrad[self.iMaxGrad]
        maxGradDev = flatDev[self.iMaxGrad]
        self.fullDelta = - maxGradDev / maxGrad;
        self.delta = self.fullDelta * LRFraction
        self.AdjustTableLocally(tabGen, self.delta)

    def Step(self, tabGen):
        if abs(self.prevScore) < abs(self.score):
            self.prevTabValue, self.tabValue = self.tabValue, self.prevTabValue
            self.prevDev, self.dev = self.dev, self.prevDev
            print('<Step> WARNING - *** from PrevDev ***')
            
        self.ComputeGrad()
            
        #multiStep = self.dev / self.grad
        #for m in multiStep.view(4):
        #    self.csv.AddItem(m.item())
        #mSum = multiStep.sum().item()
        
        flatGrad = self.grad.view(4)
        flatDev = self.dev.view(4)
        absGrad = flatGrad.abs()
        iMaxGrad = absGrad.argmax()
        maxGrad = flatGrad[iMaxGrad]
        maxGradDev = flatDev[iMaxGrad]
        fullDelta = - maxGradDev / maxGrad;
        delta = fullDelta * LRFraction
        
        self.csv.AddItem(iMaxGrad)
        self.csv.AddItem(fullDelta)
        self.csv.AddItem(delta)
        self.AdjustTable(tabGen, delta)
        
        
    def Print(self):
        print(f'CTabValue {self.signature}:')
        print(f'[{self.iRow},{self.iDet}] --> [{self.iIm}, {self.iRad}]')
        if self.prevDev is not None:
            print(f'{self.prevDev=}')
        if self.dev is not None:
            print(f'{self.dev=}, score {self.score:.6f}')
        if self.deltaDev is not None:
            print(f'{self.deltaDev=}')
            
        if self.deltaTable is not None:
            print(f'Table {self.prevTabValue*1000:.3f} --> {self.tabValue*1000:.3f}, delta {self.deltaTable*1000:.3f}')
        if self.grad is not None:
            print(f'{self.grad=}')
        
    def ShortPrint(self):
        dev = self.dev
        if dev is None:
            dev = torch.zeros([2,2])
        val1000 = self.tabValue * 1000
        print(f'<TV_{self.signature}-{self.count}> {val1000:.3f} -> {dev[0,0]:.3f}, {dev[0,1]:.3f}, {dev[1,0]:.3f}, {dev[1,1]:.3f}, score {self.score:.6f}')

def ReadData(sfName):
    print('<ReadData>', sfName)
    if not path.isfile(sfName):
        print('Missing file:', sfName)
        sys.exit()
        
    df = pd.read_csv(sfName)

    # Display the DataFrame
    print(df)  
    return df     

def RunInitialTable(scorer, sample, tableGenerator):
    print('*** <RunInitialTable>')
    RunAiRecon('InitialTable')
    scorer.OldScore(Config.sfVolumeAi, sample, bSikpFirst=True)
    #scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample, bSingle=True)
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
      
    initialDevMap = scorer.devRaster.dev.clone()
    return initialDevMap

def RunSecondTable(tabVal, scorer, sample, tableGenerator):
    print('*** <RunSecondTable>')
    tabVal.LogNoGrad()
    tabVal.AdjustTable(tableGenerator, firstStepAmplitude)
            
    RunAiRecon('SecondTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    secondDevMap = scorer.devRaster.dev.clone()
    return secondDevMap

def RunSecondTableValues(tabValues, scorer, sample, tableGenerator):
    print('*** <RunSecondTable>')
    for tv in tabValues:
        tv.LogNoGrad()
        tv.AdjustTableLocally(tableGenerator, firstStepAmplitude)
    tableGenerator.SaveTable(0) # NOTE: iTube
            
    RunAiRecon('SecondTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    secondDevMap = scorer.devRaster.dev.clone()
    return secondDevMap

def RunNextTable(scorer, sample):
    if verbosity > 2:
        print('*** <RunNextTable>')
           
    RunAiRecon('NextTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    devMap = scorer.devRaster.dev.clone()
    return devMap

def Train0(tabVal, tableGenerator, scorer, sample):
    for i in range(2,10):
        tabVal.Step(tableGenerator)
        devMap = RunNextTable(scorer, sample)
        tabVal.SetDev(devMap)
        tabVal.ShortPrint()
        
        if path.exists(sfAbort):
            print('Aborting...')
            break   

def Train(tabVals, tableGenerator, scorer, sample):
    for i in range(2,10):
        print('<Train>', i)
        for tv in tabVals:
            tv.Step(tableGenerator)
        devMap = RunNextTable(scorer, sample)
        for tv in tabVals:
            tv.SetDev(devMap)
            tv.ShortPrint()
        
        if path.exists(sfAbort):
            print('Aborting...')
            break   

def Prologue():
    print('*** Read Impulse Response Data')
    VeifyReconRunning()
    sfName = path.join(sIRDir, sfIR)
    df = ReadData(sfName)
    
    scorer = CPolyScorer()
    radIm = CRadiusImage()
    originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(originalVol)
    sample = CSample(maskVol, radIm)
    tableGenerator = CPolyTables() # Prepares initial table
        
    initialDevMap = RunInitialTable(scorer, sample, tableGenerator)
    Config.WriteDevToFile(initialDevMap, 'flatTab_initial')
    
    return df, scorer, sample, tableGenerator, initialDevMap

"""
def mainSingle():
    df, scorer, sample, tableGenerator, initialDevMap = Prologue()
    
    tabVal = CTabValue(0, df.iloc[3])
    tabVal.Print()

    # Run Initial
    tabVal.SetInitialDev(initialDevMap)
    tabVal.Print()

    #Run Second
    secondDevMap = RunSecondTable(tabVal, scorer, sample, tableGenerator)
    Config.WriteDevToFile(secondDevMap, 'Tab_second')
    tabVal.SetDev(secondDevMap)
    tabVal.Print()
    
    # Try Step
    tabVal.Step(tableGenerator)
    devMap = RunNextTable(scorer, sample)
    tabVal.SetDev(devMap)
    tabVal.Print()
    tabVal.ShortPrint()
    
    Train0(tabVal, tableGenerator, scorer, sample)
    """
    
def main():
    df, scorer, sample, tableGenerator, initialDevMap = Prologue()
    tabVals = []
    for i in range(3):
        tabVal = CTabValue(0, df.iloc[3+i])
        tabVals.append(tabVal)
        
    for tv in tabVals:
        tv.SetInitialDev(initialDevMap)
        tv.Print()
    
    secondDevMap = RunSecondTableValues(tabVals, scorer, sample, tableGenerator)

    for tv in tabVals:
        tv.SetDev(secondDevMap)
        #tv.Print()
        tv.ShortPrint()
    
    # Try Step
    for tv in tabVals:
        tv.Step(tableGenerator)
    devMap = RunNextTable(scorer, sample)
    for tv in tabVals:
        tv.SetDev(devMap)
        #tv.Print()
        tv.ShortPrint()
    
    Train(tabVals, tableGenerator, scorer, sample)
    
if __name__ == '__main__':
    main()
