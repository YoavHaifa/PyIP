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


sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_70_c170_1_346'
sfIR = 'Tube0_IR_grid_r67_d170.csv'

firstStepAmplitude = 0.001
LRFraction = 0.1

verbosity = 1

class CTabValue:
    """
    Single value in the poly table
    """
    def __init__(self, iTube, csvIRLine):
        """
        """
        print(f'{csvIRLine=}')
        self.iTube = iTube
        self.iRow = int(csvIRLine['row'].item())
        self.iDet = int(csvIRLine[' det'].item())
        self.iIm = int(csvIRLine[' im'].item())
        self.iRad = int(csvIRLine[' rad'].item())
        
        self.prevDev = None
        self.dev = None
        self.deltaDev = None
        self.deltaTable = None
        self.grad = None
        self.prevTabValue = 1.0

    def SetInitialDev(self, devMap):
        self.initialDev = devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.prevDev = self.initialDev.clone()
        self.prevTabValue = 1.0

    def SetSecondDev(self, devMap):
        self.secondDev = devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.dev = self.secondDev.clone()
        self.deltaDev = self.dev - self.prevDev
        if self.deltaTable != 0:
            self.grad = self.deltaDev / self.deltaTable
        else:
            self.grad = torch.ones([2,2])

    def SetNextDev(self, devMap):
        self.prevDev = self.dev
        self.dev = devMap[self.iIm:self.iIm+2, self.iRad:self.iRad+2].clone()
        self.deltaDev = self.dev - self.prevDev
        if self.deltaTable != 0:
            self.grad = self.deltaDev / self.deltaTable
        else:
            self.grad = torch.ones([2,2])
        
    def AdjustTable(self, tabGen, delta):
        self.tabValue = self.prevTabValue + delta
        tabGen.SetValue(self.iTube, self.iRow, self.iDet, self.tabValue)
        self.deltaTable = self.tabValue - self.prevTabValue
        
    def StepFrom(self, tabValue, dev, tabGen):
        multiStep = dev / self.grad
        self.deltaTable = multiStep.sum().item() * LRFraction
        if verbosity > 2:
            print(f'<StepFrom> {dev=}')
            print(f'<StepFrom> {self.grad=}')
            print(f'<StepFrom> {multiStep=}')
            print(f'<StepFrom> {self.deltaTable=}')
        self.dev = dev
        self.prevTabValue = tabValue
        self.tabValue = tabValue + self.deltaTable
        tabGen.SetValue(self.iTube, self.iRow, self.iDet, self.tabValue)
        if verbosity > 2:
            print(f'<StepFrom> {self.tabValue=}')
        
        
    def Step(self, tabGen):
        prevDevAvg = self.prevDev.mean().item()
        devAvg = self.dev.mean().item()
        if abs(prevDevAvg) < abs(devAvg):
            print('<Step> from PrevDev')
            self.StepFrom(self.prevTabValue, self.prevDev, tabGen)
        else:
            if verbosity > 2:
                print('<Step> from Last Dev')
            self.StepFrom(self.tabValue, self.dev, tabGen)
        
    def Print(self):
        print(f'CTabValue: [{self.iRow},{self.iDet}] --> [{self.iIm}, {self.iRad}]')
        if self.prevDev is not None:
            print(f'{self.prevDev=}')
        if self.dev is not None:
            print(f'{self.dev=}')
        if self.deltaDev is not None:
            print(f'{self.deltaDev=}')
            
        if self.deltaTable is not None:
            print(f'Table {self.prevTabValue} --> {self.tabValue}, delta {self.deltaTable}')
        if self.grad is not None:
            print(f'{self.grad=}')
        
    def ShortPrint(self, i):
        dev = self.dev
        print(f'<TV-{i}> {self.tabValue} -> {dev[0,0]:.3f}, {dev[0,1]:.3f}, {dev[1,0]:.3f}, {dev[1,1]:.3f}, ')

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
    tabVal.AdjustTable(tableGenerator, firstStepAmplitude)
            
    RunAiRecon('SecondTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    secondDevMap = scorer.devRaster.dev.clone()
    return secondDevMap

def RunNextTable(scorer, sample):
    print('*** <RunSecondTable>')
           
    RunAiRecon('NextTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    devMap = scorer.devRaster.dev.clone()
    return devMap

def Train(tabVal, tableGenerator, scorer, sample):
    for i in range(1,11):
        tabVal.Step(tableGenerator)
        devMap = RunNextTable(scorer, sample)
        tabVal.SetNextDev(devMap)
        tabVal.ShortPrint(i)

def main():
    print('*** Read Impulse Response Data')
    VeifyReconRunning()
    sfName = path.join(sIRDir, sfIR)
    df = ReadData(sfName)
    tabVal = CTabValue(0, df.iloc[3])
    tabVal.Print()
    
    scorer = CPolyScorer()
    radIm = CRadiusImage()
    originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
    maskVol = CMaskVolume(originalVol)
    sample = CSample(maskVol, radIm)
    tableGenerator = CPolyTables() # Prepares initial table

    # Run Initial
    initialDevMap = RunInitialTable(scorer, sample, tableGenerator)
    Config.WriteDevToFile(initialDevMap, 'flatTab_initial')

    tabVal.SetInitialDev(initialDevMap)
    tabVal.Print()

    #Run Second
    secondDevMap = RunSecondTable(tabVal, scorer, sample, tableGenerator)
    Config.WriteDevToFile(secondDevMap, 'Tab_second')
    tabVal.SetSecondDev(secondDevMap)
    tabVal.Print()
    
    # Try Step
    tabVal.Step(tableGenerator)
    devMap = RunNextTable(scorer, sample)
    tabVal.SetNextDev(devMap)
    tabVal.Print()
    tabVal.ShortPrint(0)
    
    Train(tabVal, tableGenerator, scorer, sample)
    
    """
    Config.SaveAiVolume(self.count)
    self.prevDevAtPoint = self.initialDevAtPoint
    self.devAtPoint = self.secondDevMap[iImage, iRad]
    if abs(self.devAtPoint) > abs(self.prevDevAtPoint):
        self.prevDevAtPoint, self.devAtPoint = self.devAtPoint, self.prevDevAtPoint
        self.prevTabValue, self.tabValue = self.tabValue, self.prevTabValue
        print('<RunSecondTable> SWAP first and secon')
    devDiff = self.devAtPoint - self.prevDevAtPoint
    print(f'<RunSecondTable> tabValue {self.tabValue}, dev {self.prevDevAtPoint} --> {self.devAtPoint}, diff {devDiff}')
    """

"""
    initialDevAtPoint = initialDevMap[iImage, iRad]
    
    Config.SaveAiVolume(self.count)
    s = f'<RunInitialTable> dev at point {self.initialDevAtPoint}, first score {self.firstScore}'
    print(s)
    """
    
    
if __name__ == '__main__':
    main()
