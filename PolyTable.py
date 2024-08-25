# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:37:15 2024
Control Poly Table
@author: yoav.bar
"""

import os
from os import path
import torch
import random
import sys

from RunRecon import RunRecon
from Flatness import ScoreFlatness
from Utils import VerifyDir

nDetectors = 688
nRows = 192
nLayers = 3
nDetsPerReading = nDetectors * nRows

sAiFlag = 'd:\Config\Poly\GetAiTable.txt'
sAiFlagRemoved = 'd:\Config\Poly\GetAiTable_x.txt'

sfNominalTab0 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT0.bin'
sfNominalTab1 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT1.bin'
sfAiTab0 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT0_ai.bin'
sfAiTab1 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT1_ai.bin'

    
def SaveTable(table, sfName):
    npTable = table.numpy()
    with open (sfName, 'wb') as file:
        file.write(npTable.tobytes())
    print('Poly Table Saved:', sfName)
    
def PrepareEmptyTable():
    table = torch.zeros([nLayers,nRows,nDetectors])
    table[0] = torch.ones([nRows,nDetectors])
    SaveTable(table, sfAiTab0)
    SaveTable(table, sfAiTab1)
    return table

def TryRename(sFrom, sTo):
    print('Try Rename')
    if path.exists(sFrom):
        os.rename(sFrom, sTo)
        print(f'Renamed {sFrom} to {sTo}')

def SetPolyNominal():
    TryRename(sAiFlag, sAiFlagRemoved)

def SetPolyByAi():
    TryRename(sAiFlagRemoved, sAiFlag)
    

def RunNominalRecon():
    SetPolyNominal()
    RunRecon()
    
class CPolyTables:
    """
    Learn poly table by trying to improve flatness
    """
    def __init__(self):
        """
        Starts with 2 empty tables
        Run reconstruction and score flatness
        Try variations on the tables and keeps beneficial variants

        Returns 2 improved tables

        """
        SetPolyByAi()
        self.table0 = PrepareEmptyTable()
        self.table1 = self.table0.clone().detach()
        self.bestScore = 1000.0
        self.iTableTry = 0
        self.nTry = 0
        self.nBetter = 0
        self.nBetter0 = 0
        self.nBetter1 = 0
        self.RunFirst()
        
        self.sfAllCsv = self.OpenCsv('d:\PyLog\Training_All.csv')
        self.sfImproveCsv = self.OpenCsv('d:\PyLog\Training_Improve.csv')

    def OpenCsv(self, sfName):
        f = open(sfName,'w')
        f.write('xrt, fRow, lRow, fCol, lCol, delta, score, diff\n')
        f.write(f'01, 0, {nRows}, 0, {nDetectors}, 0, {self.bestScore}, 0\n')
        f.close()
        return sfName

    def OnEndTraining(self):
        #self.fAllCsv.close()
        #self.fImproveCsv.close()
        self.SaveTables()
    
    def RunFirst(self):
        RunRecon()
        self.bestScore, centralImage = ScoreFlatness()
        #print(f'{centralImage.fName=}')
        newName = centralImage.fName.replace('Central_Image', 'Central_Image_step0')
        centralImage.fName = newName
        #print(f'{centralImage.fName=}')
        centralImage.WriteToFile()
        #sys.exit()
        
    def SaveTables(self):
        SaveTable(self.table0, sfAiTab0)
        SaveTable(self.table1, sfAiTab1)
        
        
    def TryTableStep(self, table, sfName, iTable):
        self.nTry += 1
        print(f'<TryTableStep> {self.nTry}')
        tmpTable = table.clone().detach()
        iFirstRow = random.randint(0, nRows-1)
        iRowAfter = random.randint(iFirstRow+1, nRows)
        iFirstCol = random.randint(0, nDetectors-1)
        iColAfter = random.randint(iFirstCol+1, nDetectors)
        delta = random.random() - 0.5
        delta = delta / 3000
        tmpTable[0,iFirstRow:iRowAfter,iFirstCol:iColAfter] += delta
        SaveTable(tmpTable, sfName)
        #print(f'Try table {sfName}')
        print(f'CHANGED: rows {iFirstRow}:{iRowAfter}, cols {iFirstCol}:{iColAfter}, {delta=}')
        RunRecon()
        score, centralImage = ScoreFlatness()
        diff = self.bestScore - score
        fAll = open(self.sfAllCsv,'a')
        fAll.write(f'{iTable}, {iFirstRow}, {iRowAfter}, {iFirstCol}, {iColAfter}, {delta}, {score}, {diff}')

        if score < self.bestScore:
            if iTable == 0:
                self.table0 = tmpTable
                self.nBetter0 += 1
            else:
                self.table1 = tmpTable
                self.nBetter1 += 1
                
            self.nBetter += 1
            print(f'New table better {self.nBetter} {score=} < {self.bestScore}')
            self.bestScore = score
            centralImage.fName = centralImage.fName.replace('Central_Image', f'Central_Image_step{self.nBetter}')
            centralImage.WriteToFile()
            fAll.write(', *\n')
            fImp = open(self.sfImproveCsv, 'a')
            fImp.write(f'{iTable}, {iFirstRow}, {iRowAfter}, {iFirstCol}, {iColAfter}, {delta}, {score}, {diff}\n')
            fImp.close()
            return True
        
        fAll.write('\n')
        fAll.close()
        SaveTable(table, sfName)
        print(f'New table NOT better {score=} >= {self.bestScore} ({diff=})')
        return False
            
    def SaveBetter(self, iXrt, iBetter, table):
        sfName = f'd:\PolyCalib\Better_XRT{iXrt}\Poly_XRT{iXrt}_Better{iBetter}_width{nDetectors}_height{nRows}.float.bin'
        SaveTable(table,sfName)
        
    def TryStep(self):
        self.iTry = random.randint(0, 1)
        if (self.iTry == 0):
            bBetter = self.TryTableStep(self.table0, sfAiTab0, 0)
            if bBetter:
                self.SaveBetter(0, self.nBetter0, self.table0)
        else:
            bBetter = self.TryTableStep(self.table1, sfAiTab1, 1)
            if bBetter:
                self.SaveBetter(1, self.nBetter1, self.table1)

    
def TrainPolyCal(nTrials,nMaxBetter=1000000):
    VerifyDir('d:\Dump')
    VerifyDir('d:\PyLog')
    VerifyDir('d:\PolyCalib')
    VerifyDir('d:\PolyCalib\Better_XRT0')
    VerifyDir('d:\PolyCalib\Better_XRT1')
    learner = CPolyTables()
    for i in range(nTrials):
        learner.TryStep()
        if learner.nBetter >= nMaxBetter:
            break
        
    learner.OnEndTraining()


def main():
    bNominal = False
    nTrain = 200
    
    if bNominal:
        print('*** Try Nominal')
        SetPolyNominal()
        RunRecon()
        ScoreFlatness(bNominal=True)
    else:
        TrainPolyCal(nTrain, nMaxBetter = 1)

if __name__ == '__main__':
    main()
