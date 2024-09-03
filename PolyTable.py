# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:37:15 2024
Control Poly Table
@author: yoav.bar
"""

#import os
from os import path
import torch
import random
#import sys

import Config
#from RunRecon import RunRecon
#from Flatness import ScoreFlatness
#from Utils import VerifyDir, TryRename

nDetectors = 688
nRows = 192
nLayers = 3
nDetsPerReading = nDetectors * nRows

#sAiFlag = 'd:\Config\Poly\GetAiTable.txt'
#sAiFlagRemoved = 'd:\Config\Poly\GetAiTable_x.txt'

sfNominalTab0 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT0.bin'
sfNominalTab1 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT1.bin'
sfAiTab0 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT0_ai.bin'
sfAiTab1 = 'D:\SpotlightScans\SCANPLAN_830\Calibrations\PolyCalibration_kVp120_FOV250_Collimator140_XRT1_ai.bin'

verbosity = 1
    
def SaveTable(table, sfName):
    npTable = table.numpy()
    with open (sfName, 'wb') as file:
        file.write(npTable.tobytes())
    if verbosity > 1:
        print('Poly Table Saved:', sfName)
    
def PrepareEmptyTable():
    table = torch.zeros([nLayers,nRows,nDetectors])
    table[0] = torch.ones([nRows,nDetectors])
    SaveTable(table, sfAiTab0)
    SaveTable(table, sfAiTab1)
    return table

#def SetPolyNominal():
#    TryRename(sAiFlag, sAiFlagRemoved)

#def SetPolyByAi():
#    TryRename(sAiFlagRemoved, sAiFlag)
    

#def RunNominalRecon():
#    SetPolyNominal()
#    RunRecon()
    
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
        self.table0 = PrepareEmptyTable()
        self.table1 = self.table0.clone().detach()

    def OnEndTraining(self):
        #self.fAllCsv.close()
        #self.fImproveCsv.close()
        self.SaveTables()
    
    """
    def RunFirst(self):
        RunRecon()
        self.bestScore, centralImage = ScoreFlatness()
        #print(f'{centralImage.fName=}')
        newName = centralImage.fName.replace('Central_Image', 'Central_Image_step0')
        centralImage.fName = newName
        #print(f'{centralImage.fName=}')
        centralImage.WriteToFile()
        #sys.exit()
        """
        
    def SaveTables(self):
        SaveTable(self.table0, sfAiTab0)
        SaveTable(self.table1, sfAiTab1)
        
        
    def TryTableStep(self, iTable):
        if verbosity > 1:
            print(f'<TryTableStep> T{iTable}')
        if iTable == 0:
            table = self.table0
            sfName = sfAiTab0
        else:
            table = self.table1
            sfName = sfAiTab1
            
        self.tmpTable = table.clone().detach()
        iFirstRow = random.randint(0, nRows-1)
        iRowAfter = random.randint(iFirstRow+1, nRows)
        iFirstCol = random.randint(0, nDetectors-1)
        iColAfter = random.randint(iFirstCol+1, nDetectors)
        delta = random.random() - 0.5
        delta = delta / 1000
        self.tmpTable[0,iFirstRow:iRowAfter,iFirstCol:iColAfter] += delta
        SaveTable(self.tmpTable, sfName)
        #print(f'Try table {sfName}')
        if verbosity > 1:
            print(f'CHANGED: rows {iFirstRow}:{iRowAfter}, cols {iFirstCol}:{iColAfter}, {delta=}')
        return delta
            
    def SaveBetter(self, iXrt, iBetter):
        sfName = f'Poly_XRT{iXrt}_Better{iBetter}_width{nDetectors}_height{nRows}.float.bin'
        if iXrt == 0:
            sfName = path.join(Config.sTabDir0, sfName)
            self.table0 = self.tmpTable
            SaveTable(self.table0, sfName)
        else:
            sfName = path.join(Config.sTabDir1, sfName)
            self.table1 = self.tmpTable           
            SaveTable(self.table1, sfName)
    
    def RestoreTable(self, iXrt):
        if iXrt == 0:
            SaveTable(self.table0, sfAiTab0)
        else:
            SaveTable(self.table1, sfAiTab1)
        
"""
def TrainPolyCal(nTrials,nMaxBetter=1000000):
    #VerifyDir('d:\PolyCalib')
    #VerifyDir('d:\PolyCalib\Better_XRT0')
    #VerifyDir('d:\PolyCalib\Better_XRT1')
    SetPolyByAi()
    learner = CPolyTables()
    for i in range(nTrials):
        learner.TryStep()
        if learner.nBetter >= nMaxBetter:
            break
        
    learner.OnEndTraining()
    """


def main():
    print('*** Nothing here')
    #bNominal = False
    #nTrain = 50000
    
    #VerifyDir('d:\Dump')
    #VerifyDir('d:\PyLog')
    
    """
    if bNominal:
        print('*** Try Nominal')
        SetPolyNominal()
        RunRecon()
        ScoreFlatness(bNominal=True)
    else:
        TrainPolyCal(nTrain)
        #TrainPolyCal(nTrain, nMaxBetter = 2)
        """

if __name__ == '__main__':
    main()
