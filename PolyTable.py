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
from Utils import VerifyDir
from Patch import CPatch

nDetectors = 688
nRows = 192
nLayers = 3
nDetsPerReading = nDetectors * nRows

patchSize = 20

sCalTabDir = 'D:/SpotlightScans/SCANPLAN_830/Calibrations'
sfNominalTab0 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT0.bin'
sfNominalTab1 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT1.bin'
sfAiTab0 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT0_ai.bin'
sfAiTab1 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT1_ai.bin'
sDebugSaveDir = ''

verbosity = 1

    
class CPolyTables:
    """
    Learn poly table by trying to improve flatness
    """
    def __init__(self):
        """
        Starts with 2 empty tables
        As the trainer runs reconstruction and scores flatness
        This class is creating variations on the tables and keeps beneficial variants

        Returns 2 improved tables

        """
        self.PrepareEmptyTables()
        self.patch = CPatch(patchSize)
        self.nTrySteps0 = 0
        self.nTrySteps1 = 0
        self.nRealSteps0 = 0
        self.nRealSteps1 = 0
        self.sLast = 'None'
    
    def PrepareEmptyTables(self):
        table = torch.zeros([nLayers,nRows,nDetectors])
        table[0] = torch.ones([nRows,nDetectors])
        self.table0 = table
        self.table1 = self.table0.clone().detach()
        self.SaveTables()

    
    def SaveTable(self, table, sfName):
        if sfName.find(':') < 0:
            if len(sDebugSaveDir) > 0:
                VerifyDir(sDebugSaveDir)
                sfName = path.join(sDebugSaveDir, sfName)
            else:
                sfName = path.join(sCalTabDir, sfName)
            
        npTable = table.numpy()
        with open (sfName, 'wb') as file:
            file.write(npTable.tobytes())
        if verbosity > 1:
            print('Poly Table Saved:', sfName)
        
    def OnEndTraining(self):
        self.SaveTables()
    
    def SaveTables(self):
        self.SaveTable(self.table0, sfAiTab0)
        self.SaveTable(self.table1, sfAiTab1)
        
    def AddRectangle(self):
        self.tmpTable[0,self.iFirstRow:self.iRowAfter,self.iFirstCol:self.iColAfter] += self.delta
        if verbosity > 1:
            print(f'Add Rectangle: rows {self.iFirstRow}:{self.iRowAfter}, cols {self.iFirstCol}:{self.iColAfter}, {self.delta=}')
        
    def AddRandomRectangle(self):
        self.iFirstRow = random.randint(0, nRows-1)
        self.iRowAfter = random.randint(self.iFirstRow+1, nRows)
        self.iFirstCol = random.randint(0, nDetectors-1)
        self.iColAfter = random.randint(self.iFirstCol+1, nDetectors)
        self.AddRectangle()
        self.sLast = 'R-Rectangle'
       
    def AddPatch(self):
        side = self.patch.side
        add = self.patch.raster * self.delta
        self.tmpTable[0,self.iFirstRow:self.iFirstRow+side,self.iFirstCol:self.iFirstCol+side] += add
        if verbosity > 1:
            print(f'Add Circular Patch: rows {self.iFirstRow}, cols {self.iFirstCol}, {self.delta=}')
       
    def AddRandomPatch(self):
        side = self.patch.side
        self.iFirstRow = random.randint(0, nRows-side-1)
        self.iFirstCol = random.randint(0, nDetectors-side-1)
        self.AddPatch()
        self.sLast = f'R-Patch{self.patch.side}'

    def SaveTmpTable(self):
        if self.iTable == 0:
            sfName = sfAiTab0
            self.nTrySteps0 += 1
        else:
            sfName = sfAiTab1
            self.nTrySteps1 += 1
        self.SaveTable(self.tmpTable, sfName)
       
        
    def TryRandomTableStep(self, iTable):
        self.iTable = iTable
        if verbosity > 1:
            print(f'<TryRandomTableStep> T{iTable}')
        if self.iTable == 0:
            table = self.table0
        else:
            table = self.table1
            
        self.tmpTable = table.clone().detach()
        self.delta = (random.random() - 0.5) / 1000
        
        self.iType = random.randint(0,10)
        if self.iType < 7:
            self.AddRandomPatch()
        else:
            self.AddRandomRectangle()
        
        self.SaveTmpTable()
        return self.delta
    
    def TryOnFailure(self):
        self.delta = - self.delta
        if self.iType < 7:
            self.AddPatch()
            self.sLast = f'F-Patch{self.patch.side}'
        else:
            self.AddRectangle()
            self.sLast = 'F-Rectangle'
        self.SaveTmpTable()
        return self.delta
    
    def TryOnSuccess(self):
        self.delta = self.delta * 2
        if self.iType < 7:
            self.AddPatch()
            self.sLast = f'G-Patch{self.patch.side}'
        else:
            self.AddRectangle()
            self.sLast = 'G-Rectangle'
        self.SaveTmpTable()
        return self.delta
        
    def SaveBetter(self, iXrt, iBetter):
        if iXrt == 0:
            self.table0 = self.tmpTable
        else:
            self.table1 = self.tmpTable           
            
        # Save as best for resuming training later
        sfName = f'Poly_XRT{iXrt}_Best_width{nDetectors}_height{nRows}.float.bin'
        self.SaveBest(sfName)
        
        # Save with better index for the record
        sfName = f'Poly_XRT{iXrt}_Better{iBetter}_width{nDetectors}_height{nRows}.float.bin'
        if iXrt == 0:
            self.nRealSteps0 += 1
        else:
            self.nRealSteps1 += 1
        self.SaveAside(iXrt, sfName)
            
    def SaveDebug(self, iXrt):
        if iXrt == 0:
            iStep = self.nTrySteps0
        else:
            iStep = self.nTrySteps1
        sfName = f'Poly_XRT{iXrt}_step{iStep}_width{nDetectors}_height{nRows}.float.bin'
        self.SaveAside(iXrt, sfName)
            
    def SaveBest(self, sfName):
        sfName = path.join(Config.sBestTabsDir, sfName)
        self.SaveTable(self.tmpTable, sfName)
            
    def SaveAside(self, iXrt, sfName):
        if iXrt == 0:
            sfName = path.join(Config.sTabDir0, sfName)
            self.SaveTable(self.table0, sfName)
        else:
            sfName = path.join(Config.sTabDir1, sfName)
            self.SaveTable(self.table1, sfName)
    
    def RestoreTable(self, iXrt):
        if iXrt == 0:
            self.SaveTable(self.table0, sfAiTab0)
        else:
            self.SaveTable(self.table1, sfAiTab1)
        


def main():
    global sDebugSaveDir
    Config.OnInitRun()
    print('*** Check Poly Table Class')
    sDebugSaveDir = 'd:/Dump'
    tabs = CPolyTables()
    for i in range(10):
        tabs.TryRandomTableStep(0)
        tabs.SaveDebug(0)
    
    #bNominal = False
    #nTrain = 50000
    
    #VerifyDir('d:/Dump')
    #VerifyDir('d:/PyLog')
    
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
