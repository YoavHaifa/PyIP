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
import sys
import numpy as np

import Config
from Utils import VerifyDir, VerifyJointDir
from Patch import CCircularPatch, CRectangularPatch

nDetectors = 688
nRows = 192
nLayers = 3
nDetsPerReading = nDetectors * nRows

patchSize = 20

sCalTabDir = 'D:/SpotlightScans/SCANPLAN_830/Calibrations'
#sfNominalTab0 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT0.bin'
#sfNominalTab1 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT1.bin'
#sfAiTab0 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT0_ai.bin'
#sfAiTab1 = 'PolyCalibration_kVp120_FOV250_Collimator140_XRT1_ai.bin'
sDebugSaveDir = ''

verbosity = 1

class CPolyTable:
    """
    Single table for one XRT
    """
    def __init__(self, iXrt):
        """
        Hold learning table for one XRT
        Table can be initialized to flat or "best"
        If there is "best" in the designated directory - load it by default
        """
        self.iXrt = iXrt
        self.sfAiTab = f'PolyCalibration_kVp120_FOV250_Collimator140_XRT{iXrt}_ai.bin'
        self.sfNomTab = f'PolyCalibration_kVp120_FOV250_Collimator140_XRT{iXrt}.bin'
        
        sfBestName = f'Poly_XRT{self.iXrt}_Best_width{nDetectors}_height{nRows}.float.rmat'
        self.sfBestFullName = path.join(Config.sBestTabsDir,sfBestName)
        
        self.sHistoryDir = VerifyJointDir(Config.sBaseDir, f'Tab{iXrt}')

        if not self.LoadBest():
            self.PrepareFlatTable()

        self.Save()
        self.tempTable = self.table.clone().detach()
        self.nTry = 0
        self.nBetter = 0
        self.sumBetter = 0
        self.cPatch = CCircularPatch(patchSize)
        self.rPatch = CRectangularPatch()
        self.lastPatch = self.cPatch

    def LoadBest(self):
        if not path.isfile(self.sfBestFullName):
            return False

        npTable = np.memmap(self.sfBestFullName, dtype='float32', mode='r').__array__()
        table = torch.from_numpy(npTable.copy())
        self.table = table.view(-1,nRows,nDetectors)
        
        nLayersInTab = self.table.size()[0]
        if nLayersInTab != nLayers:
            print(f'<CPolyTable::LoadBest> ERROR {nLayersInTab=}')
            sys.exit()
        print('<CPolyTable::LoadBest> ', self.sfBestFullName)
        return True
        
    def PrepareFlatTable(self):
        self.table = torch.zeros([nLayers,nRows,nDetectors])
        self.table[0] = torch.ones([nRows,nDetectors])
        print('<CPolyTable::PrepareFlatTable> ', self.sfAiTab)

    def SaveTable(self, table2Save, sfName=None):
        if not sfName:
            sfName = self.sfAiTab
        if sfName.find(':') < 0:
            if len(sDebugSaveDir) > 0:
                VerifyDir(sDebugSaveDir)
                sfName = path.join(sDebugSaveDir, sfName)
            else:
                sfName = path.join(sCalTabDir, sfName)
            
        npTable = table2Save.numpy()
        #print('<SaveTable>', sfName)
        with open (sfName, 'wb') as file:
            file.write(npTable.tobytes())
        if verbosity > 1:
            print('Poly Table Saved:', sfName)

    def Save(self, sfName=None):
        self.SaveTable(self.table, sfName)

    def TryRandomTableStep(self):
        self.nTry += 1         
        self.tempTable = self.table.clone().detach()
        self.delta = (random.random() - 0.5) / 1000
        
        iType = random.randint(0,10)
        if iType < 7:
            self.lastPatch = self.cPatch
        else:
            self.lastPatch = self.rPatch
            
        self.lastPatch.AddRandom(self.tempTable,self.delta)
        self.SaveTable(self.tempTable)

    def TrySamePatch(self, delta):
        self.nTry += 1         
        self.tempTable = self.table.clone().detach()
        self.delta = delta
        self.lastPatch.Add(self.tempTable,self.delta)
        self.SaveTable(self.tempTable)

    def TryOnFailure(self):
        self.TrySamePatch(-self.delta)
        
    def TryOnSuccess(self):
        self.TrySamePatch(self.delta * 2)

    def OnBetter(self, d):
        self.nBetter += 1
        self.sumBetter += d
        self.lastPatch.OnBetter(d)
        
        # Make the last temp new nominal
        self.table = self.tempTable.clone().detach()
        
        # Save to directory with best tables
        self.Save(self.sfBestFullName)
        
        # Save to table-specific directory to remember history
        sfName = f'Poly_XRT{self.iXrt}_Better{self.nBetter}_width{nDetectors}_height{nRows}.float.rmat'
        sfFullName = path.join(self.sHistoryDir, sfName)
        self.Save(sfFullName)
        

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
        self.tables = []
        for iXrt in range(2):
            self.tables.append(CPolyTable(iXrt))
            
        self.sLast = 'None'
        self.iCurTab = 0
        self.iTry = 0 
        self.iBetter = 0
        

    def OnEndTraining(self):
        self.SaveTables()
    
    def SaveTables(self):
        for table in self.tables:
            table.Save()

    """
    def SaveTmpTable(self):
        
        if self.iTable == 0:
            sfName = sfAiTab0
            self.nTrySteps0 += 1
        else:
            sfName = sfAiTab1
            self.nTrySteps1 += 1
        self.SaveTable(self.tmpTable, sfName)
        """
       
        
    def TryRandomTableStep(self):
        self.iTry += 1
        self.iCurTab = random.randint(0, 1)
        if verbosity > 1:
            print(f'<TryRandomTableStep> T{self.iCurTab}')

        self.tables[self.iCurTab].TryRandomTableStep()
    
    def TryOnFailure(self):
        self.iTry += 1
        self.tables[self.iCurTab].TryOnFailure()
    
    def TryOnSuccess(self):
        self.iTry += 1
        self.tables[self.iCurTab].TryOnSuccess()
        
    def OnBetter(self, d):
        self.iBetter += 1
        self.tables[self.iCurTab].OnBetter(d)
        
"""            
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
        """

def TestSingleTable():
    print('*** Check Poly Table Class')
    tab0 = CPolyTable(0)
    #tab0.Save()
    tab0.OnBetter(0)
    tab0.TryRandomTableStep()
    tab0.OnBetter(0.3)
    tab0.TryOnFailure()
    tab0.OnBetter(0.2)
    tab0.TryOnSuccess()
    tab0.OnBetter(0.1)

def TestTables():
    print('*** Check Poly Tables Class')
    tables = CPolyTables()
    tables.Save()
    tables.iCurTab = 0
    for i in range(10):
        tables.TryRandomTableStep(0)
        tables.SaveDebug(0)

def main():
    global sDebugSaveDir, verbosity
    verbosity = 5
    Config.OnInitRun()
    sDebugSaveDir = 'd:/Dump'
    
    TestSingleTable()
    
    #TestTables()

if __name__ == '__main__':
    main()
