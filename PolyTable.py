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
from Patch import CCircularPatch
from Log import CLog, Log

nDetectors = 688
nRows = 192
nLayers = 3
nDetsPerReading = nDetectors * nRows

#aPatchSizes = [10,20,30]
#aPatchSizes = [20]
nPatchesToInit = 11
circPC = 100

sCalTabDir = 'D:/SpotlightScans/SCANPLAN_830/Calibrations'
sDebugSaveDir = ''

MIN_DELTA = 0.05

verbosity = 1

class CPolyTable:
    """
    Single table for one XRT
    """
    def __init__(self, iXrt, log=None):
        """
        Hold learning table for one XRT
        Table can be initialized to flat or "best"
        If there is "best" in the designated directory - load it by default
        """
        self.iXrt = iXrt
        self.log = log
        self.sfAiTab = f'PolyCalibration_kVp120_FOV250_Collimator140_XRT{iXrt}_ai.bin'
        self.sfNomTab = f'PolyCalibration_kVp120_FOV250_Collimator140_XRT{iXrt}.bin'
        
        sfBestName = f'Poly_XRT{self.iXrt}_Best_width{nDetectors}_height{nRows}.float.rmat'
        self.sfBestFullName = path.join(Config.sBestTabsDir,sfBestName)
        
        self.sHistoryDir = VerifyJointDir(Config.sBaseDir, f'Tab{iXrt}')

        if not self.LoadBest():
            self.PrepareFlatTable()

        self.Save()
        self.tempTable = self.table.clone().detach()
        self.prevAvg = self.table.mean().item()
        if self.log:
            self.log.Log(f'<CPolyTable::__init__> avg {self.prevAvg}')
        self.nTry = 0
        self.nBetter = 0
        self.sumGain = 0
        self.aPatches = []
        self.iDebugSave = 0

    def Set(self, tableOffsets):
        self.tempTable = torch.zeros([nLayers,nRows,nDetectors])
        self.tempTable[0] = torch.ones([nRows,nDetectors]) + tableOffsets
        self.SaveTable(self.tempTable)
        
        
    def InitPatches(self, best):
        if len(self.aPatches) > 0:
            return
        
        for iPatch in range(nPatchesToInit):
            radius = iPatch + 1
            self.aPatches.append(CCircularPatch(best, radius, Config.firstDelta, self.iXrt))
        self.lastPatch = self.aPatches[0]

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
            
        if self.log:
            avg = table2Save.mean().item()
            self.log.Log(f'<SaveTable> {avg} --> {sfName}')
            
        npTable = table2Save.numpy()
        #print('<SaveTable>', sfName)
        with open (sfName, 'wb') as file:
            file.write(npTable.tobytes())
        if verbosity > 1:
            print('Poly Table Saved:', sfName)

    def Save(self, sfName=None):
        self.SaveTable(self.table, sfName)

    def Restore(self):
        self.SaveTable(self.table)

    def GetRandomDelta(self):
        delta = random.random() - 0.5
        if abs(delta) < MIN_DELTA:
            if delta < 0:
                delta = -MIN_DELTA
            else:
                delta = MIN_DELTA
                
        return delta / 1000
    
    def SelectRandomPatch(self):
        nPatches = len(self.aPatches)
        if nPatches > 1:
            iPatch = random.randint(0,nPatches-1)
            self.lastPatch = self.aPatches[iPatch]
            if self.log:
                self.log.Log(f'<TryRandomTableStep> {iPatch=}')
        else:
            self.lastPatch = self.aPatches[0]

    def TryRandomTableStep(self):
        
        self.nTry += 1         
        self.tempTable = self.table.clone().detach()
        self.SelectRandomPatch()

        self.lastPatch.AddRandom(self.tempTable, self.log)
        self.SaveTable(self.tempTable)
        return self.lastPatch.delta

    def TryTargetedTableStep(self, iRow, iDet, deviation, width):
        
        self.nTry += 1    
        self.tempTable = self.table.clone().detach()
        #self.SelectRandomPatch()
        if width >= nPatchesToInit:
            self.lastPatch = self.aPatches[-1]
        else:
            self.lastPatch = self.aPatches[width-1]            
        
        # Set direction of correction
        absDelta = abs(self.lastPatch.delta)
        if deviation > 0:
            self.lastPatch.delta = -absDelta
        else:
            self.lastPatch.delta = absDelta
        Log(f'<TryTargetedTableStep> tube {self.iXrt} at[{iRow}, {iDet}] add {self.lastPatch.delta} width {width}')
        
        self.lastPatch.AddAt(self.tempTable, iRow, iDet)
        self.SaveTable(self.tempTable)
        return self.lastPatch.delta
        

    def TrySamePatch(self):
        self.nTry += 1         
        self.tempTable = self.table.clone().detach()
        self.lastPatch.Add(self.tempTable)
        self.SaveTable(self.tempTable)

    def TryOnFailure(self):
        self.nTry += 1         
        self.tempTable = self.table.clone().detach()
        self.lastPatch.AddOnFailure(self.tempTable)
        self.SaveTable(self.tempTable)
        return self.lastPatch.delta

    def OnNewScore(self, prevBest, newScore):
        self.lastPatch.OnNewScore(prevBest, newScore)

    def OnBetter(self, gain):
        self.nBetter += 1
        self.sumGain += gain
        
        # Make the last temp new nominal
        self.table = self.tempTable.clone().detach()
        
        # Save to directory with best tables
        self.Save(self.sfBestFullName)
        
        # Save to table-specific directory to remember history
        sfNameBetter = f'Poly_XRT{self.iXrt}_Better{self.nBetter}_width{nDetectors}_height{nRows}.float.rmat'
        sfFullNameBetter = path.join(self.sHistoryDir, sfNameBetter)
        self.Save(sfFullNameBetter)
        self.sfLastSaved = sfFullNameBetter
        
    def SaveDebug(self):
        self.iDebugSave += 1 
        sfNameDebug = f'Poly_XRT{self.iXrt}_Debug{self.iDebugSave}_width{nDetectors}_height{nRows}.float.rmat'
        sfFullNameDebug = path.join(self.sHistoryDir, sfNameDebug)
        self.Save(sfFullNameDebug)
        self.sfLastSaved = sfFullNameDebug
        
        
    def Log(self, log, sAt):
        log.Log(f'<PolyTable::Log> at {sAt}: {self.sfLastSaved}')
        avg = self.table.mean().item()
        avgTmp = self.tempTable.mean().item()

        if (avg == self.prevAvg):
            s = 'SAME'
        else:
            diff = avg - self.prevAvg
            s = f'DIFF {diff:.9f}'
        log.Log(f'{avg=:.9f}, {avgTmp=:.9f} prev {self.prevAvg:.9f} {s}')
        log.Log('---')
        self.prevAvg = avg
        

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
        self.nTry = 0 
        self.nBetter = 0
        self.nNotBetter = 0
        
    def Set(self, tabs):
        self.tables[0].Set(tabs[0])
        self.tables[1].Set(tabs[1])
        
    def InitPatches(self, best):
        for table in self.tables:
            table.InitPatches(best)

    def OnEndTraining(self):
        self.Save()
    
    def Save(self):
        for table in self.tables:
            table.Save()
    
    def SaveDebug(self):
        self.tables[self.iCurTab].SaveDebug()
    
    def Log(self, log, sAt):
        if log:
            for table in self.tables:
                table.Log(log, sAt)

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
       
    def TryTargetedTableStep(self, iTube, iRow, iDet, deviation, width):
        self.nTry += 1
        self.iCurTab = iTube
        if verbosity > 1:
            print(f'<TryRandomTableStep> T{self.iCurTab}')

        self.sLast = 'T'
        return self.tables[self.iCurTab].TryTargetedTableStep(iRow, iDet, deviation, width)
        
    def TryRandomTableStep(self):
        self.nTry += 1
        self.iCurTab = random.randint(0, 1)
        if verbosity > 1:
            print(f'<TryRandomTableStep> T{self.iCurTab}')

        self.sLast = 'R'
        return self.tables[self.iCurTab].TryRandomTableStep()
        #self.delta = self.tables[self.iCurTab].delta
    
    def TryOnFailure(self):
        self.nTry += 1
        self.sLast = 'F'
        return self.tables[self.iCurTab].TryOnFailure()
        #self.delta = self.tables[self.iCurTab].delta
    
    """
    def TryOnSuccess(self):
        self.nTry += 1
        self.tables[self.iCurTab].TryOnSuccess()
        self.delta = self.tables[self.iCurTab].delta
        """

    def OnNewScore(self, prevBest, newScore):
        self.tables[self.iCurTab].OnNewScore(prevBest, newScore)
        gain = prevBest - newScore
        if gain > 0:
            self.nBetter += 1
            self.tables[self.iCurTab].OnBetter(gain)
        else:
            self.nNotBetter += 1
            self.tables[self.iCurTab].Restore()
        
        
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
    log = CLog('TestSingleTable')
    tab0 = CPolyTable(0, log)
    tab0.InitPatches(100)
    #tab0.Save()
    tab0.OnBetter(0)
    tab0.Log(log, 'Init')
    tab0.TryRandomTableStep()
    tab0.OnBetter(0.3)
    tab0.Log(log, 'random added')
    tab0.TryRandomTableStep()
    tab0.OnBetter(0.4)
    tab0.Log(log, 'random added')
    tab0.TryOnFailure()
    tab0.OnBetter(0.2)
    tab0.Log(log, 'On Failure')
    tab0.TryRandomTableStep()
    tab0.OnBetter(0.4)
    tab0.Log(log, 'random added')
    #tab0.TryOnSuccess()
    #tab0.OnBetter(0.1)
    #tab0.Log(log, 'On Success')

def TestTables():
    print('*** Check Poly Tables Class')
    log = CLog('TestTables')
    tables = CPolyTables()
    tables.Save()
    tables.iCurTab = 0
    tables.SaveDebug()
    tables.iCurTab = 1
    tables.SaveDebug()
    tables.Log(log,'Start')
    tables.InitPatches(100)
    
    for i in range(4):
        tables.TryRandomTableStep()
        log.Log(f'<TryRandomTableStep> {tables.iCurTab}')
        tables.OnBetter(0.1)
        tables.SaveDebug()
        tables.Log(log, f'TryRand_{i} - {tables.iCurTab}')

def main():
    global sDebugSaveDir, verbosity
    verbosity = 5
    Config.OnInitRun()
    sDebugSaveDir = 'd:/Dump'
    
    TestSingleTable()
    #TestTables()
    
    #TestTables()

if __name__ == '__main__':
    main()
