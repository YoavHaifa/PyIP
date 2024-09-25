# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:57:40 2024
Configuration file for Poly Training
@author: yoavb
"""

import os
from os import path
import sys

from Utils import VerifyJointDir, VerifyDir, VerifyDirIsNew, DeleteFilesInDir

sAiFlag = 'd:/Config/Poly/GetAiTable.txt'
sAiFlagRemoved = 'd:/Config/Poly/GetAiTable_x.txt'

sfVolumeNominal = 'BP_nom_Output_width512_height512.float.rvol'
sfVolumeAi = 'BP_PolyAI_Output_width512_height512.float.rvol'

sfRoot = 'd:/PolyCalib'
iExperiment = 18
sExp = 'Mat250'
sExp = 'try'
sBaseDir = ''
sLogDir = 'd:/Log'
sVolDir = ''
#sTabDir0 = ''
#sTabDir1 = ''
sBestTabsDir = ''
bInitialized = False
gLog = None

verbosity = 1

def OnInitRun(sSpecialVolDir=None):
    global sBaseDir, sLogDir, sVolDir
    global sfVolumeNominal, sfVolumeAi, bInitialized
    global sBestTabsDir
    
    if bInitialized:
        print('Attempt to call <OnInitRun> twice. Exiting...')
        sys.exit()
    bInitialized = True
    
    
    VerifyDir(sfRoot)
    VerifyDir('d:/Config')
    if VerifyDirIsNew('d:/Config/Poly'):
        with open(sAiFlag, 'w') as file:
            file.write('Content is not important/n')
        
    sDirName = f'Exp{iExperiment}_{sExp}'
    sBaseDir = VerifyJointDir(sfRoot, sDirName)
    
    sLogDir = VerifyJointDir(sBaseDir, 'Log')
    
    if sSpecialVolDir:
        if path.isdir(sSpecialVolDir):
            sVolDir = sSpecialVolDir
            print('Using special Volumes Dir: ', sSpecialVolDir)
        else:
            print('Missing special Volumes Dir: ', sSpecialVolDir)
            sys.exit()
    else:
        sVolDir = VerifyJointDir(sBaseDir, 'Vol')
    
    sfVolumeNominal = path.join(sVolDir, sfVolumeNominal)
    sfVolumeAi = path.join(sVolDir, sfVolumeAi)
    
    sBestTabsDir = VerifyJointDir(sBaseDir, 'BestTabs')

    
def LogFileName(sfName):
    _, sExt = os.path.splitext(sfName)
    if sExt not in ['.log', '.txt', '.csv']:
        sfName = sfName + '.log'
    return path.join(sLogDir, sfName)
    
def OpenLog(sfName):
    if len(sLogDir) < 1:
        OnInitRun()
    sfFullName = LogFileName(sfName)
    f = open(sfFullName, 'w')
    if verbosity > 1:
        print(f'<OpenLog> {sfFullName}')
    return f

def OpenLogGetName(sfName):
    sfFullName = LogFileName(sfName)
    f = open(sfFullName, 'w')
    return f, sfFullName

def TryRename(sFrom, sTo):
    #print('Try Rename')
    if path.exists(sFrom):
        os.rename(sFrom, sTo)
        print(f'Renamed {sFrom} to {sTo}')
        
def SetBpDumpFile(sfName):
    with open('d:/Config/Poly/BPDumpFileName.txt', 'w') as file:
        file.write(f'{sfName}\n')

def SetPolyNominal():
    if not bInitialized:
        OnInitRun()
    TryRename(sAiFlag, sAiFlagRemoved)
    SetBpDumpFile(sfVolumeNominal)

def SetPolyByAi():
    TryRename(sAiFlagRemoved, sAiFlag)
    SetBpDumpFile(sfVolumeAi)

def SaveAiVolume(iSave):
    sfPostfix = f'_save{iSave}.float.rvol'
    sfSaveName = sfVolumeAi.replace('.float.rvol', sfPostfix)
    if not path.exists(sfSaveName):
        TryRename(sfVolumeAi, sfSaveName)

def Clean():
    DeleteFilesInDir(sVolDir)
    DeleteFilesInDir(sLogDir)
    #DeleteFilesInDir(sTabDir0)
    #DeleteFilesInDir(sTabDir1)


def WriteMatrixToFile(matrix, sfName, sfType='float'):
    nCols = matrix.shape[-1]
    nLines = matrix.shape[-2]
    sfName = sfName + f'_width{nCols}_height{nLines}.{sfType}.rmat'
    sfFullName = path.join(sLogDir, sfName)
    
    npmat = matrix.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npmat.tobytes())
    print('Matrix saved:', sfFullName)

def Log(s):
    if gLog:
        gLog.Log(s)

def Start(sSection, sComment=''):
    if gLog:
        gLog.Start(sSection, sComment)

def End(sSection, sComment=''):
    if gLog:
        gLog.End(sSection, sComment)
    
def main():
    OnInitRun()
    OpenLog('Try.log')
    SetPolyNominal()
    Clean()
    
if __name__ == '__main__':
    main()
