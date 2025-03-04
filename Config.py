# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:57:40 2024
Configuration file for Poly Training
@author: yoavb
"""

import os
from os import path
import sys
import shutil

from Utils import VerifyJointDir, VerifyDir, VerifyDirIsNew, DeleteFilesInDir, FPrivateName

sAiFlag = 'd:/Config/Poly/GetAiTable.txt'
sAiFlagRemoved = 'd:/Config/Poly/GetAiTable_x.txt'

sfVolumeNominal = 'BP_nom_Output_width512_height512.float.rvol'
sfVolumeAi = 'BP_PolyAI_Output_width512_height512.float.rvol'

sfImpulseIndices = 'd:/Config/Poly/Impulse.txt'


matrix = 256
firstDelta = 0.001
nToEvaluate = 200
nToEvaluateTry = 10

dump = 0 # 255
debug = 3


sfRoot = 'd:/PolyCalib'
iExperiment = 52
sExp = 'mult8lines_allGrad_LR008'
sExp = 'try'
bDeleteOnStart = False
sBaseDir = ''
sLogDir = 'd:/Log'
sVolDir = ''
sRootVolDir = ''
sDevDir = ''
sBestTabsDir = ''
asSaveTabsDir = ['','']
bInitialized = False

verbosity = 1
sfBPOutput = ''


iImage = 163
iRad = 30


def SetBPOutputFileName(sPrefix):
    global sfBPOutput
    s = sPrefix + f'_Output_width{matrix}_height{matrix}'
    if matrix == 256:
        s = s + '_zoom2'
    s = s + '.float.rvol'
    sfBPOutput = path.join(sVolDir, s)
    return sfBPOutput

def SaveLastBpOutput(i, zAt = None):
    if path.isfile(sfBPOutput):
        sNewExt = f'_save{i:02d}'
        if zAt is not None:
            sNewExt += f'_{zAt}'
        sNewExt += '.float.rvol'
        sfNew = sfBPOutput.replace('.float.rvol', sNewExt)
        shutil.copyfile(sfBPOutput, sfNew)
        print(f'Last BP output was saved as {sfNew}')

def OnVolDirSet():
    global sfVolumeNominal, sfVolumeAi
    sfVolumeNominal = SetBPOutputFileName('BP_nom')
    sfVolumeAi = SetBPOutputFileName('BP_PolyAI')
    print(f'{sfVolumeNominal=}')
    print(f'{sfVolumeAi=}')

def SetSpecialVolDir(sSpecialVolDir):
    global sVolDir
    if path.isdir(sSpecialVolDir):
        sVolDir = sSpecialVolDir
        print('Using special Volumes Dir: ', sSpecialVolDir)
        OnVolDirSet()
    else:
        print('Missing special Volumes Dir: ', sSpecialVolDir)
        sys.exit()

def OnInitRun():
    global sBaseDir, sLogDir, sVolDir, sDevDir, sRootVolDir
    global sfVolumeNominal, sfVolumeAi, bInitialized
    global sBestTabsDir, nToEvaluate, asSaveTabsDir
    
    if bInitialized:
        return 
    bInitialized = True
    
    VerifyDir(sfRoot)
    VerifyDir('d:/Config')
    if VerifyDirIsNew('d:/Config/Poly'):
        with open(sAiFlag, 'w') as file:
            file.write('Content is not important/n')
        
    sDirName = f'Exp{iExperiment}_{sExp}'
    sBaseDir = VerifyJointDir(sfRoot, sDirName)
    
    sLogDir = VerifyJointDir(sBaseDir, 'Log')
    sDevDir = VerifyJointDir(sBaseDir, 'Dev')
    
    if sVolDir == '':
        sVolDir = VerifyJointDir(sBaseDir, 'Vol')
        OnVolDirSet()

    sBestTabsDir = VerifyJointDir(sBaseDir, 'BestTabs')
    VerifyJointDir(sBaseDir, 'Manualy Saved Results')
    
    if asSaveTabsDir[0] == '':
        asSaveTabsDir[0] = VerifyJointDir(sBaseDir, 'Tab0')
        asSaveTabsDir[1] = VerifyJointDir(sBaseDir, 'Tab1')
    
    if sExp == 'try':
        nToEvaluate = nToEvaluateTry
        
    if sRootVolDir == '':
        print(f'{sfRoot=}')
        sRootVolDir = VerifyJointDir(sfRoot, 'Vol')
        print(f'{sRootVolDir=}')

    
def LogFileName(sfName, sSubDir=None):
    if not bInitialized:
        OnInitRun()
    _, sExt = os.path.splitext(sfName)
    if sExt not in ['.log', '.txt', '.csv']:
        sfName = sfName + '.log'
    if sSubDir is None:
        return path.join(sLogDir, sfName)
    sDir = VerifyJointDir(sLogDir, sSubDir)
    return path.join(sDir, sfName)
    
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

def SetPolyByAi(bDefaultOutput = True):
    TryRename(sAiFlagRemoved, sAiFlag)
    if bDefaultOutput:
        SetBpDumpFile(sfVolumeAi)

def SaveAiVolume(iSave):
    sfPostfix = f'_save{iSave}.float.rvol'
    sfSaveName = sfVolumeAi.replace('.float.rvol', sfPostfix)
    if not path.exists(sfSaveName):
        TryRename(sfVolumeAi, sfSaveName)

def Clean():
    if bDeleteOnStart:
        DeleteFilesInDir(sVolDir)
        DeleteFilesInDir(sLogDir)
        DeleteFilesInDir(sDevDir)
        DeleteFilesInDir(sBestTabsDir)
        #DeleteFilesInDir(sTabDir1)
        
    if path.exists(sfImpulseIndices):
        os.remove(sfImpulseIndices)

def WriteMatrixToFile(matrix, sfName, sfType='float'):
    nCols = matrix.shape[-1]
    nLines = matrix.shape[-2]
    sfName = sfName + f'_width{nCols}_height{nLines}.{sfType}.rmat'
    sfFullName = path.join(sLogDir, sfName)
    
    npmat = matrix.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npmat.tobytes())
    print('Matrix saved:', sfFullName)

def WriteMatrixSpec(matrix, sfName, sSubDir, sfType='float'):
    nCols = matrix.shape[-1]
    nLines = matrix.shape[-2]
    sfName = sfName + f'_width{nCols}_height{nLines}'
    if nCols <= 300 and nLines <= 300:
        sfName += '_zoom2'
    sfName = sfName + f'.{sfType}.rmat'
    sDir = VerifyJointDir(sBaseDir, sSubDir)
    sfFullName = path.join(sDir, sfName)
    
    npmat = matrix.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npmat.tobytes())
    print(f'Matrix {sSubDir} saved:', sfFullName)

def WriteDevToFile(devMap, sfName, sfType='float'):
    nCols = devMap.shape[-1]
    nLines = devMap.shape[-2]
    sfName = sfName + f'_width{nCols}_height{nLines}'
    if nCols <= 300 and nLines <= 300:
        sfName += '_zoom2'
    sfName = sfName + f'.{sfType}.rmat'
    sfFullName = path.join(sDevDir, sfName)
    
    npmat = devMap.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npmat.tobytes())
    print('Dev map saved:', sfFullName)

def WriteVolToFile(vol, sfName, sfType='float'):
    nCols = vol.shape[-1]
    nLines = vol.shape[-2]
    sfName += f'_width{nCols}_height{nLines}'
    if nCols <= 300 and nLines <= 300:
        sfName += '_zoom2'
    sfName += f'.{sfType}.rvol'
    sfFullName = path.join(sVolDir, sfName)
    
    npvol = vol.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npvol.tobytes())
    print('Volume saved:', sfFullName)

def GetLocalOrSharedVol(sfVol):
    if path.exists(sfVol):
        return sfVol
    
    sfVol = FPrivateName(sfVol)

    sfFullName = path.join(sVolDir, sfVol)
    if path.exists(sfFullName):
        return sfFullName
    
    sfTry = path.join(sRootVolDir, sfVol)
    if path.exists(sfTry):
        return sfTry
        
    print(f'<Config::GetLocalOrSharedVol> MISSING file: {sfFullName}')
    sys.exit()
    
    
def main():
    OnInitRun()
    OpenLog('Try.log')
    SetPolyNominal()
    Clean()
    
if __name__ == '__main__':
    main()
