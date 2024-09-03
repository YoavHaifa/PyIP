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

sAiFlag = 'd:\Config\Poly\GetAiTable.txt'
sAiFlagRemoved = 'd:\Config\Poly\GetAiTable_x.txt'

sfVolumeNominal = 'BP_nom_Output_width512_height512.float.rvol'
sfVolumeAi = 'BP_PolyAI_Output_width512_height512.float.rvol'

sfRoot = 'd:\PolyCalib'
iExperiment = 4
sExp = 'MultiSamples'
sBaseDir = ''
sLogDir = ''
sVolDir = ''
sTabDir0 = ''
sTabDir1 = ''
bInitialized = False

def OnInitRun():
    global sBaseDir, sLogDir, sVolDir, sTabDir0, sTabDir1
    global sfVolumeNominal, sfVolumeAi, bInitialized
    
    if bInitialized:
        print('Attempt to call <OnInitRun> twice. Exiting...')
        sys.exit()
    bInitialized = True
    
    VerifyDir(sfRoot)
    VerifyDir('d:\Config')
    if VerifyDirIsNew('d:\Config\Poly'):
        with open(sAiFlag, 'w') as file:
            file.write('Content is not important\n')
        
    sDirName = f'Exp{iExperiment}_{sExp}'
    sBaseDir = VerifyJointDir(sfRoot, sDirName)
    
    sLogDir = VerifyJointDir(sBaseDir, 'Log')
    sVolDir = VerifyJointDir(sBaseDir, 'Vol')
    
    sfVolumeNominal = path.join(sVolDir, sfVolumeNominal)
    sfVolumeAi = path.join(sVolDir, sfVolumeAi)
    
    sTabDir0 = VerifyJointDir(sBaseDir, 'Tab0')
    sTabDir1 = VerifyJointDir(sBaseDir, 'Tab1')

    
def LogFileName(sfName):
    return path.join(sLogDir, sfName)
    
def OpenLog(sfName):
    sfFullName = path.join(sLogDir, sfName)
    f = open(sfFullName, 'w')
    return f

def OpenLogGetName(sfName):
    sfFullName = path.join(sLogDir, sfName)
    f = open(sfFullName, 'w')
    return f, sfFullName

def TryRename(sFrom, sTo):
    #print('Try Rename')
    if path.exists(sFrom):
        os.rename(sFrom, sTo)
        print(f'Renamed {sFrom} to {sTo}')
        
def SetBpDumpFile(sfName):
    with open('d:\Config\Poly\BPDumpFileName.txt', 'w') as file:
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
    TryRename(sfVolumeAi, sfSaveName)

def Clean():
    DeleteFilesInDir(sVolDir)
    DeleteFilesInDir(sLogDir)

def main():
    OnInitRun()
    OpenLog('Try.log')
    SetPolyNominal()
    Clean()
    
if __name__ == '__main__':
    main()
