# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 01:00:41 2021

@author: yoavb
"""
import os
from os import path
import sys
import time
import glob
import torch

import Config

sDrives = "DEF"

lossDisplayFactor = 1000000

def TryRename(sFrom, sTo):
    if path.exists(sFrom):
        os.rename(sFrom, sTo)
        print(f'Renamed {sFrom} to {sTo}')

def FormatLoss(loss):
    ld = loss * lossDisplayFactor
    if ld > 0.1:
        return f'{ld:7.3f}'
    if ld > 0.0001:
        return f'{ld:7.6f}'
    return f'{ld:10.9f}'

def FPrivateName(sPath):
    sClean = sPath.replace('\\', '/')
    return os.path.basename(sClean)

def RemoveExt(sfName):
    return os.path.splitext(sfName)[0]

def RemoveEExt(sfName):
    sp = os.path.splitext(sfName)[0]
    if sp.endswith('.float') or sp.endswith('.short'):
        return RemoveExt(sp)
    return sp

def FindExistingDir(sDirName):
    if os.path.isdir(sDirName):
        return sDirName
    for sd in sDrives:
        sTry = sd + sDirName[1:]
        if os.path.isdir(sTry):
            print(f'Found dir: {sDirName} --> {sTry}')
            return sTry
    return False

def FindOrCreateDir(sDirName):
    sFound = FindExistingDir(sDirName)
    if sFound:
        return sFound
    for sd in sDrives:
        sTry = sd + sDirName[1:]
        sCreated = TryCraeteDir(sTry)
        if sCreated:
            return sCreated
    print(f'<FindOrCreateDir> filed for {sDirName}')
    return False 

def GetDataRoot():
    sDataRoot = FindExistingDir('D:/Data')
    if sDataRoot:
        return sDataRoot

    print('Data dir was not found - exiting...')
    sys.exit()
    
def GetDataDir(sFor):
    sRoot = GetDataRoot()
    return path.join(sRoot, sFor)

def GetTaskRoot(taskname):
    dataRoot = GetDataRoot()
    taskDataRoot = os.path.join(dataRoot, taskname)
    if not os.path.isdir(taskDataRoot):
        print(f'Root dir for task {taskDataRoot} was not found - exiting...')
        sys.exit()
        
    return taskDataRoot

def GetLogDir():
    sLogRoot = FindOrCreateDir('g:/PyLog')
    if sLogRoot:
        return sLogRoot
    
    print('Failed to create log dir - exiting...')
    sys.exit()

def GetDumpDir():
    sDumpRoot = FindOrCreateDir('D:/Dump')
    if sDumpRoot:
        return sDumpRoot
    
    print('Failed to create log dir - exiting...')
    sys.exit()

def VerifyJointDir(sPath, sDir):
    sDirPath = os.path.join(sPath, sDir)
    return VerifyDir(sDirPath)

def TryCraeteDir(sDir):
    if os.path.isdir(sDir):
        return sDir
        
    os.mkdir(sDir)
    if os.path.isdir(sDir):
        return sDir
    return False

def VerifyDir(sDir):
    if os.path.isdir(sDir):
        return sDir
        
    os.mkdir(sDir)
    if not os.path.isdir(sDir):
        print(f'Failed to create directory {sDir} - exiting...')
        sys.exit()

    print (f'Created Directory: {sDir}')
    return sDir

def VerifyDirIsNew(sDir):
    if os.path.isdir(sDir):
        return False
        
    os.mkdir(sDir)
    if not os.path.isdir(sDir):
        print(f'Failed to create directory {sDir} - exiting...')
        sys.exit()

    print (f'Created Directory: {sDir}')
    return True

def AssertDir(sDir, sDesc):
    if not os.path.exists(sDir):
        print(f'{sDesc} {sDir} was not found - exiting...')
        sys.exit()
    if not os.path.isdir(sDir):
        print(f'{sDesc} {sDir} is not directory - exiting...')
        sys.exit()
    return sDir

def DictGetS(d, parName, defaultValue):
    if parName in d:
        return d[parName]
    return defaultValue

def DictGetDict(d, parName):
    if parName in d:
        return d[parName]
    return {}

def DictGetInt(d, parName, defaultValue):
    if parName in d:
        return int(d[parName])
    return defaultValue

def DictGetFloat(d, parName, defaultValue):
    if parName in d:
        return float(d[parName])
    return defaultValue

def DictRestore(d, other):
    for key, value in other.items():
        if key in d:
            d[key] = value
            """
            if value.isnumeric():
                d[key] = int(value)
            elif value.find('.') > 0:
                d[key] = float(value)
            else:
                d[key] = value
                """

def GetTimeString(start):
    deltaSec = time.time() - start
    if deltaSec < 60:
        return f'{deltaSec:.2f} sec'
    
    deltaMin = deltaSec / 60
    if deltaMin < 60:
        return f'{deltaMin:.2f} min'
    
    deltaHours = deltaMin / 60
    return f'{deltaHours:.2f} hours'

def GetAbortFileName():
    if os.path.isdir('D:/Config/AI'):
        configDir = 'D:/Config/AI'
    else:
        configDir = 'E:/Config/AI/'

    sfAbortName = os.path.join(configDir, 'StopLearning.txt')
    if os.path.exists(sfAbortName):
        sfAbortNameX =  os.path.join(configDir, 'StopLearning_x.txt')
        os.rename(sfAbortName, sfAbortNameX)
    return sfAbortName
    #sfPauseName = configDir + 'PauseLearning.txt'
    #if path.exists(sfPauseName):
    #    sfPauseNameX = configDir + 'PauseLearning_x.txt'
    #    os.rename(sfPauseName, sfPauseNameX)
    
def DeleteFilesInDir(sDir):
    print(f'<DeleteFilesInDir> {sDir}')
    #files = [f for f in os.listdir(sDir) if os.path.isfile(f)]
    files = glob.glob(sDir+'/*')
    #print (files)
    for f in files:
        if os.path.isfile(f):
            print(f'remove {f}')
            os.remove(f)

def WriteMatrixToFile(matrix, sfName, sfType='float'):
    nCols = matrix.shape[-1]
    nLines = matrix.shape[-2]
    sfName = sfName + f'_width{nCols}_height{nLines}.{sfType}.rmat'
    sfFullName = path.join(Config.sLogDir, sfName)
    
    npmat = matrix.numpy()
    with open (sfFullName, 'wb') as file:
        file.write(npmat.tobytes())
    print('Matrix saved:', sfFullName)

def TestWriteMatrix():
    print('*** Test Write Matrix')
    nImages = 10
    mat = torch.zeros([nImages,256,512])
    for i in range(nImages):
        mat[i] = (i+1) * 100
    
    WriteMatrixToFile(mat, 'trySaveMatrix')
    
def main():
    print('*** Test Utils')
    Config.OnInitRun()
    TestWriteMatrix()

if __name__ == '__main__':
    main()
