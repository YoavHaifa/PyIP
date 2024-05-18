# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 06:47:20 2024

@author: USER
"""
import random

from Utils import GetTaskRoot, VerifyDir, FindExistingDir, GetDataRoot, GetLogDir

def TestFindExistingDir():
    print('*** Test FindExistingDir')
    sTryList = ["k:\\tmp", 'r:/notExist', 'e:/data', 'E:/data', 'E:\\data']
    for sTry in sTryList:
        sFound = FindExistingDir(sTry)
        if (sFound):
            print(f'FindExistingDir({sTry}) --> {sFound}')
        else:
            print(f'FindExistingDir({sTry}) --> False')

def TestGetDirs():
    print('*** Test Get Dirs')
    sDataRoot = GetDataRoot()
    print(f'GetDataRoot() --> {sDataRoot}')
    sLogDir = GetLogDir()
    print(f'GetLogDir() --> {sLogDir}')

def TestVerifyDir():
    print('*** Test VerifyDir')
    sDirsList = ['d:/Tmp', f'd:/Tmp/TestDir_{random.randint(1,100)}']
    sDirsList.append( f'f:/Tmp/XXX/TestDir_{random.randint(1,100)}')
    sDirsList.append( f'f:/Tmp/TestDir<Illegal>_{random.randint(1,100)}')
    for sDir in sDirsList:
        try:
            if VerifyDir(sDir):
                print(f'{sDir} verified')
        except FileNotFoundError:
            print(f'Verify {sDir} failed with <FileNotFoundError>')
        except:
            print(f'Verify {sDir} failed but NOT <FileNotFoundError>')

def TestGetTaskRoot():
    print('*** Test GetTaskRoot')
    sTasksList = ['same', 'NOT-A-TASK']
    for sTask in sTasksList:
        sDir = GetTaskRoot(sTask)
        print(f'Root dir for task <{sTask}>: {sDir}')

    
def main():
    TestFindExistingDir()
    TestGetDirs()
    TestVerifyDir()
    TestGetTaskRoot()


if __name__ == '__main__':
    main()
