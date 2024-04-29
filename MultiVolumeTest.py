# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:34:25 2024

@author: USER
"""

import time

from Utils import GetDataDir
from MultiVolume import CMultiVolume

def TestSelectRandomVoxel(mv):
    print('<MultiVolume::TestSelectRandomVoxel>')
    
    mvSave = CMultiVolume(mv.sDir)
    
    for i in range(3):
        vEnv = mv.GetItem()
        print(f'Selection {i:2d}: im {mv.iImage:3d}, line {mv.iLine:3d}, col {mv.iCol:3d}')
        vEnv.Print()
        
    bSame = mvSave.IsEqual(mv)
    if bSame:
        print('OK - volume is same after sampling')
    else:
        print('ERROR - volume changed after sampling!!!')

def TestMultiVolume(mv):
    nLayersFrom1st = 3
    nInputChannels = 4
    nLinesOnEachSide = 2
    nColsOnEachSide = 2
    mv.SetEnvSize(nLayersFrom1st, nInputChannels, nLinesOnEachSide, nColsOnEachSide)
    mv.Print()
    TestSelectRandomVoxel(mv)
    
def TestRandomAccessTime(mv):
    #nCols = 7
    iMidCol = 2
    #nLines = 5
    iPrevLine = 1
    iNextLine = 3
    iMidLayer = 1
    n = 200
    startTime = time.time()
    for i in range(n):
        vEnv = mv.GetItem()
        x = vEnv.x
        if i < 1:
            print('x.shape', x.shape)
        if i < 5:
            gap = abs (x[iMidLayer, iPrevLine, iMidCol] - x[iMidLayer, iNextLine, iMidCol])
            print(f'{i}: nTried {mv.nTried}, gap {gap:6f}')
            #print(x[iMidLayer])
    deltaSec = time.time() - startTime
    avgms = deltaSec / n * 1000
    print(f'<TestRandomAccessTime> {n} in {deltaSec} sec, average {avgms} ms')

def main():
    bTest = True
    
    print('Test Class Multi Volume')
    sDir = GetDataDir('MultiVolumeTeat', 'Random_tube0_03')
    mv= CMultiVolume(sDir)
    if bTest:
        TestMultiVolume(mv)

    TestRandomAccessTime(mv)

if __name__ == '__main__':
    main()

