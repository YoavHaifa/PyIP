# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:42:40 2024
Inverse IR Tables - from [image, radius] to location [row, col] in poly table
@author: yoavb
"""

from os import path
import numpy as np
import torch
import random

from Log import Log

nRows = 192
nDets = 688
detMargins = 170
nImages = 280
maxRadius = 260

sTabsDir = 'D:/PolyCalib/InverseIRTables'

verbosity = 5

class CInverseIRTable:
    """
    Holds a single table
    """
    def __init__(self, iTube, sName):
        if iTube == 0:
            sChar = 'a'
        else:
            sChar = 'b'
            
        self.sfName = f'InverseIR_Tube{iTube}{sChar}_{sName}_per_image_and_radius_width{maxRadius}_height{nImages}_dzoom2.short.rtab'
        self.Read()
        
    def Read(self):
        sfFullName = path.join(sTabsDir, self.sfName)        
        npTab = np.memmap(sfFullName, dtype='int16', mode='r').__array__()
            
        self.pTab = torch.from_numpy(npTab.copy())
        self.pTab = self.pTab.view(nImages,maxRadius)
        if verbosity > 1:
            print(f'<CInverseIRTable> loaded {sfFullName}')
            print(f'Size: {self.pTab.size()}')

class CInverseIRTablesPerTube:
    """
    Holds row and col tables per a single tube
    """
    def __init__(self, iTube):
        self.iTube = iTube
        self.rowTab = CInverseIRTable(iTube, 'row')
        self.colTab = CInverseIRTable(iTube, 'col')

    def Inverse(self, iImage, iRad):
        iRow = self.rowTab.pTab[iImage, iRad]
        iCol = self.colTab.pTab[iImage, iRad] + detMargins
        return iRow, iCol

giImage = 0 
giRad = 0 
gTube0Tab = CInverseIRTablesPerTube(0)
gTube1Tab = CInverseIRTablesPerTube(1)


class CInverseIRTablesPerTubes:
    """
    Holds inverse IR tables for both tubes
    """
    def __init__ (self):
        global gTube0Tab, gTube1Tab
        print('*** ===>>> Create inverse impulse response table for 2 tubes')
        self.tube0 = CInverseIRTablesPerTube(0)
        self.tube1 = CInverseIRTablesPerTube(1)
        #gTube0Tab = self.tube0
        #gTube1Tab = self.tube1

    def InverseByRandomTubeGPar():
        iTube = random.randint(0,1)
        if iTube == 0:
            iRow, iCol = gTube0Tab.Inverse(giImage, giRad)
        else:
            iRow, iCol = gTube1Tab.Inverse(giImage, giRad)
        return iTube, iRow, iCol

    def InverseByRandomTube(self, iImage, iRad):
        iTube = random.randint(0,1)
        if iTube == 0:
            iRow, iCol = self.tube0.Inverse(iImage, iRad)
        else:
            iRow, iCol = self.tube1.Inverse(iImage, iRad)
        return iTube, iRow, iCol

"""
def GInverseByRandomTube(iImage, iRad):
    iTube = random.randint(0,1)
    if iTube == 0:
        iRow, iCol = gTube0Tab.Inverse(iImage, iRad)
    else:
        iRow, iCol = gTube1Tab.Inverse(iImage, iRad)
    Log(f'<GInverseByRandomTube> vol [{iImage},{iRad}] ==> [{iTube},{iRow},{iCol}]')
    return iTube, iRow, iCol
"""

def GInverseByTube(iTube, iImage, iRad):
    if iTube == 0:
        iRow, iCol = gTube0Tab.Inverse(iImage, iRad)
    else:
        iRow, iCol = gTube1Tab.Inverse(iImage, iRad)
    Log(f'<GInverseByRandomTube> vol [{iImage},{iRad}] ==> [{iTube},{iRow},{iCol}]')
    return iRow, iCol

def CheckInverse(tabs):
    iFirstImage = 100
    iFirstRadius = 0
    for iIm in range(3):
        iImage = iFirstImage + iIm * 5
        for iR in range(4):
            iRad = iFirstRadius + iR * 2
            iRow, iCol = tabs.Inverse(iImage, iRad)
            print(f'<CheckInverse> [{iImage}, {iRad}] ==> [{iRow}, {iCol}]')
    
def CheckTabsForTube(iTube):
    print('*** ===>>> Check inverse impulse response table for tube {iTube}')
    tabs = CInverseIRTablesPerTube(iTube)
    print(tabs)
    CheckInverse(tabs)
    
def CheckInverseBy2Tubes(tabs2):
    iImage = 100
    iRad = 20
    iTube, iRow, iCol = tabs2.InverseByRandomTube(iImage, iRad)
    print(f'<CheckInverseBy2Tubes> [{iImage}, {iRad}] ==> [{iTube}, {iRow}, {iCol}]')
    
    iRad += 2
    iTube, iRow, iCol = GInverseByRandomTube(iImage, iRad)
    print(f'<CheckInverseBy2Tubes> [{iImage}, {iRad}] ==> [{iTube}, {iRow}, {iCol}]')

def main():
    
    #CheckTabsForTube(0)
    
    tabs2 = CInverseIRTablesPerTubes()
    print(tabs2)
    CheckInverseBy2Tubes(tabs2)
    #for iTube in range(1):
    #    CreateInverseTables(iTube+1)
    
if __name__ == '__main__':
    main()

