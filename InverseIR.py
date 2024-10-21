# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 00:42:40 2024
Inverse IR Tables - from [image, radius] to location [row, col] in poly table
@author: yoavb
"""

from os import path
import numpy as np
import torch

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
        iCol = self.colTab.pTab[iImage, iRad]
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
    

def main():
    #global sTableDirPrints
    print('*** ===>>> Use inverse impulse response table')
    tabs = CInverseIRTablesPerTube(0)
    print(tabs)
    CheckInverse(tabs)
    #for iTube in range(1):
    #    CreateInverseTables(iTube+1)
    
if __name__ == '__main__':
    main()

