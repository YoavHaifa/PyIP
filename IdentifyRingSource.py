# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:39:50 2024
Identify ring and its source in the readings space
@author: yoavb
"""

from os import path

from Volume import CVolume
from InverseIR import CInverseIRTablesPerTube

sDir = 'D:/PolyCalib/Impulse'
sfVol = 'Poli_AI_t1_r70_d300_width256_height256_zoom2.float.rvol'


def main():
    #global sTableDirPrints
    print('*** ===>>> Use inverse impulse response table on volume ring')
    sfName = path.join(sDir, sfVol)
    vol = CVolume('irVol', sfName)
    print(vol)
    vol.Print()
    
    iImage, iLine, iCol = vol.FindMaxPosition()
    radius = vol.findRadius(iLine, iCol)
    iRad = int(radius + 0.5)
    print(f'MAX {iImage=}, {iLine=}, {iCol=}, {radius=}, {iRad=}')
    
    tabs = CInverseIRTablesPerTube(1)
    print(tabs)
    
    iRow, iCol = tabs.Inverse(iImage, iRad)
    print(f'<CheckInverse> [{iImage}, {iRad}] ==> [{iRow}, {iCol}]')
    
    #CheckInverse(tabs)
    #for iTube in range(1):
    #    CreateInverseTables(iTube+1)
    
if __name__ == '__main__':
    main()



