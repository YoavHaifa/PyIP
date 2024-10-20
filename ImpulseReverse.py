# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:28:21 2024
Create Impulse reverse table - using image and radius tables
@author: yoavb
"""

from os import path
import numpy as np
import torch

from Utils import VerifyJointDir

nRows = 192
nDets = 688
detMargins = 170
nImages = 280
maxRadius = 260

sTableDir = 'D:/PolyCalib/ImpulseResponseTab'
sTableDirPrints = ''

verbosity = 1

class CInputTable:
    """
    Read and analyze 2 input tables per tube
    """
    def __init__(self, iTube, sName):
        self.iTube = iTube
        self.sName = sName
        self.ReadTable()
        
    def ReadTable(self):
        sfPrivateName = f'Tube{self.iTube}_{self.sName}Margined_width{nDets}_height{nRows}_zoom2.float.rtab'
        self.sfName = path.join(sTableDir, sfPrivateName)
        
        images = np.memmap(self.sfName, dtype='float32', mode='r').__array__()
                
        self.pTab = torch.from_numpy(images.copy())
        self.pTab = self.pTab.view(nRows,nDets)
        halfDets = int(nDets/2)
        self.pTab = self.pTab[:,detMargins:-halfDets]
        self.nCols = self.pTab.size()[1]
        if verbosity > 1:
            print(self.sfName, self.pTab.size())
            
    def PrintRowRange(self):
        sfName = f'Tube{self.iTube}_{self.sName}_rowsRange_margins{detMargins}.csv'
        sfName = path.join(sTableDirPrints, sfName)
        with open(sfName, 'w') as file:
            file.write('row, min, max\n')
            for i in range(nRows):
                row = self.pTab[i]
                minVal = row.min()
                maxVal = row.max()
                if i < 10:
                    print(f'{i}, {minVal:.2f}, {maxVal:.2f}')
                file.write(f'{i}, {minVal:.2f}, {maxVal:.2f}\n')
        print('Saved: ', sfName)
            
    def PrintColRange(self):
        sfName = f'Tube{self.iTube}_{self.sName}_colsRange_margins{detMargins}.csv'
        sfName = path.join(sTableDirPrints, sfName)
        with open(sfName, 'w') as file:
            file.write('row, min, max\n')

            for i in range(self.nCols):
                col = self.pTab[:,i]
                minVal = col.min()
                maxVal = col.max()
                if i < 5:
                    print(f'{i}, {minVal:.2f}, {maxVal:.2f}')
                file.write(f'{i}, {minVal:.2f}, {maxVal:.2f}\n')
        print('Saved: ', sfName)

class CInputTables:
    """
    Read and analyze 2 input tables per tube
    """
    def __init__(self,iTube):
        """
        """
        self.iTube = iTube
        self.imTab = CInputTable(iTube, 'im')
        self.imTab.PrintRowRange()
        self.radTab = CInputTable(iTube, 'rad')
        self.radTab.PrintRowRange()
        self.radTab.PrintColRange()
        self.nCols = self.imTab.nCols

    def FindSource(self,iImage,iRad):
        deltaIm = self.imTab.pTab - iImage
        deltaRad = self.radTab.pTab - iRad
        dist = torch.square(deltaIm) + torch.square(deltaRad)
        #print(f'{dist.size()=}')
        minResult = dist.min()
        ind = dist.argmin()
        iRow = int(ind / self.nCols)
        iCol = ind % self.nCols
        if verbosity > 2:
            print(f'{iImage=}, {iRad=}, {minResult=} at [{iRow}, {iCol}]')
            image = self.imTab.pTab[iRow,iCol]
            radius = self.radTab.pTab[iRow,iCol]
            print(f'Original: {image=:.2f}, {radius=:.2f}')
        return iRow, iCol
        
def SaveInverseTables(iTube, tab, sName):
    spName = f'InverseIR_Tube{iTube}_{sName}_per_image_and_radius_width{maxRadius}_height{nImages}_dzoom2.short.rtab'
    sfName = path.join(sTableDirPrints, spName)
    
    npTable = tab.numpy()
    with open (sfName, 'wb') as file:
        file.write(npTable.tobytes())
    if verbosity > 0:
        print('Table Saved:', sfName)
        
def CreateInverseTables(iTube):
    print('<CreateInverseTables>', iTube)
    rowTab = torch.zeros([nImages, maxRadius], dtype=torch.int16)
    colTab = torch.zeros([nImages, maxRadius], dtype=torch.int16)
    inTabs = CInputTables(iTube)
    iFirstIm = 0
    iFirstRad = 0
    for iIm in range (nImages):
        if iIm % 28 == 0:
            print('image', iIm)
        for iR in range(maxRadius):
            iImage = iFirstIm + iIm
            iRad = iFirstRad + iR
            iRow, iCol = inTabs.FindSource(iImage,iRad)
            rowTab[iImage,iRad] = iRow
            colTab[iImage,iRad] = iCol
            #print(f'{iImage}, {iRad} <== {iRow}, {iCol}')
            
    SaveInverseTables(iTube, rowTab, 'row')
    SaveInverseTables(iTube, colTab, 'col')

def CheckFindSource():
    global verbosity
    print('*** ===>>> Check Find Source')
    verbosity = 5
    iTube = 0
    inTabs = CInputTables(iTube)
    for im in range(2):
        for r in range(2):
            inTabs.FindSource(100+im,20+r)
    verbosity = 1
    
def main():
    global sTableDirPrints
    print('*** ===>>> Create inverse impulse response table')
    sTableDirPrints = VerifyJointDir(sTableDir, 'Prints')
    CheckFindSource()
    for iTube in range(1):
        CreateInverseTables(iTube+1)
    
if __name__ == '__main__':
    main()


