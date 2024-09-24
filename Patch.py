# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:57:01 2024
C Patch - patches to be applied to poly tables
@author: yoavb
"""

import torch
import math
import random

import Config
from Log import CLog

nDetectors = 688
nRows = 192

verbosity = 1

class CPatch:
    """
    Base class for different patches
    """
    def __init__(self, sName):
        self.sName = sName
        self.iFirstRow = 0
        self.iFirstCol = 0
        self.nTry = 0
        self.nBetter = 0
        self.sumBetter = 0
        self.nEdge = 0

    def OnBetter(self, d):
        self.nBetter += 1
        self.sumBetter += d
   

class CCircularPatch(CPatch):
    """
    Class Patch to add to Poly Table
    """
    def __init__(self, radius):
        """
        """
        CPatch.__init__(self, f'Patch{radius}')
        self.radius = radius
        self.diameter = radius * 2
        self.side = self.diameter + 2
        self.raster = torch.zeros([self.side, self.side])
        
        #Fill each pixel within radius distance from center with 1/distance
        centerX = radius - 0.5
        centerY = radius - 0.5
        
        for iLine in range(self.side):
            dy = centerY - iLine
            dy2 = dy * dy
            for iCol in range(self.side):
                dx = centerX - iCol
                dx2 = dx * dx
                distance = math.sqrt(dx2+dy2)
                if distance < radius:
                    self.raster[iLine, iCol] =1 - distance / radius

    def Add(self, table, delta):
        self.nTry += 1
        if verbosity > 1:
            print(f'Add Circular {self.nTry}, {delta=}')
            print(f'table[{self.iFirstRow}:{self.iLastRow}, {self.iFirstCol}:{self.iLastCol}]')
            print(f'patch[{self.iFirstRowInPatch}:{self.iLastRowInPatch}, {self.iFirstColInPatch}:{self.iLastColInPatch}]')
        add = self.raster[self.iFirstRowInPatch:self.iLastRowInPatch,self.iFirstColInPatch:self.iLastColInPatch] * delta
        table[0,self.iFirstRow:self.iLastRow,self.iFirstCol:self.iLastCol] += add
       
    def AddRandom(self, table, delta, log):
        halfSide = int(self.side / 2)
        bEdge = False
        self.iCenterRow = random.randint(0, nRows-1)
        self.iCenterCol = random.randint(0, nDetectors-1)
        if log:
            log.Log(f'<Circ::AddRandom> center [{self.iFirstRow},{self.iCenterCol}] {delta=:.9f}')
            
        self.iFirstRow = self.iCenterRow - halfSide
        if self.iFirstRow < 0:
            self.iFirstRowInPatch = -self.iFirstRow
            self.iFirstRow = 0
            bEdge = True
        else:
            self.iFirstRowInPatch = 0
            
        self.iFirstCol = self.iCenterCol - halfSide
        if self.iFirstCol < 0:
            self.iFirstColInPatch = -self.iFirstCol
            self.iFirstCol = 0
            bEdge = True
        else:
            self.iFirstColInPatch = 0
              
        self.iLastRow = self.iCenterRow + halfSide
        if self.iLastRow > nRows:
            delta = self.iLastRow - nRows
            self.iLastRowInPatch = -delta
            self.iLastRow = nRows
            bEdge = True
        else:
            self.iLastRowInPatch = self.side
              
        self.iLastCol = self.iCenterCol + halfSide
        if self.iLastCol > nDetectors:
            delta = self.iLastCol - nDetectors
            self.iLastColInPatch = -delta
            self.iLastCol = nDetectors
            bEdge = True
        else:
            self.iLastColInPatch = self.side

        if bEdge:
            self.nEdge += 1
            if self.nEdge <= 10:
                print(f'===>>> Edge {self.nEdge}')
        self.Add(table, delta)

    def Dump(self):
        sfName = f'Patch_radius{self.radius}'
        Config.WriteMatrixToFile(self.raster, sfName)

"""
class CRectangularPatch(CPatch):

    def __init__(self):
        CPatch.__init__(self, 'Rectangle')
        self.iRowAfter = 1
        self.iColAfter = 1 
        self.delta = 0
        
    def Add(self, table, delta):
        table[0,self.iFirstRow:self.iRowAfter,self.iFirstCol:self.iColAfter] += delta
        self.nTry += 1
        if verbosity > 1:
            print(f'Add Rectangle {self.nTry}: rows {self.iFirstRow}:{self.iRowAfter}, cols {self.iFirstCol}:{self.iColAfter}, {delta=}')
        
    def AddRandom(self, table, delta, log):
        self.iFirstRow = random.randint(0, nRows-20)
        self.iRowAfter = random.randint(self.iFirstRow+1, nRows)
        self.iFirstCol = random.randint(0, nDetectors-20)
        self.iColAfter = random.randint(self.iFirstCol+1, nDetectors)
        if log:
            log.Log(f'<Rect::AddRandom> first [{self.iFirstRow},{self.iFirstCol}] after [{self.iRowAfter},{self.iColAfter}] {delta=:.9f}')
        self.Add(table, delta)
        #return 'R-Rectangle'
        """

def main():
    global verbosity
    Config.OnInitRun()
    log = CLog('TestPatch')
    verbosity = 3
    print('*** Test Patch - Circular and Rectangular')
    circ = CCircularPatch(200)
    circ.Dump()
    circ = CCircularPatch(10)
    circ.Dump()
    #rect = CRectangularPatch()
    tab = torch.ones([1,nRows,nDetectors])
    for i in range(100):
        delta = (random.random() - 0.5) / 100
        circ.AddRandom(tab, delta, log)
        sfName = f'd:/dump/PatcTest_table_{i}'
        Config.WriteMatrixToFile(tab, sfName)


if __name__ == '__main__':
    main()
