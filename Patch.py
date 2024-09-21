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
        side = self.side
        add = self.raster * delta
        print(f'{add.shape=}')
        print(f'{table.shape=}')
        table[0,self.iFirstRow:self.iFirstRow+side,self.iFirstCol:self.iFirstCol+side] += add
        self.nTry += 1
        if verbosity > 1:
            print(f'Add Circular {self.nTry}: rows {self.iFirstRow}, cols {self.iFirstCol}, {delta=}')
        return table
       
    def AddRandom(self, table, delta, log):
        side = self.side
        self.iFirstRow = random.randint(0, nRows-side-1)
        self.iFirstCol = random.randint(0, nDetectors-side-1)
        if log:
            log.Log(f'<Circ::AddRandom> iFirst [{self.iFirstRow},{self.iFirstCol}] {delta=:.9f}')
        return self.Add(table, delta)
        #return f'R-Circ{self.radius}'

    def Dump(self):
        sfName = f'Patch_radius{self.radius}'
        Config.WriteMatrixToFile(self.raster, sfName)

class CRectangularPatch(CPatch):
    """
    """
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
        return table
        
    def AddRandom(self, table, delta, log):
        self.iFirstRow = random.randint(0, nRows-20)
        self.iRowAfter = random.randint(self.iFirstRow+1, nRows)
        self.iFirstCol = random.randint(0, nDetectors-20)
        self.iColAfter = random.randint(self.iFirstCol+1, nDetectors)
        if log:
            log.Log(f'<Rect::AddRandom> first [{self.iFirstRow},{self.iFirstCol}] after [{self.iRowAfter},{self.iColAfter}] {delta=:.9f}')
        return self.Add(table, delta)
        #return 'R-Rectangle'
       

def main():
    global verbosity
    Config.OnInitRun()
    verbosity = 3
    print('*** Test Patch - Circular and Rectangular')
    circ = CCircularPatch(200)
    circ.Dump()
    circ = CCircularPatch(10)
    circ.Dump()
    rect = CRectangularPatch()
    tab = torch.ones([nRows,nDetectors])
    for i in range(10):
        delta = (random.random() - 0.5) / 100
        iType = random.randint(0,10)
        if iType < 7:
            circ.AddRandom(tab, delta)
        else:
            rect.AddRandom(tab, delta)
        sfName = f'd:/dump/PatcTest_table_{i}'
        Config.WriteMatrixToFile(tab, sfName)


if __name__ == '__main__':
    main()
