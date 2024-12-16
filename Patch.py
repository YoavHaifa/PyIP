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
from Log import CLog, Log

nDetectors = 688
nRows = 192

ratioThreshold = 0.10
deltaFactor = 0.8

verbosity = 1
bLogPatch = False

class CPatchStat:
    """
    Base class for different patches
    """
    def __init__(self, best, delta, iFirst):
        self.firstBest = best
        self.lastBest = best
        self.delta = delta
        self.iFirst = iFirst
        self.nTry = 0
        self.nBetter = 0
        self.nFailed = 0
        self.sumGain = 0
                
    def OnScoreBetter(self, score, gain):
        self.nTry += 1
        self.nBetter += 1
        self.lastBest = score
        self.sumGain += gain

    def OnScoreNotBetter(self):
        self.nTry += 1
        self.nFailed += 1
        
    def BetterRatio(self):
        if self.nTry < 1:
            return 0
        return float(self.nBetter) / self.nTry

    def OnEnd(self, sfLog):
        ratio = self.BetterRatio()
        with open(sfLog, 'a') as f:
            s = f'{self.delta}, {self.iFirst}, {self.nTry}, {self.nBetter}, {self.firstBest}, {self.lastBest}, {ratio:.2f}'
            f.write(f'{s}\n')
            #print(f'<CPatchStat::OnEnd> {s}')
        

class CPatch:
    """
    Base class for different patches
    """
    def __init__(self, sName, best, delta):
        self.sName = sName
        self.firstBest = best
        self.lastBest = best
        self.firstDelta = delta
        self.delta = delta
        
        self.iFirstRow = 0
        self.iFirstCol = 0
        self.nTry = 0
        self.nBetter = 0
        self.nEdge = 0
        self.sumGain = 0
        self.stat = CPatchStat(best, delta, self.nTry)

    def OnNewScore(self, prevBest, score):
        gain = prevBest - score
        if gain > 0:
            self.nBetter += 1
            self.lastBest = score
            self.sumGain += gain
            self.stat.OnScoreBetter(score, gain)
            
            # Log all better transactions
            ratio = self.stat.BetterRatio()
            if bLogPatch:
                with open(self.sfLog, 'a') as f:
                    f.write(f'{self.delta}, {self.nTry}, {self.nBetter}, {score}, {ratio:.2f}\n')
                
        else:
            self.lastBest = prevBest # Might have improved due to another patch/table
            self.stat.OnScoreNotBetter()
            
        if self.stat.nTry == Config.nToEvaluate:
            ratio = self.stat.BetterRatio()
            if ratio < ratioThreshold:
                self.delta = abs(self.delta) * deltaFactor
                print(f'\n<{self.sName}> Delta reduced to {self.delta}')
            if bLogPatch:
                self.stat.OnEnd(self.sfLogStat)
            self.stat = CPatchStat(self.lastBest, self.delta, self.nTry)
  

class CCircularPatch(CPatch):
    """
    Class Patch to add to Poly Table
    """
    def __init__(self, best, radius, delta, iXrt):
        """
        """
        CPatch.__init__(self, f'Patch{radius}-{iXrt}', best, delta)
        
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
                    
        if bLogPatch:
            self.sfLog = Config.LogFileName(f'CircPatchAll_xrt{iXrt}_radius{radius}.csv')
            with open(self.sfLog, 'w') as f:
                f.write('delta, nTry, nBetter, best, ratio\n')
            
        if bLogPatch:
            self.sfLogStat =  Config.LogFileName(f'CircPatchStat_xrt{iXrt}_radius{radius}.csv')
            with open(self.sfLogStat, 'w') as f:
                f.write('delta, iFirst, nTry, nBetter, start, best, ratio\n')

    def Add(self, table):
        self.nTry += 1
        if verbosity > 1:
            print(f'Add Circular {self.nTry}, {self.delta=}')
            print(f'table[{self.iFirstRow}:{self.iLastRow}, {self.iFirstCol}:{self.iLastCol}]')
            print(f'patch[{self.iFirstRowInPatch}:{self.iLastRowInPatch}, {self.iFirstColInPatch}:{self.iLastColInPatch}]')
        Log(f'<Add> [{self.iFirstRowInPatch}:{self.iLastRowInPatch}, {self.iFirstColInPatch}:{self.iLastColInPatch}] delta {self.delta}')
        add = self.raster[self.iFirstRowInPatch:self.iLastRowInPatch,self.iFirstColInPatch:self.iLastColInPatch] * self.delta
        table[0,self.iFirstRow:self.iLastRow,self.iFirstCol:self.iLastCol] += add
       
    def AddOnFailure(self, table):
        self.delta = - self.delta
        self.Add(table)
        
    def AddRandom(self, table, log):
        iRow = random.randint(0, nRows-1)
        iCol = random.randint(0, nDetectors-1)
        if log:
            log.Log(f'<Circ::AddRandom> center [{iRow},{iCol}] {self.delta=:.9f}')
        self.AddAt(table, iRow, iCol)
            
        
    def AddAt(self, table, iRow, iCol):
        self.iCenterRow = iRow
        self.iCenterCol = iCol
        halfSide = int(self.side / 2)
        bEdge = False
        
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
            if verbosity > 1 and self.nEdge <= 10:
                print(f'===>>> Edge {self.nEdge}')
            Log(f'===>>> Edge {self.nEdge}')
        self.Add(table)

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
    circ = CCircularPatch(100, 200, 0.001, 0)
    circ.Dump()
    circ = CCircularPatch(100, 10, 0.002, 1)
    circ.Dump()
    #rect = CRectangularPatch()
    tab = torch.ones([1,nRows,nDetectors])
    for i in range(100):
        #delta = (random.random() - 0.5) / 100
        circ.AddRandom(tab, log)
        sfName = f'd:/dump/PatcTest_table_{i}'
        Config.WriteMatrixToFile(tab, sfName)


if __name__ == '__main__':
    main()
