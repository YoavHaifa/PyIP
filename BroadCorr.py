# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:19:26 2025
Broad Correction - Try to move whole table points together
@author: yoavb
"""

import torch

import Config
#from RunRecon import RunAiRecon, VeifyReconRunning
from RunRecon import VeifyReconRunning
from TrainEnv import CTrainEnv
from IRUse import gIR1

nRows = 192
nDets = 688
iTube = 1

def SafeDiv(a, b):
    good_inds = (b != 0)
    fill_value = 0 # or whatever
    c = (
        torch.where(good_inds, a, fill_value) /
        torch.where(good_inds, b, 1)
        )
    return c

def AvoidSmallNumbers(t, minPos=0.0001):
    #minPos = 0.0001
    bPos = t > 0
    bSmall = t < minPos
    #print(f'{bSmall=}')
    bSmallPos = torch.logical_and(bPos, bSmall)
    #print(f'{bSmallPos=}')
    t1 = torch.where(bSmallPos, minPos, t)
    bNeg = t1 < 0
    bSmall = t > -minPos
    bSmallNeg = torch.logical_and(bNeg, bSmall)
    t2 = torch.where(bSmallNeg, -minPos, t1)
    return t2


class CBroaCorr:
    """
    """
    def __init__(self):
        """
        """
        print('<CBroaCorr::__init__>')
        self.env = CTrainEnv()
        self.env.SaveAll('BroadCorr_initialFlat')
        self.count = 0
        self.tabDev = torch.zeros([nRows, nDets])

    def ComputeTVDev(self, iRow, iDet):
        iIm, iRad = gIR1.GetTragetI(iRow, iDet)
        dev = self.env.devMap[iIm:iIm+2, iRad:iRad+2].mean().item()
        self.tabDev[iRow, iDet] = dev
        #delta = - dev * fraction / 1000
        #self.env.tableGenerator.AddDeltaLocally(iTube, iRow, iDet, delta)
        
    def ComputeTabDev(self):
        self.prevTabDev = self.tabDev
        self.tabDev = torch.zeros([nRows, nDets])
        for iRow in range(2,190):
            for iDet in range(170,518):
                self.ComputeTVDev(iRow, iDet)
        Config.WriteMatrixSpec(self.tabDev, f'TabDev{self.count}', 'TabDev')

    def FirstCorrect(self, fraction):
        self.count += 1
        self.ComputeTabDev()
        
        self.deltaTab = self.tabDev * (- fraction / 1000)
        Config.WriteMatrixSpec(self.deltaTab, f'DeltaTab{self.count}_first', 'DeltaTab')
        
        self.env.tableGenerator.AddDeltaToTable(iTube, self.deltaTab)
        self.env.RunNextTable()
        self.env.SaveAll(f'BroadCorr{self.count}_first')


    def Correct(self, fraction):
        self.count += 1
        self.prevTabDev = self.tabDev.clone()
        self.ComputeTabDev()
        
        deltaDev = self.tabDev - self.prevTabDev
        Config.WriteMatrixSpec(deltaDev, f'DeltaTabDev{self.count}', 'DeltaTabDev')
        
        prevDeltaTabClipped = AvoidSmallNumbers(self.deltaTab)
        Config.WriteMatrixSpec(prevDeltaTabClipped, f'PrevDeltaTabClipped{self.count}', 'PrevDeltaTabClipped')
        
        grad = SafeDiv(deltaDev, prevDeltaTabClipped)
        Config.WriteMatrixSpec(grad, f'DevGrad{self.count}', 'DevGrad')

        #gradClipped = AvoidSmallNumbers(grad)
        #Config.WriteMatrixSpec(gradClipped, f'GradClipped{self.count}', 'GradClipped')
        
        self.deltaTab = SafeDiv(self.tabDev, grad) * ( - fraction)
        Config.WriteMatrixSpec(self.deltaTab, f'DeltaTabRaw{self.count}', 'DeltaTabRaw')
        
        maxDeltaAmp = 0.001
        self.deltaTab = torch.clamp(self.deltaTab, min=-maxDeltaAmp, max=maxDeltaAmp)
        Config.WriteMatrixSpec(self.deltaTab, f'DeltaTab{self.count}', 'DeltaTab')
        
        self.env.tableGenerator.AddDeltaToTable(iTube, self.deltaTab)
        self.env.RunNextTable()
        self.env.SaveAll(f'BroadCorr{self.count}')
            

def main():
    VeifyReconRunning()
    print('*** Test Broad Correction')
    bc = CBroaCorr()
    bc.FirstCorrect(0.001)
    print(bc)
    bc.Correct(0.1)
    

if __name__ == '__main__':
    main()

