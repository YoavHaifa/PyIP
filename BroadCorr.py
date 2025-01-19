# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:19:26 2025
Broad Correction - Try to move whole table points together
@author: yoavb
"""

#import Config
#from RunRecon import RunAiRecon, VeifyReconRunning
from RunRecon import VeifyReconRunning
from TrainEnv import CTrainEnv
from IRUse import gIR1

nRows = 192
iTube = 1

class CBroaCorr:
    """
    """
    def __init__(self):
        """
        """
        print('<CBroaCorr::__init__>')
        self.env = CTrainEnv()
        self.env.SaveVolumeAndDevMap('BroadCorr_initial')
        self.count = 0
        
       
       
    def Correct1(self, iRow, iDet, fraction):
        iIm, iRad = gIR1.GetTragetI(iRow, iDet)
        dev = self.env.devMap[iIm:iIm+2, iRad:iRad+2].mean().item()
        delta = - dev * fraction / 1000
        self.env.tableGenerator.AddDeltaLocally(iTube, iRow, iDet, delta)
        
       
    def Correct(self, fraction):
        self.count += 1
        for iRow in range(2,190):
            for iDet in range(170,518):
                self.Correct1(iRow, iDet, fraction)
        self.env.tableGenerator.SaveTable(iTube) # NOTE: iTube
        self.env.RunNextTable()
        self.env.SaveVolumeAndDevMap(f'BroadCorr{self.count}')
            

def main():
    VeifyReconRunning()
    print('*** Test Broad Correction')
    bc = CBroaCorr()
    bc.Correct(0.1)
    print(bc)
    

if __name__ == '__main__':
    main()

