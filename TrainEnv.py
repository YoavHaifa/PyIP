# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:38:26 2024
Tools for training the tables
@author: yoavb
"""

from os import path
import sys
import pandas as pd

import Config
from PolyTable import CPolyTables
from RunRecon import RunAiRecon, VeifyReconRunning
from PolyScorer import CPolyScorer
from Sample import CSample
from RadiusImage import CRadiusImage
from MaskVolume import CMaskVolume
from Volume import CVolume

sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_70_c170_1_346'
sfIR = 'Tube0_IR_grid_r67_d170.csv'

verbosity = 1

class CTrainEnv:
    """
    """
    def __init__(self):
        """
        """

        VeifyReconRunning()
        Config.OnInitRun()
        
        print('*** Read Impulse Response Data')
        self.ReadData()
        
        self.scorer = CPolyScorer()
        radIm = CRadiusImage()
        originalVol = CVolume('nominalVol', Config.sfVolumeNominal)
        maskVol = CMaskVolume(originalVol)
        self.sample = CSample(maskVol, radIm)
        self.tableGenerator = CPolyTables() # Prepares initial table
            
        self.RunInitialTable()
        Config.WriteDevToFile(self.initialDevMap, 'flatTab_initial')

    def ReadData(self):
        sfName = path.join(sIRDir, sfIR)
        print('<ReadData>', sfName)
        if not path.isfile(sfName):
            print('Missing file:', sfName)
            sys.exit()
            
        self.df = pd.read_csv(sfName)
    
        # Display the DataFrame
        #print(df)  

    def RunInitialTable(self):
        print('*** <RunInitialTable>')
        RunAiRecon('InitialTable')
        self.scorer.OldScore(Config.sfVolumeAi, self.sample, bSikpFirst=True)
        self.scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, self.sample)
          
        self.initialDevMap = self.scorer.devRaster.dev.clone()
        self.devMap = self.initialDevMap
        
    def RunNextTable(self):
        self.prevDev = self.devMap
        if verbosity > 2:
            print('*** <RunNextTable>')
               
        RunAiRecon('NextTable')
        self.scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, self.sample)
            
        self.devMap = self.scorer.devRaster.dev.clone()

    def SaveDevMap(self, zAt):
        sfName = f'DevMap_{zAt}'
        Config.WriteDevToFile(self.devMap, sfName)
    
        
    
def main():
    print('*** Test training environment')
    env = CTrainEnv()
    print(env)
    
if __name__ == '__main__':
    main()

