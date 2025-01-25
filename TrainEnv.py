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
from TargetCsv import CTargetCSVs

sIRDir = 'D:/PolyCalib/Impulse/Impulse_r67_1_74_c170_1_386'
sfIR = 'Tube0_IR_grid_r67_d170.csv'

verbosity = 1

class CTrainEnv:
    """
    """
    def __init__(self):
        """
        """

        print('<CTrainEnv::__init__>')
        VeifyReconRunning()
        Config.OnInitRun()
        
        print('*** Read Impulse Response Data')
        self.ReadData()
        
        self.scorer = CPolyScorer()
        radIm = CRadiusImage()
        sfVol = Config.GetLocalOrSharedVol(Config.sfVolumeNominal)
        originalVol = CVolume('nominalVol', sfVol)
        maskVol = CMaskVolume(originalVol)
        self.sample = CSample(maskVol, radIm)
        self.tableGenerator = CPolyTables() # Prepares initial table
            
        self.RunInitialTable()
        Config.WriteDevToFile(self.initialDevMap, 'flatTab_initial')
        
        self.keptDevMap = None

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
        self.targetCsvs = CTargetCSVs(self.scorer.targetAverage)
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

    def SaveDevMap(self, sAt):
        sfName = f'DevMap_{sAt}'
        Config.WriteDevToFile(self.devMap, sfName)

    def SaveVolume(self, sAt):
        sfName = f'Volume_{sAt}'
        Config.WriteVolToFile(self.scorer.vol.pImages, sfName)
        
    def SaveVolumeAndDevMap(self, sAt):
        self.SaveDevMap(sAt)
        self.SaveVolume(sAt)
        
    def SaveAll(self, sAt):
        self.tableGenerator.SaveTablesAt(sAt)
        self.SaveDevMap(sAt)
        self.SaveVolume(sAt)
        
    def KeepLastResultsForFutureSave(self, sAt):
        self.sKeptAt = sAt
        self.keptDevMap = self.devMap.clone()
        self.keptVolume = self.scorer.vol.pImages.clone()

    def SaveKeptResults(self):
        if self.keptDevMap is None:
            return
        
        sfName = f'DevMapKept_{self.sKeptAt}'
        Config.WriteDevToFile(self.keptDevMap, sfName)
        
        sfName = f'VolumeKept_{self.sKeptAt}'
        Config.WriteVolToFile(self.keptVolume, sfName)
    
def main():
    print('*** Test training environment')
    env = CTrainEnv()
    print(env)
    
if __name__ == '__main__':
    main()

