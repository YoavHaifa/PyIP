# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:41:40 2025
Target CSV - follow all changes to specific target point
@author: yoavb
"""

import Config
from CsvLog import CCsvLog

class CTargetCSV:
    """
    """
    def __init__(self, iImage, iRad):
        self.iImage = iImage
        self.iRad = iRad
        sfName = f'Target_i{iImage}_r{iRad}.csv'
        self.csv = CCsvLog(sfName, 'i, tabR, tabD, im, rad, pre, tabVal, delta, imRadVal', sSubDir='target')
        
    def Trace(self, devMap, tvSet):
        self.csv.StartNewLine()
        tv = tvSet.SelectTV(self.iImage, self.iRad)
        self.csv.AddItem(tv.iRow)
        self.csv.AddItem(tv.iDet)
        self.csv.AddItem(tv.iIm)
        self.csv.AddItem(tv.iRad)
        self.csv.AddItem(tv.prevTabValue)
        self.csv.AddItem(tv.tabValue)
        self.csv.AddItem(tv.tabValue - tv.prevTabValue)
        self.csv.AddLastItem(devMap[self.iImage, self.iRad])

class CTargetCSVs:
    """
    All targets to follow
    """
    def __init__(self, goal):
        """
        """
        self.goal = goal
        self.iImage = 163
        self.targets = []
        for iRad in range(50):
            csv = CTargetCSV(self.iImage, iRad)
            self.targets.append(csv)

    def Trace(self, devMap, tvSet):
        for csv in self.targets:
            csv.Trace(devMap, tvSet)
            
    def Print(self):
        for csv in self.targets:
            print('CSV started:', csv.csv.sfName)


def main():
    Config.OnInitRun()
    print('*** Test Target CSV')
    #csv = CTargetCSV(163, 30)
    #print('CSV started:', csv.csv.sfName)
    csvs = CTargetCSVs(10)
    csvs.Print()

    
if __name__ == '__main__':
    main()
