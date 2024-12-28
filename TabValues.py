# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:31:13 2024
Group of tab values to teach together
@author: yoavb
"""

from os import path

import Config
from TabValue import CTabValue
from TrainEnv import CTrainEnv
from CsvLog import CCsvLog
from Utils import GetAbortFileName
sfAbort = GetAbortFileName()


# Select section to train
iTube = 0
iFirstRow = 67 
iLastRow = 67 
iFirstDet = 307
iLastDet = 346
detInc = 4
detOffset = 170

setId = 0

firstStepAmplitude = 0.001

verbosity = 1


class CTabValues:
    """
    """
    def __init__(self, iTube, iRow, iFirstDet, iLastDet, env):
        """
        """
        global setId
        setId += 1
        self.id = setId
        self.vals = []
        iAfter = iLastDet+1
        iLastUsed = iFirstDet
        sTitle = 'i, '
        for iDet in range (iFirstDet, iAfter, detInc):
            iLastUsed = iDet
            iDetInDF = iDet - detOffset
            tabVal = CTabValue(0, env.df.iloc[iDetInDF])
            tabVal.SetDevFromEnv(env)
            self.vals.append(tabVal)
            sTitle += f'd{iDet}, '
        self.n = len(self.vals)
        self.score = 1000

        self.signature = f'set{self.id}_t{iTube}_r{iRow}_d{iFirstDet}_{iLastUsed}'
        sfName = f'Train_set_{self.signature}.csv'
        sTitle += 'all'
        self.csv = CCsvLog(sfName, sTitle)

            
    def ShortPrint(self):
        print(f'<CTabValues> {self.id}: score {self.score:.6f}')
        for tv in self.vals:
            tv.ShortPrint()
    
    def Run(self, env):
        env.tableGenerator.SaveTable(0) # NOTE: iTube
        env.RunNextTable()
        self.score = 0
        self.csv.StartNewLine()
        for tv in self.vals:
            tv.ComputeGrad2(env)
            self.score += tv.score
            self.csv.AddItem(tv.score)
        self.score /= self.n
        self.csv.AddLastItem(self.score)
        
    def FirstStep(self, env):
        for tv in self.vals:
            tv.AdjustTableLocally(env.tableGenerator, firstStepAmplitude)
        self.Run(env)
        
    def RetraceBadSteps(self, env):
        nChanged = 0
        for tv in self.vals:
            if tv.SelectLastOrPrev(env.tableGenerator):
                nChanged += 1
        if nChanged > 0:
            self.Run(env)
            
    def NextStep(self, env):
        for tv in self.vals:
            tv.SetNextStep(env.tableGenerator)
        self.Run(env)
        
        
        
"""
def RunSecondTableValues(tabValues, scorer, sample, tableGenerator):
    print('*** <RunSecondTable>')
    for tv in tabValues:
        tv.LogNoGrad()
        tv.AdjustTableLocally(tableGenerator, firstStepAmplitude)
    tableGenerator.SaveTable(0) # NOTE: iTube
            
    RunAiRecon('SecondTable')
    scorer.ComputeNewScoreOfVolume12(Config.sfVolumeAi, sample)
        
    secondDevMap = scorer.devRaster.dev.clone()
    return secondDevMap
    """
            

class CTabValSets:
    """
    """
    def __init__(self):
        """
        """
        self.env = CTrainEnv()
        self.sets = []
        sTitle = 'i, '
        for iSet in range(detInc):
            i1stDet = iFirstDet+iSet
            tvSet = CTabValues(iTube, iFirstRow, i1stDet, iLastDet, self.env)
            self.sets.append(tvSet)
            sTitle += f's{tvSet.id}, '
            
        self.n = len(self.sets)
        self.signature = f'metaset_t{iTube}_r{iFirstRow}_d{iFirstDet}_{iLastDet}'
        sfName = f'Train_{self.signature}.csv'
        sTitle += 'all'
        self.csv = CCsvLog(sfName, sTitle)
        self.count = 0
        
    def ShortPrint(self):
        for s in self.sets:
            s.ShortPrint()
            
    def FirstStep(self):
        for s in self.sets:
            s.FirstStep(self.env)
            if verbosity > 2:
                s.ShortPrint()
            s.RetraceBadSteps(self.env)
            if verbosity > 2:
                s.ShortPrint()
            
    def Train1(self):
        self.count += 1
        self.csv.StartNewLine()
        self.score = 0
        for s in self.sets:
            s.NextStep(self.env)
            if verbosity > 2:
                s.ShortPrint()
            self.score += s.score
            self.csv.AddItem(s.score)
            
        self.score /= self.n
        self.csv.AddLastItem(self.score)
        print(f'<CTabValSets::Train> {self.count}: {self.score:.6f}')
        
        if self.count == 1 or self.count % 10 == 0:
            self.env.SaveDevMap(f'step{self.count}')
            Config.SaveLastBpOutput(self.count, zAt = 'Multiset_Training')
            
    def Train(self, n):
        for i in range(n):
            self.Train1()
        
            if path.exists(sfAbort):
                print('Aborting...')
                break   
        
    
def main():
    print('*** Test training of tab groupss')
    sets = CTabValSets()
    print(sets)
    sets.ShortPrint()
    sets.FirstStep()
    sets.Train(100)
    
    
    
if __name__ == '__main__':
    main()
