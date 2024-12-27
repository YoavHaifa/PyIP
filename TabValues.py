# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:31:13 2024
Group of tab values to teach together
@author: yoavb
"""

from TabValue import CTabValue
from TrainEnv import CTrainEnv


# Select section to train
iFirstRow = 67 
iLastRow = 67 
iFirstDet = 335
iLastDet = 346
detInc = 4
detOffset = 170

setId = 0

firstStepAmplitude = 0.001


class CTabValues:
    """
    """
    def __init__(self, iRow, iFirstDet, iLastDet, env):
        """
        """
        global setId
        setId += 1
        self.id = setId
        self.vals = []
        for iDet in range (iFirstDet, iLastDet, detInc):
            iDetInDF = iDet - detOffset
            tabVal = CTabValue(0, env.df.iloc[iDetInDF])
            tabVal.SetDevFromEnv(env)
            self.vals.append(tabVal)
            
    def ShortPrint(self):
        print(f'<CTabValues> {self.id}:')
        for tv in self.vals:
            tv.ShortPrint()
            
    def FirstStep(self, env):
        for tv in self.vals:
            tv.AdjustTableLocally(env.tableGenerator, firstStepAmplitude)
        env.tableGenerator.SaveTable(0) # NOTE: iTube
        env.RunNextTable()
        for tv in self.vals:
            tv.ComputeGrad2(env)
        
    def RetraceBadSteps(self):
        for tv in self.vals:
            tv.SelectLastOrPrev()
            
    def NextStep(self, env):
        for tv in self.vals:
            tv.SetNextStep(env.tableGenerator)
        env.tableGenerator.SaveTable(0) # NOTE: iTube
        env.RunNextTable()
        for tv in self.vals:
            tv.ComputeGrad2(env)
        
        
        
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
        tvSet = CTabValues(iFirstRow, iFirstDet, iLastDet, self.env)
        self.sets.append(tvSet)
        
    def ShortPrint(self):
        for s in self.sets:
            s.ShortPrint()
            
    def FirstStep(self):
        for s in self.sets:
            s.FirstStep(self.env)
            s.ShortPrint()
            s.RetraceBadSteps()
            s.ShortPrint()
            
    def Train1(self):
        for s in self.sets:
            s.NextStep(self.env)
            s.ShortPrint()
            
    def Train(self, n):
        for i in range(n):
            self.Train1()
        
    
def main():
    print('*** Test training of tab groupss')
    sets = CTabValSets()
    print(sets)
    sets.ShortPrint()
    sets.FirstStep()
    sets.Train(2)
    
    
    
if __name__ == '__main__':
    main()
