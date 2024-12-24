# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:16:35 2024
Log CSV for analyzing training progress
@author: yoavb
"""

import Config

debug = 15

class CCsvLog():
    """
    """
    def __init__(self, sfName, sTitle):
        """
        """
        self.count = 0
        self.nElements = sTitle.count(',') + 1
        self.nInCurLine = 0
        self.sfName = Config.LogFileName(sfName)
        self.nWarnings = 0
        
        with open(self.sfName, 'w') as f:
            f.write(f'{sTitle}\n')

    def StartNewLine(self):
        self.count += 1
        self.s = f'{self.count}'
        self.nInCurLine = 1
        
    def AddItem(self, value):
        if debug:
            print('<AddItem>', value)
        if self.nInCurLine == 0:
            self.StartNewLine()
        self.s = self.s + f', {value}'
        self.nInCurLine += 1
        
    def AddLastItem(self, value):
        if debug:
            print('<AddLastItem>', value)
        self.AddItem(value)
        if self.nInCurLine != self.nElements:
            self.nWarnings += 1
            if self.nWarnings < 3:
                print(f'<CCsvLog::AddLastItem> WARNING {self.nWarnings}: n {self.nInCurLine} != {self.nElements} expected')
                print(f'<CCsvLog::AddLastItem> WARNING {self.nWarnings}: last line is {self.s}')
        
        with open(self.sfName, 'a') as f:
            f.write(f'{self.s}\n')
            
        self.s = ''
        self.nInCurLine = 0

gCsvLog = CCsvLog('Train_tab_value.csv', 'i, tab, g00, g01, g10, g11, d00, d01, d10, d11, score')


def main():
    Config.OnInitRun()
    print('*** Test Log')
    csv = CCsvLog('i, dev, loss')
    csv.StartNewLine()
    csv.AddItem(-1.5)
    csv.AddLastItem(1.5)
    #csv.StartNewLine()
    csv.AddItem(-1)
    csv.AddLastItem('1')
    
    csv.AddItem(-0.3)
    csv.AddItem(-0.33)
    csv.AddLastItem('0.3')
    
    
if __name__ == '__main__':
    main()
